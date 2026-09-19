//! Runs inferlets: one wasm instance per run, with the `model` interface
//! implemented against the shared engine.

use crate::engine::{Engine, Reply, Request};
use crate::model::Seq;
use anyhow::Result;
use pie::core::model::{self, Distribution};
use std::sync::Arc;

use wasmtime::component::{Component, Linker, Resource, ResourceTable};
use wasmtime::{Engine as Wasm, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

wasmtime::component::bindgen!({
    path: "../wit",
    world: "inferlet",
    imports: { default: async },
    exports: { default: async },
    with: {
        "pie:core/model.kv-working-set": KvWorkingSet,
        "pie:core/model.pending-forward": PendingForward,
    },
});

pub struct KvWorkingSet {
    engine: Arc<Engine>,
    pages: Vec<u32>,
}

impl Drop for KvWorkingSet {
    fn drop(&mut self) {
        self.engine.free(self.pages.drain(..));
    }
}

/// The host side of a `pending-forward`: where its result will arrive.
pub struct PendingForward {
    reply: Option<Reply>,
}

struct State {
    engine: Arc<Engine>,
    /// Forwards submitted since the inferlet last waited.
    unsent: Vec<Request>,
    wasi: WasiCtx,
    table: ResourceTable,
}

impl WasiView for State {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl model::HostKvWorkingSet for State {
    async fn new(&mut self) -> Resource<KvWorkingSet> {
        let ws = KvWorkingSet {
            engine: self.engine.clone(),
            pages: vec![],
        };
        self.table.push(ws).expect("resource table full")
    }

    async fn page_len(&mut self, ws: Resource<KvWorkingSet>) -> u32 {
        self.table.get(&ws).map_or(0, |ws| ws.pages.len() as u32)
    }

    async fn reserve(&mut self, ws: Resource<KvWorkingSet>, n: u32) -> Result<(), String> {
        let pages = self.engine.alloc(n).ok_or("out of KV pages")?;
        self.table.get_mut(&ws).map_err(|e| e.to_string())?.pages.extend(pages);
        Ok(())
    }

    async fn fork(&mut self, ws: Resource<KvWorkingSet>) -> Resource<KvWorkingSet> {
        let pages = self.table.get(&ws).map_or(vec![], |ws| ws.pages.clone());
        self.engine.share(&pages);
        let child = KvWorkingSet {
            engine: self.engine.clone(),
            pages,
        };
        self.table.push(child).expect("resource table full")
    }

    async fn drop(&mut self, ws: Resource<KvWorkingSet>) -> wasmtime::Result<()> {
        self.table.delete(ws)?;
        Ok(())
    }
}

impl model::HostPendingForward for State {
    async fn wait(&mut self, p: Resource<PendingForward>) -> Result<Distribution, String> {
        // Everything submitted so far goes to the engine together, so it
        // lands in one model step.
        if !self.unsent.is_empty() {
            self.engine.send(std::mem::take(&mut self.unsent))?;
        }
        let pending = self.table.get_mut(&p).map_err(|e| e.to_string())?;
        let reply = pending.reply.take().ok_or("already waited")?;
        let d = reply.await.map_err(|_| "engine stopped")??;
        Ok(Distribution {
            ids: d.ids,
            probs: d.probs,
        })
    }

    async fn drop(&mut self, p: Resource<PendingForward>) -> wasmtime::Result<()> {
        self.table.delete(p)?;
        Ok(())
    }
}

impl model::Host for State {
    async fn kv_page_size(&mut self) -> u32 {
        self.engine.page_size
    }

    async fn tokenize(&mut self, text: String) -> Vec<u32> {
        self.engine
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    async fn detokenize(&mut self, tokens: Vec<u32>) -> String {
        self.engine.tokenizer.decode(&tokens, false).unwrap_or_default()
    }

    async fn eos_tokens(&mut self) -> Vec<u32> {
        self.engine.eos.clone()
    }

    async fn forward(
        &mut self,
        kv: Resource<KvWorkingSet>,
        kv_len: u32,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        top_k: u32,
    ) -> Result<Resource<PendingForward>, String> {
        let ws = self.table.get_mut(&kv).map_err(|e| e.to_string())?;

        // Translate logical pages to physical ones for the pages in use.
        let ps = self.engine.page_size;
        let need = kv_len.div_ceil(ps) as usize;

        if need > ws.pages.len() {
            return Err(format!(
                "kv-len {kv_len} needs {need} pages, working set has {}",
                ws.pages.len()
            ));
        }

        if tokens.is_empty() || tokens.len() != positions.len() || tokens.len() > kv_len as usize {
            return Err("tokens/positions do not fit kv-len".into());
        }

        // Copy-on-write: a page this call writes into and a fork still holds
        // is copied to a fresh page first, and this working set moves to it.
        // The working set's hold on the old page passes to the request, which
        // keeps it until the copy has run.
        let first = (kv_len - tokens.len() as u32) / ps;
        let shared: Vec<usize> = (first as usize..need)
            .filter(|&i| self.engine.is_shared(ws.pages[i]))
            .collect();
        let fresh = self.engine.alloc(shared.len() as u32).ok_or("out of KV pages")?;
        let copies = shared
            .into_iter()
            .zip(fresh)
            .map(|(i, to)| (std::mem::replace(&mut ws.pages[i], to), to))
            .collect();

        let seq = Seq {
            copies,
            tokens,
            positions,
            pages: ws.pages[..need].to_vec(),
            kv_len: kv_len as usize,
        };

        let (request, reply) = self.engine.request(seq, top_k as usize);
        self.unsent.push(request);
        let pending = PendingForward { reply: Some(reply) };
        Ok(self.table.push(pending).map_err(|e| e.to_string())?)
    }
}

pub struct Host {
    wasm: Wasm,
    linker: Linker<State>,
    engine: Arc<Engine>,
}

impl Host {
    pub fn new(engine: Arc<Engine>) -> Result<Self> {
        let wasm = Wasm::default();
        let mut linker = Linker::new(&wasm);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        model::add_to_linker::<_, wasmtime::component::HasSelf<_>>(&mut linker, |s| s)?;
        Ok(Self { wasm, linker, engine })
    }

    pub fn load(&self, path: &str) -> Result<Component> {
        Ok(Component::from_file(&self.wasm, path)?)
    }

    pub async fn run(&self, component: &Component, args: Vec<String>) -> Result<Result<String, String>> {
        let state = State {
            engine: self.engine.clone(),
            unsent: vec![],
            wasi: WasiCtx::builder().inherit_stdio().build(),
            table: ResourceTable::new(),
        };
        let mut store = Store::new(&self.wasm, state);
        let app = Inferlet::instantiate_async(&mut store, component, &self.linker).await?;
        Ok(app.pie_core_run().call_run(&mut store, &args).await?)
    }
}
