//! Runs inferlets: one wasm instance per run, with the `model` interface
//! implemented against the shared engine.

use crate::engine::{Engine, Reply, Request};
use crate::model::Seq;
use anyhow::Result;
use pie::core::model::{self, Distribution};
use pie::core::{chat, session};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

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

pub struct PendingForward {
    reply: Option<Reply>,
}

/// Where an inferlet's messages go and come from: a client connection, or
/// the terminal. It outlives restarts by the planner.
#[derive(Clone)]
pub struct Session {
    pub out: mpsc::UnboundedSender<String>,
    pub inbox: Arc<Mutex<mpsc::UnboundedReceiver<String>>>,
}

struct State {
    engine: Arc<Engine>,
    session: Session,
    id: u64,
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
        let pages = self.engine.alloc_wait(self.id, n).await?;
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

    async fn discard(&mut self, ws: Resource<KvWorkingSet>, start: u32, len: u32) -> Result<(), String> {
        let ws = self.table.get_mut(&ws).map_err(|e| e.to_string())?;
        let (start, end) = (start as usize, start as usize + len as usize);
        if end > ws.pages.len() {
            return Err(format!("pages {start}..{end} past the end ({})", ws.pages.len()));
        }
        // A forward still queued on these pages holds them until it has run.
        self.engine.free(ws.pages.drain(start..end));
        Ok(())
    }

    async fn update_index(&mut self, ws: Resource<KvWorkingSet>, key: String) {
        if let Ok(ws) = self.table.get(&ws) {
            self.engine.publish(key, &ws.pages);
        }
    }

    async fn from_index(&mut self, key: String) -> Option<Resource<KvWorkingSet>> {
        let pages = self.engine.open(&key)?;
        let ws = KvWorkingSet {
            engine: self.engine.clone(),
            pages,
        };
        Some(self.table.push(ws).expect("resource table full"))
    }

    async fn remove_index(&mut self, key: String) -> bool {
        self.engine.unpublish(&key)
    }

    async fn drop(&mut self, ws: Resource<KvWorkingSet>) -> wasmtime::Result<()> {
        self.table.delete(ws)?;
        Ok(())
    }
}

impl model::HostPendingForward for State {
    async fn wait(&mut self, p: Resource<PendingForward>) -> Result<Vec<Distribution>, String> {
        // Everything submitted so far goes to the engine together, so it
        // lands in one model step.
        if !self.unsent.is_empty() {
            self.engine.send(std::mem::take(&mut self.unsent))?;
        }
        let pending = self.table.get_mut(&p).map_err(|e| e.to_string())?;
        let reply = pending.reply.take().ok_or("already waited")?;
        let dists = reply.await.map_err(|_| "engine stopped")??;
        Ok(dists
            .into_iter()
            .map(|d| Distribution {
                ids: d.ids,
                probs: d.probs,
            })
            .collect())
    }

    async fn drop(&mut self, p: Resource<PendingForward>) -> wasmtime::Result<()> {
        self.table.delete(p)?;
        Ok(())
    }
}

impl session::Host for State {
    async fn send(&mut self, message: String) {
        let _ = self.session.out.send(message);
    }

    async fn receive(&mut self) -> Option<String> {
        self.session.inbox.lock().await.recv().await
    }
}

/// The chat format of Qwen models (ChatML). Other model families write
/// their turns differently; the reference reads each model's own template.
impl chat::Host for State {
    async fn system(&mut self, message: String) -> Vec<u32> {
        self.turn("system", &message)
    }

    async fn user(&mut self, message: String) -> Vec<u32> {
        self.turn("user", &message)
    }

    async fn assistant(&mut self, message: String) -> Vec<u32> {
        self.turn("assistant", &message)
    }

    async fn cue(&mut self) -> Vec<u32> {
        self.encode("<|im_start|>assistant\n")
    }

    async fn seal(&mut self) -> Vec<u32> {
        self.encode("<|im_end|>\n")
    }

    async fn stop_tokens(&mut self) -> Vec<u32> {
        self.engine.eos.clone()
    }
}

impl State {
    fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.engine.tokenizer.encode(text, false);
        encoding.map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }

    fn turn(&self, role: &str, message: &str) -> Vec<u32> {
        self.encode(&format!("<|im_start|>{role}\n{message}<|im_end|>\n"))
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
        outputs: Vec<u32>,
        allowed: Option<Vec<u32>>,
        top_k: u32,
    ) -> Result<Resource<PendingForward>, String> {
        let ws = self.table.get_mut(&kv).map_err(|e| e.to_string())?;

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

        if outputs.iter().any(|&i| i as usize >= tokens.len()) {
            return Err("output index past the new tokens".into());
        }

        let first = (kv_len - tokens.len() as u32) / ps;
        let shared: Vec<usize> = (first as usize..need)
            .filter(|&i| self.engine.is_shared(ws.pages[i]))
            .collect();

        let fresh = self.engine.alloc_wait(self.id, shared.len() as u32).await?;
        let copies = shared
            .into_iter()
            .zip(fresh)
            .map(|(i, to)| (std::mem::replace(&mut ws.pages[i], to), to))
            .collect();

        let seq = Seq {
            copies,
            tokens,
            positions,
            outputs,
            pages: ws.pages[..need].to_vec(),
            kv_len: kv_len as usize,
        };

        let (request, reply) = self.engine.request(seq, top_k as usize, allowed);

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
        chat::add_to_linker::<_, wasmtime::component::HasSelf<_>>(&mut linker, |s| s)?;
        session::add_to_linker::<_, wasmtime::component::HasSelf<_>>(&mut linker, |s| s)?;
        Ok(Self { wasm, linker, engine })
    }

    pub fn load(&self, path: &str) -> Result<Component> {
        Ok(Component::from_file(&self.wasm, path)?)
    }

    /// Compile an inferlet a client sent.
    pub fn compile(&self, wasm: &[u8]) -> Result<Component> {
        Ok(Component::new(&self.wasm, wasm)?)
    }

    /// Takes the session the inferlet talks over.
    pub async fn run(
        &self,
        component: &Component,
        args: Vec<String>,
        session: Session,
    ) -> Result<Result<String, String>> {
        loop {
            let (id, kill) = self.engine.planner.join();
            let result = tokio::select! {
                r = self.run_once(id, component, &args, session.clone()) => Some(r),
                _ = kill.notified() => None,
            };

            // Dropping the losing branch above dropped the instance, and with
            // it every page it held.
            self.engine.planner.leave(id, result.is_none());
            match result {
                Some(r) => return r,
                None => {
                    eprintln!("inferlet {id} evicted to free KV pages, restarting");
                    self.engine.planner.until_exit().await;
                }
            }
        }
    }

    async fn run_once(
        &self,
        id: u64,
        component: &Component,
        args: &[String],
        session: Session,
    ) -> Result<Result<String, String>> {
        let state = State {
            engine: self.engine.clone(),
            session,
            id,
            unsent: vec![],
            wasi: WasiCtx::builder().inherit_stdio().build(),
            table: ResourceTable::new(),
        };
        let mut store = Store::new(&self.wasm, state);
        let app = Inferlet::instantiate_async(&mut store, component, &self.linker).await?;
        Ok(app.pie_core_run().call_run(&mut store, args).await?)
    }
}
