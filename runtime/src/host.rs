//! Runs inferlets: one wasm instance per run, with the `model` interface
//! implemented against the shared engine.

use crate::engine::Engine;
use crate::model::Seq;
use anyhow::Result;
use pie::core::model::{self, Distribution};
use std::collections::HashSet;
use std::sync::Arc;

use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Engine as Wasm, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

wasmtime::component::bindgen!({
    path: "../wit",
    world: "inferlet",
    imports: { default: async },
    exports: { default: async },
});

struct State {
    engine: Arc<Engine>,
    /// KV pages this instance holds;
    /// checked on every use, freed on exit.
    pages: HashSet<u32>,
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

impl Drop for State {
    fn drop(&mut self) {
        self.engine.free(self.pages.drain());
    }
}

impl model::Host for State {
    async fn kv_page_size(&mut self) -> u32 {
        self.engine.page_size
    }

    async fn alloc_pages(&mut self, n: u32) -> Result<Vec<u32>, String> {
        // Owned until `free_pages` or exit.
        let pages = self.engine.alloc(n).ok_or("out of KV pages")?;
        self.pages.extend(&pages);
        Ok(pages)
    }

    async fn free_pages(&mut self, pages: Vec<u32>) {
        let owned: Vec<u32> = pages.into_iter().filter(|p| self.pages.remove(p)).collect();
        self.engine.free(owned);
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
        pages: Vec<u32>,
        last_page_len: u32,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        top_k: u32,
    ) -> Result<Distribution, String> {
        let ps = self.engine.page_size;
        if let Some(p) = pages.iter().find(|p| !self.pages.contains(p)) {
            return Err(format!("page {p} is not yours"));
        }
        if pages.is_empty() || last_page_len == 0 || last_page_len > ps {
            return Err("bad page geometry".into());
        }
        let kv_len = (pages.len() as u32 - 1) * ps + last_page_len;
        if tokens.is_empty() || tokens.len() != positions.len() || tokens.len() > kv_len as usize {
            return Err("tokens/positions do not fit the pages".into());
        }
        let seq = Seq {
            tokens,
            positions,
            pages,
            kv_len: kv_len as usize,
        };
        let d = self.engine.forward(seq, top_k as usize).await?;
        Ok(Distribution {
            ids: d.ids,
            probs: d.probs,
        })
    }
}

pub struct Host {
    wasm: Wasm,
    linker: Linker<State>,
    engine: Arc<Engine>,
}

impl Host {
    pub fn new(engine: Arc<Engine>) -> Result<Self> {
        // hold wasm and linker here so we can instantiate multiple components with the same engine
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
            pages: HashSet::new(),
            wasi: WasiCtx::builder().inherit_stdio().build(),
            table: ResourceTable::new(),
        };
        let mut store = Store::new(&self.wasm, state);
        let app = Inferlet::instantiate_async(&mut store, component, &self.linker).await?;
        Ok(app.pie_core_run().call_run(&mut store, &args).await?)
    }
}
