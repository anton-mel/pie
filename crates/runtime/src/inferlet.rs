//! Runs inferlets: one wasm instance per run, with every WIT interface
//! implemented against the shared engine (one file each in `host/`).

mod host;
mod process;
mod program;

pub use process::{Event, Process, ProcessId, ProcessInfo, Processes};
pub use program::Programs;

use crate::engine::{Engine, Reply, Request};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Engine as Wasm, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

wasmtime::component::bindgen!({
    path: "../inferlet/wit",
    world: "inferlet",
    imports: { default: async },
    exports: { default: async },
    with: {
        "pie:inferlet/working-set.kv-working-set": KvWorkingSet,
        "pie:inferlet/forward.pending-forward": PendingForward,
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
        Inferlet::add_to_linker::<_, HasSelf<_>>(&mut linker, |s| s)?;
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
        Ok(app.pie_inferlet_run().call_run(&mut store, args).await?)
    }
}
