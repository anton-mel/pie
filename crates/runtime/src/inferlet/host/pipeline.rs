//! `pipeline`: the order work runs in.

use crate::inferlet::pie::inferlet::pipeline;
use crate::inferlet::{Pipeline, State};
use std::sync::atomic::{AtomicU64, Ordering};
use wasmtime::component::Resource;

static NEXT: AtomicU64 = AtomicU64::new(1);

impl pipeline::Host for State {}

impl pipeline::HostPipeline for State {
    async fn new(&mut self) -> Resource<Pipeline> {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        self.table.push(Pipeline { id }).expect("resource table full")
    }

    async fn drop(&mut self, p: Resource<Pipeline>) -> wasmtime::Result<()> {
        self.table.delete(p)?;
        Ok(())
    }
}
