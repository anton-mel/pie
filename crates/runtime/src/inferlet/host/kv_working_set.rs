//! `working-set`: a sequence's KV cache as logical pages.

use crate::inferlet::pie::inferlet::working_set;
use crate::inferlet::{KvWorkingSet, State};
use wasmtime::component::Resource;

impl working_set::Host for State {}

impl working_set::HostKvWorkingSet for State {
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
