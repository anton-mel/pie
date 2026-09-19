//! `forward`: submit tokens to the model and wait for the result.

use crate::inferlet::pie::inferlet::forward::{self, Distribution};
use crate::inferlet::{KvWorkingSet, PendingForward, State};
use models::Seq;
use wasmtime::component::Resource;

impl forward::Host for State {
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

impl forward::HostPendingForward for State {
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
