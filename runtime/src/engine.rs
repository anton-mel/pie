//! The engine: one model, one KV page pool, and a batcher that runs every
//! forward submitted since the last step as one batch.

use crate::model::{Model, Seq};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

pub struct Distribution {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

pub type Reply = oneshot::Receiver<Result<Distribution, String>>;

/// A forward on its way to the model.
pub struct Request {
    seq: Seq,
    top_k: usize,
    reply: oneshot::Sender<Result<Distribution, String>>,
    hold: Hold,
}

/// Pages a queued forward reads or writes. Holding them keeps them from
/// being freed and handed to someone else before the model has run it, even
/// if the inferlet drops its working set in the meantime.
struct Hold {
    pool: Arc<Mutex<Pool>>,
    pages: Vec<u32>,
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.pool.lock().unwrap().free(self.pages.drain(..));
    }
}

pub struct Engine {
    pub tokenizer: Tokenizer,
    pub eos: Vec<u32>,
    pub page_size: u32,
    pool: Arc<Mutex<Pool>>,
    queue: mpsc::UnboundedSender<Vec<Request>>,
}

/// Physical KV pages. A page can be held by several working sets after a
/// fork; `refs` counts them and the page is free again at zero.
struct Pool {
    free: Vec<u32>,
    refs: Vec<u32>,
}

impl Pool {
    /// One less holder for each page; pages nobody holds go back to the pool.
    fn free(&mut self, pages: impl IntoIterator<Item = u32>) {
        for p in pages {
            self.refs[p as usize] -= 1;
            if self.refs[p as usize] == 0 {
                self.free.push(p);
            }
        }
    }
}

impl Engine {
    pub fn new(model: Model, tokenizer: Tokenizer, eos: Vec<u32>, pages: u32) -> Self {
        let page_size = model.page_size as u32;
        let (queue, rx) = mpsc::unbounded_channel();
        std::thread::spawn(move || batch_loop(model, rx));
        Self {
            tokenizer,
            eos,
            page_size,
            pool: Arc::new(Mutex::new(Pool {
                free: (0..pages).rev().collect(),
                refs: vec![0; pages as usize],
            })),
            queue,
        }
    }

    pub fn alloc(&self, n: u32) -> Option<Vec<u32>> {
        let mut pool = self.pool.lock().unwrap();
        let at = pool.free.len().checked_sub(n as usize)?;
        let pages = pool.free.split_off(at);
        pages.iter().for_each(|&p| pool.refs[p as usize] = 1);
        Some(pages)
    }

    /// One more holder for each page.
    pub fn share(&self, pages: &[u32]) {
        let mut pool = self.pool.lock().unwrap();
        pages.iter().for_each(|&p| pool.refs[p as usize] += 1);
    }

    pub fn is_shared(&self, page: u32) -> bool {
        self.pool.lock().unwrap().refs[page as usize] > 1
    }

    pub fn free(&self, pages: impl IntoIterator<Item = u32>) {
        self.pool.lock().unwrap().free(pages);
    }

    /// A request for one forward, and where its result will arrive. It holds
    /// `seq.pages`; for copy sources the caller hands over a hold it has.
    pub fn request(&self, seq: Seq, top_k: usize) -> (Request, Reply) {
        self.share(&seq.pages);
        let mut pages = seq.pages.clone();
        pages.extend(seq.copies.iter().map(|&(from, _)| from));
        let hold = Hold {
            pool: self.pool.clone(),
            pages,
        };
        let (reply, rx) = oneshot::channel();
        (
            Request {
                seq,
                top_k,
                reply,
                hold,
            },
            rx,
        )
    }

    /// Queue requests; they go into the same model step.
    pub fn send(&self, requests: Vec<Request>) -> Result<(), String> {
        self.queue.send(requests).map_err(|_| "engine stopped".into())
    }
}

/// Take whatever is queued, run it as one batch, repeat.
fn batch_loop(mut model: Model, mut rx: mpsc::UnboundedReceiver<Vec<Request>>) {
    while let Some(mut batch) = rx.blocking_recv() {
        while let Ok(more) = rx.try_recv() {
            batch.extend(more);
        }
        let mut seqs = vec![];
        let mut rest = vec![];
        let mut holds = vec![];
        for r in batch {
            seqs.push(r.seq);
            rest.push((r.top_k, r.reply));
            holds.push(r.hold);
        }
        match model.forward(&seqs).and_then(|l| Ok(l.to_vec2::<f32>()?)) {
            Ok(logits) => {
                for (row, (k, reply)) in logits.into_iter().zip(rest) {
                    let _ = reply.send(Ok(top_k(row, k)));
                }
            }
            Err(e) => {
                for (_, reply) in rest {
                    let _ = reply.send(Err(e.to_string()));
                }
            }
        }
        // The model is done with these pages.
        drop(holds);
    }
}

fn top_k(logits: Vec<f32>, k: usize) -> Distribution {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.clamp(1, idx.len());
    let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    idx.select_nth_unstable_by(k - 1, by_logit);
    idx.truncate(k);
    idx.sort_unstable_by(by_logit);
    let probs = idx.iter().map(|&i| (logits[i as usize] - max).exp() / sum).collect();
    Distribution { ids: idx, probs }
}
