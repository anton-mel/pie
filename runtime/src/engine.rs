//! The engine: one model, one KV page pool, and a queue of forwards that
//! the scheduler (`scheduler.rs`) turns into model steps.

use crate::model::{Model, Seq};
use crate::planner::Planner;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::{Notify, mpsc, oneshot};

pub struct Distribution {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

pub type Reply = oneshot::Receiver<Result<Vec<Distribution>, String>>;

/// UPDATED
/// Its fields are read by the scheduler.
pub struct Request {
    pub seq: Seq,
    pub top_k: usize,
    /// Token ids the distributions are restricted to, if any.
    pub allowed: Option<Vec<u32>>,
    pub reply: oneshot::Sender<Result<Vec<Distribution>, String>>,
    /// Released when the request is dropped, after the model has run it.
    _hold: Hold,
}

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
    pub planner: Planner,
    /// Published working sets, by key.
    index: Mutex<Index>,
}

/// Pages published under a key, and when each key was last used. The index
/// holds one reference to every page in it, so they outlive the working set
/// that published them. It is a cache: under memory pressure the entry used
/// longest ago is dropped.
#[derive(Default)]
struct Index {
    entries: HashMap<String, (Vec<u32>, u64)>,
    clock: u64,
}

struct Pool {
    free: Vec<u32>,
    refs: Vec<u32>,
    freed: Arc<Notify>,
}

impl Pool {
    fn free(&mut self, pages: impl IntoIterator<Item = u32>) {
        for p in pages {
            self.refs[p as usize] -= 1;
            if self.refs[p as usize] == 0 {
                self.free.push(p);
            }
        }
        self.freed.notify_waiters();
    }
}

impl Engine {
    /// UPDATED
    /// Takes the scheduler's token budget per step.
    pub fn new(model: Model, tokenizer: Tokenizer, eos: Vec<u32>, pages: u32, step_tokens: usize) -> Self {
        let page_size = model.page_size as u32;
        let (queue, rx) = mpsc::unbounded_channel();
        std::thread::spawn(move || crate::scheduler::run(model, rx, step_tokens));
        let planner = Planner::new();
        Self {
            tokenizer,
            eos,
            page_size,
            pool: Arc::new(Mutex::new(Pool {
                free: (0..pages).rev().collect(),
                refs: vec![0; pages as usize],
                freed: planner.freed.clone(),
            })),
            queue,
            planner,
            index: Mutex::default(),
        }
    }

    pub fn alloc(&self, n: u32) -> Option<Vec<u32>> {
        let mut pool = self.pool.lock().unwrap();
        let at = pool.free.len().checked_sub(n as usize)?;
        let pages = pool.free.split_off(at);
        pages.iter().for_each(|&p| pool.refs[p as usize] = 1);
        Some(pages)
    }

    /// Drops cached prefixes before it waits on other inferlets.
    pub async fn alloc_wait(&self, id: u64, n: u32) -> Result<Vec<u32>, String> {
        loop {
            // Listen before checking, so a free between the two is not missed.
            let freed = self.planner.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            if let Some(pages) = self.alloc(n) {
                self.planner.running(id);
                return Ok(pages);
            }
            // Before waiting on other inferlets, give up cached prefixes.
            if self.evict_oldest() {
                continue;
            }
            self.planner.wait(id)?;
            freed.await;
        }
    }

    /// Publish `pages` under `key`, replacing what was there.
    pub fn publish(&self, key: String, pages: &[u32]) {
        self.share(pages);
        let mut index = self.index.lock().unwrap();
        index.clock += 1;
        let entry = (pages.to_vec(), index.clock);
        if let Some((old, _)) = index.entries.insert(key, entry) {
            self.free(old);
        }
    }

    /// The pages published under `key`, with one more holder each.
    pub fn open(&self, key: &str) -> Option<Vec<u32>> {
        let mut index = self.index.lock().unwrap();
        index.clock += 1;
        let clock = index.clock;
        let (pages, used) = index.entries.get_mut(key)?;
        *used = clock;
        self.share(pages);
        Some(pages.clone())
    }

    pub fn unpublish(&self, key: &str) -> bool {
        let old = self.index.lock().unwrap().entries.remove(key);
        old.map(|(pages, _)| self.free(pages)).is_some()
    }

    /// Drop the entry used longest ago. False if the index is empty.
    fn evict_oldest(&self) -> bool {
        let mut index = self.index.lock().unwrap();
        let Some(key) = index
            .entries
            .iter()
            .min_by_key(|(_, (_, used))| *used)
            .map(|(k, _)| k.clone())
        else {
            return false;
        };
        let (pages, _) = index.entries.remove(&key).unwrap();
        self.free(pages);
        true
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

    /// Takes the `allowed` restriction along with the request.
    pub fn request(&self, seq: Seq, top_k: usize, allowed: Option<Vec<u32>>) -> (Request, Reply) {
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
                allowed,
                reply,
                _hold: hold,
            },
            rx,
        )
    }

    pub fn send(&self, requests: Vec<Request>) -> Result<(), String> {
        self.queue.send(requests).map_err(|_| "engine stopped".into())
    }
}

/// The `k` most likely tokens, only among `allowed` if given; probabilities
/// are normalized over the tokens that could be picked.
pub fn top_k(logits: Vec<f32>, k: usize, allowed: Option<&[u32]>) -> Distribution {
    let mut idx: Vec<u32> = match allowed {
        Some(ids) => ids.iter().copied().filter(|&i| (i as usize) < logits.len()).collect(),
        None => (0..logits.len() as u32).collect(),
    };
    if idx.is_empty() {
        return Distribution {
            ids: vec![],
            probs: vec![],
        };
    }
    let max = idx
        .iter()
        .map(|&i| logits[i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = idx.iter().map(|&i| (logits[i as usize] - max).exp()).sum();
    let k = k.clamp(1, idx.len());
    let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    idx.select_nth_unstable_by(k - 1, by_logit);
    idx.truncate(k);
    idx.sort_unstable_by(by_logit);
    let probs = idx.iter().map(|&i| (logits[i as usize] - max).exp() / sum).collect();
    Distribution { ids: idx, probs }
}
