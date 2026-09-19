//! Who gets KV pages when there are not enough.
//!
//! An inferlet asking for pages that are not free waits for someone to free
//! some. That is fine while at least one inferlet is still running: it will
//! finish or drop a working set. When every live inferlet is waiting, nobody
//! will ever free a page, so the planner evicts one: it kills the youngest,
//! which frees its pages, and it is restarted from scratch once some other
//! inferlet has finished. The oldest is never evicted, so it always progress.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

pub struct Planner {
    inner: Mutex<Inner>,
    /// Wakes waiters when pages are freed or an evicted inferlet is gone.
    pub freed: Arc<Notify>,
    /// Wakes evicted inferlets when another one finishes.
    exited: Notify,
}

struct Inner {
    next: u64,
    /// Live inferlets by start order: the last one is the youngest.
    live: BTreeMap<u64, Member>,
    /// Evicted inferlets whose pages are not freed yet.
    dying: usize,
}

struct Member {
    waiting: bool,
    kill: Arc<Notify>,
}

impl Planner {
    pub fn new() -> Self {
        let inner = Inner {
            next: 0,
            live: BTreeMap::new(),
            dying: 0,
        };
        Self {
            inner: Mutex::new(inner),
            freed: Arc::new(Notify::new()),
            exited: Notify::new(),
        }
    }

    /// Register a new inferlet.
    pub fn join(&self) -> (u64, Arc<Notify>) {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next;
        inner.next += 1;
        let kill = Arc::new(Notify::new());
        let member = Member {
            waiting: false,
            kill: kill.clone(),
        };
        inner.live.insert(id, member);
        (id, kill)
    }

    /// The inferlet has exited, or was evicted and its pages are freed.
    pub fn leave(&self, id: u64, evicted: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.live.remove(&id);
        if evicted {
            inner.dying -= 1;
        } else {
            self.exited.notify_waiters();
        }
        self.freed.notify_waiters();
    }

    /// Before restarting an evicted inferlet: wait until some other one
    /// finishes, or it would take the freed pages back and be evicted again.
    pub async fn until_exit(&self) {
        let exited = self.exited.notified();
        tokio::pin!(exited);
        exited.as_mut().enable();
        if self.inner.lock().unwrap().live.is_empty() {
            return;
        }
        exited.await;
    }

    /// id is waiting for pages. If now every live inferlet is, evict the
    /// youngest. Fails if id is alone: its request can never be met.
    pub fn wait(&self, id: u64) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(m) = inner.live.get_mut(&id) {
            m.waiting = true;
        }
        if inner.dying > 0 || !inner.live.values().all(|m| m.waiting) {
            return Ok(());
        }
        if inner.live.len() == 1 {
            return Err("out of KV pages".into());
        }
        let (_, victim) = inner.live.pop_last().unwrap();
        inner.dying += 1;
        victim.kill.notify_one();
        Ok(())
    }

    pub fn running(&self, id: u64) {
        if let Some(m) = self.inner.lock().unwrap().live.get_mut(&id) {
            m.waiting = false;
        }
    }
}
