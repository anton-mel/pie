//! Counters of what the runtime has done, for the `/metrics` endpoint.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Default)]
pub struct Metrics {
    /// Model steps run.
    pub steps: AtomicU64,
    /// Tokens run through the model.
    pub tokens: AtomicU64,
    /// Forwards finished.
    pub forwards: AtomicU64,
    /// Inferlets evicted to free KV pages.
    pub evictions: AtomicU64,
    /// Pages reused from recorded prefixes instead of computed.
    pub prefix_pages_reused: AtomicU64,
}

impl Metrics {
    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Relaxed);
    }

    /// The counters, plus the pool's current state, in Prometheus text format.
    pub fn render(&self, free_pages: usize, total_pages: usize, recorded_pages: usize) -> String {
        let counters = [
            ("pie_steps_total", "Model steps run.", &self.steps),
            ("pie_tokens_total", "Tokens run through the model.", &self.tokens),
            ("pie_forwards_total", "Forwards finished.", &self.forwards),
            (
                "pie_evictions_total",
                "Inferlets evicted to free KV pages.",
                &self.evictions,
            ),
            (
                "pie_prefix_pages_reused_total",
                "Pages reused from recorded prefixes.",
                &self.prefix_pages_reused,
            ),
        ];
        let mut out = String::new();
        for (name, help, value) in counters {
            out += &format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
                value.load(Relaxed)
            );
        }
        let gauges = [
            ("pie_kv_pages_free", "KV pages free.", free_pages),
            ("pie_kv_pages_total", "KV pages in the pool.", total_pages),
            (
                "pie_prefix_pages_recorded",
                "Pages recorded for prefix sharing.",
                recorded_pages,
            ),
        ];
        for (name, help, value) in gauges {
            out += &format!("# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n");
        }
        out
    }
}
