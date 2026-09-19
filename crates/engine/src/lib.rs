//! The runtime↔engine contract: what the runtime asks of whatever runs the
//! model. The runtime depends on this crate only, never on a model, so a
//! new backend is a new implementation of `Engine`.

/// What an engine does for the runtime.
pub trait Engine: Send {
    /// Tokens per KV page.
    fn page_size(&self) -> usize;

    /// Run one step over `seqs`. Returns one row of logits for every entry
    /// of every sequence's `outputs`, in order.
    fn forward(&mut self, seqs: &[Seq]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// One sequence's share of a step: `tokens` are the last `tokens.len()`
/// entries of a `kv_len`-long sequence whose KV lives in `pages`. `copies`
/// are pages to copy before anything is written (copy-on-write).
pub struct Seq {
    pub copies: Vec<(u32, u32)>,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    /// Which of `tokens` to return logits for, by index.
    pub outputs: Vec<u32>,
    pub pages: Vec<u32>,
    pub kv_len: usize,
}
