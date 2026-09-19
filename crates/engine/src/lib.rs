//! The runtime↔engine contract: what the runtime asks of whatever runs the
//! model. The runtime depends on this crate only, never on a model, so a
//! new backend is a new implementation of `Engine`.

/// What an engine does for the runtime.
pub trait Engine: Send {
    /// Tokens per KV page.
    fn page_size(&self) -> usize;

    /// Run one step over `seqs`. Returns one row for every entry of every
    /// sequence's `outputs`, in order: its logits, or the token sampled
    /// from them if the sequence asked for sampling.
    fn forward(&mut self, seqs: &[Seq]) -> anyhow::Result<Vec<Row>>;
}

/// One output of a step.
pub enum Row {
    /// Every token's score.
    Logits(Vec<f32>),
    /// A token picked next to the logits, and its probability.
    Sampled { token: u32, prob: f32 },
}

/// How to pick a token on the device: from softmax(logits / temperature),
/// keeping only tokens at least `min_p` times as likely as the most likely.
/// Temperature 0 picks the most likely token.
#[derive(Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub min_p: f32,
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
    /// Sample its outputs on the device instead of returning logits.
    pub sample: Option<Sampling>,
}
