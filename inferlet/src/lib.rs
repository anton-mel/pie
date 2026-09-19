//! The library every inferlet links against.
//!
//! To write an inferlet, implement `Guest::run` and export it:
//!
//! ```ignore
//! struct App;
//! impl inferlet::Guest for App {
//!     fn run(args: Vec<String>) -> Result<String, String> { ... }
//! }
//! inferlet::export!(App);
//! ```

wit_bindgen::generate!({
    path: "../wit",
    world: "inferlet",
    pub_export_macro: true,
    default_bindings_module: "inferlet",
});

mod sample;

pub use exports::pie::core::run::Guest;
pub use pie::core::model::{self, Distribution, KvWorkingSet, PendingForward};
pub use sample::Sampler;

pub struct Context {
    pub tokens: Vec<u32>,
    pending: Vec<u32>,
    kv: KvWorkingSet,
    page_size: u32,
    /// Position of the next token. Equal to `tokens.len()` until pages are
    /// discarded: then the cache gets shorter but positions keep counting.
    pos: u32,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tokens: vec![],
            pending: vec![],
            kv: KvWorkingSet::new(),
            page_size: model::kv_page_size(),
            pos: 0,
        }
    }

    pub fn fork(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            pending: self.pending.clone(),
            kv: self.kv.fork(),
            page_size: self.page_size,
            pos: self.pos,
        }
    }

    /// NEW
    /// A context holding `text`, computed once for all inferlets: the first
    /// to ask runs it and publishes its KV under `text`, later ones open that
    /// and skip the work. Add more tokens and forward as usual.
    pub fn cached(text: &str) -> Result<Self, String> {
        let mut ctx = Self::new();
        let tokens = model::tokenize(text);
        if let Some(kv) = KvWorkingSet::from_index(text) {
            ctx.kv = kv;
            ctx.pos = tokens.len() as u32;
            ctx.tokens = tokens;
            return Ok(ctx);
        }
        // Prefill without asking for any output: only the KV is needed.
        ctx.pending = tokens;
        ctx.submit_rows(&[], 1)?.wait()?;
        ctx.kv.update_index(text);
        Ok(ctx)
    }

    pub fn fill(&mut self, text: &str) {
        self.pending.extend(model::tokenize(text));
    }

    pub fn fill_tokens(&mut self, tokens: &[u32]) {
        self.pending.extend_from_slice(tokens);
    }

    // one model step
    pub fn forward(&mut self, top_k: u32) -> Result<Distribution, String> {
        self.submit(top_k)?.wait()
    }

    /// Run the pending tokens and return a distribution after each of them:
    /// row `i` predicts the token that follows pending token `i`.
    pub fn forward_all(&mut self, top_k: u32) -> Result<Vec<Distribution>, String> {
        let outputs: Vec<u32> = (0..self.pending.len() as u32).collect();
        self.submit_rows(&outputs, top_k)?.wait()
    }

    /// Forget the last `n` tokens, as if they had never been forwarded.
    /// Their KV stays in the pages but is past the end, and the next forward
    /// writes over it.
    pub fn rollback(&mut self, n: usize) {
        let n = n.min(self.tokens.len());
        self.tokens.truncate(self.tokens.len() - n);
        self.pos -= n as u32;
    }

    /// Pages holding the cached tokens.
    pub fn page_len(&self) -> u32 {
        (self.tokens.len() as u32).div_ceil(self.page_size)
    }

    /// Drop `n` pages of cached tokens starting at page `start`. The model
    /// no longer sees those tokens, and their pages go back to the pool.
    pub fn discard(&mut self, start: u32, n: u32) -> Result<(), String> {
        let ps = self.page_size as usize;
        let from = (start as usize * ps).min(self.tokens.len());
        let to = ((start + n) as usize * ps).min(self.tokens.len());
        self.kv.discard(start, n)?;
        self.tokens.drain(from..to);
        Ok(())
    }

    pub fn submit(&mut self, top_k: u32) -> Result<Pending, String> {
        let last = self.pending.len().saturating_sub(1) as u32;
        Ok(Pending(self.submit_rows(&[last], top_k)?))
    }

    /// Submit the pending tokens, asking for a distribution after each
    /// token listed in `outputs` (indices into the pending tokens).
    pub fn submit_rows(&mut self, outputs: &[u32], top_k: u32) -> Result<PendingForward, String> {
        if self.pending.is_empty() {
            return Err("nothing to forward".into());
        }

        let start = self.tokens.len() as u32;
        let len = start + self.pending.len() as u32;
        let need = len.div_ceil(self.page_size);
        let have = self.kv.page_len();

        if need > have {
            self.kv.reserve(need - have)?;
        }

        let positions: Vec<u32> = (self.pos..self.pos + self.pending.len() as u32).collect();
        self.pos += self.pending.len() as u32;

        let pending = model::forward(&self.kv, len, &self.pending, &positions, outputs, top_k)?;
        self.tokens.append(&mut self.pending);

        Ok(pending)
    }

    // wrapper around forward
    // runs until EOS or max_tockens
    pub fn generate(
        &mut self,
        max_tokens: usize,
        top_k: u32,
        mut sample: impl FnMut(&Distribution) -> u32,
    ) -> Result<String, String> {
        let eos = model::eos_tokens();
        let mut out = Vec::new();

        while out.len() < max_tokens {
            let next = sample(&self.forward(top_k)?);
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            self.pending.push(next);
        }

        Ok(model::detokenize(&out))
    }
}

/// A submitted forward that returns one distribution: after the last token.
pub struct Pending(PendingForward);

impl Pending {
    pub fn wait(self) -> Result<Distribution, String> {
        Ok(self.0.wait()?.remove(0))
    }
}

/// Always the most likely token.
pub fn greedy(d: &Distribution) -> u32 {
    d.ids[0]
}
