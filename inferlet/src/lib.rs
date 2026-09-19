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
pub use pie::core::chat;
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
        ctx.submit_rows(&[], None, 1)?.wait()?;
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
        self.submit_rows(&outputs, None, top_k)?.wait()
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
        Ok(Pending(self.submit_rows(&[last], None, top_k)?))
    }

    /// Like `forward`, but the next token can only be one of `allowed`.
    pub fn forward_allowed(&mut self, allowed: &[u32], top_k: u32) -> Result<Distribution, String> {
        let last = self.pending.len().saturating_sub(1) as u32;
        Ok(self.submit_rows(&[last], Some(allowed), top_k)?.wait()?.remove(0))
    }

    /// Submit the pending tokens, asking for a distribution after each
    /// token listed in `outputs` (indices into the pending tokens).
    pub fn submit_rows(
        &mut self,
        outputs: &[u32],
        allowed: Option<&[u32]>,
        top_k: u32,
    ) -> Result<PendingForward, String> {
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

        let pending = model::forward(&self.kv, len, &self.pending, &positions, outputs, allowed, top_k)?;
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

/// NEW
/// An assistant reply. Models that reason first (Qwen3) write their thinking
/// between `<think>` and `</think>` before the answer; it is split off here.
pub struct Reply {
    pub thinking: Option<String>,
    pub text: String,
}

impl Context {
    /// NEW
    pub fn system(&mut self, message: &str) {
        self.fill_tokens(&chat::system(message));
    }

    /// NEW
    pub fn user(&mut self, message: &str) {
        self.fill_tokens(&chat::user(message));
    }

    /// NEW
    /// Generate the assistant's reply to the conversation so far, and close
    /// its turn so the next message can follow. The context keeps every turn,
    /// so the next reply only runs the new tokens. Thinking is removed from
    /// the history afterwards, as the model's own chat template does: the
    /// reply is rolled back and replaced by the answer alone.
    pub fn reply(
        &mut self,
        max_tokens: usize,
        top_k: u32,
        mut sample: impl FnMut(&Distribution) -> u32,
    ) -> Result<Reply, String> {
        self.fill_tokens(&chat::cue());
        let stop = chat::stop_tokens();
        let mut out = vec![];
        while out.len() < max_tokens {
            let next = sample(&self.forward(top_k)?);
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            self.fill_tokens(&[next]);
        }

        let text = model::detokenize(&out);
        let reply = match text.split_once("</think>") {
            Some((thinking, answer)) => Reply {
                thinking: Some(thinking.replace("<think>", "").trim().to_string()),
                text: answer.trim().to_string(),
            },
            None => Reply {
                thinking: None,
                text: text.trim().to_string(),
            },
        };
        if reply.thinking.is_some() {
            // Every generated token but the pending last one has been forwarded.
            let forwarded = out.len() - self.pending.len();
            self.pending.clear();
            self.rollback(forwarded);
            self.fill(&reply.text);
        }
        self.fill_tokens(&chat::seal());
        Ok(reply)
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
