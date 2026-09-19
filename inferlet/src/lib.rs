//! The library every inferlet links against. It does two things:
//!
//! 1. Generates Rust bindings from `wit/pie.wit`, so an inferlet can call
//!    the runtime as plain functions.
//! 2. Adds `Context`, a helper that tracks a sequence's tokens and its KV
//!    working set. You add text, and it reserves pages, computes positions,
//!    and calls `forward` for you. Its pages are freed when it is dropped.
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

pub use exports::pie::core::run::Guest;
pub use pie::core::model::{self, Distribution, KvWorkingSet};

pub struct Context {
    /// Tokens whose K/V are in the cache.
    pub tokens: Vec<u32>,
    /// Tokens added but not yet run through the model.
    pending: Vec<u32>,
    kv: KvWorkingSet,
    page_size: u32,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tokens: vec![],
            pending: vec![],
            kv: KvWorkingSet::new(),
            page_size: model::kv_page_size(),
        }
    }

    /// A copy of this context that shares its KV cache. Both can go on
    /// independently; a shared page is copied only when one of them writes.
    pub fn fork(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            pending: self.pending.clone(),
            kv: self.kv.fork(),
            page_size: self.page_size,
        }
    }

    pub fn fill(&mut self, text: &str) {
        self.pending.extend(model::tokenize(text));
    }

    pub fn fill_tokens(&mut self, tokens: &[u32]) {
        self.pending.extend_from_slice(tokens);
    }

    // one model step
    pub fn forward(&mut self, top_k: u32) -> Result<Distribution, String> {
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

        let positions: Vec<u32> = (start..len).collect();

        let dist = model::forward(&self.kv, len, &self.pending, &positions, top_k)?;
        self.tokens.append(&mut self.pending);

        Ok(dist)
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

/// Always the most likely token.
pub fn greedy(d: &Distribution) -> u32 {
    d.ids[0]
}
