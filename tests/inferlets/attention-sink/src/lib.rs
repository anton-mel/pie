//! Generate far more tokens than the KV cache holds (attention sink).
//!
//! Example: `pie attention_sink.wasm -- "prompt" [max_tokens] [window_pages]`.
//!
//! The first page stays forever: models lean on the first tokens (the
//! "sink"), and dropping them hurts much more than dropping others. After it,
//! only the last `window_pages` pages are kept. Whenever the cache grows past
//! that, the oldest page after the sink is discarded, so memory stays fixed
//! however long the output gets.

use inferlet::{Context, greedy, tokenizer};

const SINK: u32 = 1;

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or("Once upon a time", |s| s);
        let max_tokens: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(256);
        let window: u32 = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(4);
        let eos = tokenizer::eos_tokens();

        let mut ctx = Context::new();
        ctx.fill(prompt);
        let mut out = vec![];
        let (mut discarded, mut peak) = (0, 0);

        while out.len() < max_tokens {
            let next = greedy(&ctx.forward(1)?);
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            ctx.fill_tokens(&[next]);

            // Keep the sink and the last `window` pages; drop the oldest in between.
            while ctx.page_len() > SINK + window {
                ctx.discard(SINK, 1)?;
                discarded += 1;
            }
            peak = peak.max(ctx.page_len());
        }

        Ok(format!(
            "{:?}\n    {} tokens, at most {} pages held, {} pages discarded",
            tokenizer::detokenize(&out),
            out.len(),
            peak,
            discarded
        ))
    }
}

inferlet::export!(App);
