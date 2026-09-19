//! Measure how steadily tokens come out.
//!
//! Example: `pie decode_latency.wasm -- [tokens]`.
//!
//! Decodes greedily and times the gap between one token and the next. The
//! longest gap shows how long this inferlet was held up by others: a long
//! prompt from another inferlet in the same step, for example.

use inferlet::{Context, greedy};
use std::time::Instant;

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let tokens: usize = args.first().and_then(|n| n.parse().ok()).unwrap_or(200);
        let mut ctx = Context::new();
        ctx.fill("Once upon a time");
        let mut gaps = vec![];
        let mut last = Instant::now();
        for _ in 0..tokens {
            let next = greedy(&ctx.forward(1)?);
            ctx.fill_tokens(&[next]);
            gaps.push(last.elapsed().as_secs_f64() * 1000.0);
            last = Instant::now();
        }
        gaps.remove(0); // the prompt
        gaps.sort_by(|a, b| a.total_cmp(b));
        let median = gaps[gaps.len() / 2];
        let max = gaps[gaps.len() - 1];
        Ok(format!(
            "{tokens} tokens: median gap {median:.1} ms, longest gap {max:.1} ms"
        ))
    }
}

inferlet::export!(App);
