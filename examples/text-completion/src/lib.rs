//! Greedy completion of a prompt.
// Example: 
// 
// `pie text_completion.wasm -- "prompt" [max_tokens]`.

use inferlet::{Context, greedy};

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        // get inputs
        let prompt = args.first().map_or("The capital of France is", |s| s);
        let max_tokens = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(32);
        // define inferlet library
        let mut ctx = Context::new();
        ctx.fill(prompt);
        ctx.generate(max_tokens, 1, greedy)
    }
}

inferlet::export!(App);
