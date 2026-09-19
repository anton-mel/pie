//! Several samples of one prompt, picked on the GPU.
//!
//! Example: `pie device_sampling.wasm -- "prompt" [n] [max_tokens] [temperature] [min_p]`.
//!
//! Like `parallel-sampling`: the prompt runs once and is forked into `n`
//! branches that share its pages, and every step submits all branches
//! before waiting, so they decode in one batch. The difference is where the
//! token is picked: each branch asks for it to be sampled on the device
//! (`submit_sampled`), so one token comes back per branch, not logits.

use inferlet::{Context, DeviceSampler, tokenizer};

fn arg<T: std::str::FromStr>(args: &[String], i: usize) -> Option<T> {
    args.get(i).and_then(|s| s.parse().ok())
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or("Once upon a time", |s| s);
        let n: usize = arg(&args, 1).unwrap_or(4);
        let max_tokens: usize = arg(&args, 2).unwrap_or(24);
        let sampler = DeviceSampler {
            temperature: arg(&args, 3).unwrap_or(0.8),
            min_p: arg(&args, 4).unwrap_or(0.05),
        };
        let eos = tokenizer::eos_tokens();

        // Run the prompt but its last token once; every branch runs that last
        // token itself, so each samples its own first token.
        let tokens = tokenizer::tokenize(prompt);
        let (head, last) = tokens.split_at(tokens.len() - 1);
        let mut root = Context::new();
        root.fill_tokens(head);
        root.submit_rows(&[], None, 1)?.wait()?;
        let mut branches: Vec<(Context, Vec<u32>, bool)> = (0..n)
            .map(|_| {
                let mut ctx = root.fork();
                ctx.fill_tokens(last);
                (ctx, vec![], false)
            })
            .collect();

        for _ in 0..max_tokens {
            let mut pending = vec![];
            for (ctx, _, done) in branches.iter_mut() {
                pending.push(if *done {
                    None
                } else {
                    Some(ctx.submit_sampled(sampler)?)
                });
            }
            for ((ctx, out, done), p) in branches.iter_mut().zip(pending) {
                let Some(p) = p else { continue };
                let token = p.wait()?.ids[0];
                if eos.contains(&token) {
                    *done = true;
                } else {
                    out.push(token);
                    ctx.fill_tokens(&[token]);
                }
            }
            if branches.iter().all(|b| b.2) {
                break;
            }
        }

        let lines: Vec<String> = branches
            .iter()
            .map(|b| format!("{:?}", tokenizer::detokenize(&b.1)))
            .collect();
        Ok(lines.join("\n    "))
    }
}

inferlet::export!(App);
