//! Several samples of one prompt at once.
//!
//! Example: `pie parallel_sampling.wasm -- "prompt" [n] [max_tokens] [temperature] [top_p] [seed]`.
//!
//! The prompt runs once and is forked into `n` branches that share its KV
//! pages. Each branch samples with its own `Sampler`, and every step submits
//! all branches before waiting, so they decode in one batch. Sampling is all
//! inferlet code: the engine only returns the top-k of each distribution.

use inferlet::{Context, Sampler, model};

struct Branch {
    ctx: Context,
    sampler: Sampler,
    out: Vec<u32>,
    done: bool,
}

/// Argument `i` parsed as a `T`, if it is there.
fn arg<T: std::str::FromStr>(args: &[String], i: usize) -> Option<T> {
    args.get(i).and_then(|s| s.parse().ok())
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or("Once upon a time", |s| s);
        let n: usize = arg(&args, 1).unwrap_or(4);
        let max_tokens: usize = arg(&args, 2).unwrap_or(24);
        let temperature: f32 = arg(&args, 3).unwrap_or(0.8);
        let top_p: f32 = arg(&args, 4).unwrap_or(0.95);
        let seed: Option<u64> = arg(&args, 5);
        let eos = model::eos_tokens();

        let mut root = Context::new();
        root.fill(prompt);
        let first = root.forward(64)?;

        let mut branches: Vec<Branch> = (0..n as u64)
            .map(|i| {
                let sampler = Sampler::new(temperature, top_p);
                Branch {
                    ctx: root.fork(),
                    sampler: match seed {
                        Some(s) => sampler.seed(s + i),
                        None => sampler,
                    },
                    out: vec![],
                    done: false,
                }
            })
            .collect();

        let mut dists: Vec<_> = (0..n).map(|_| first.clone()).collect();
        for _ in 0..max_tokens {
            for (b, d) in branches.iter_mut().zip(&dists) {
                if b.done {
                    continue;
                }
                let t = b.sampler.sample(d);
                if eos.contains(&t) {
                    b.done = true;
                } else {
                    b.out.push(t);
                    b.ctx.fill_tokens(&[t]);
                }
            }
            if branches.iter().all(|b| b.done) {
                break;
            }
            // Submit every live branch, then wait: one batched step.
            let mut pending = vec![];
            for b in branches.iter_mut() {
                pending.push(if b.done { None } else { Some(b.ctx.submit(64)?) });
            }
            for (d, p) in dists.iter_mut().zip(pending) {
                if let Some(p) = p {
                    *d = p.wait()?;
                }
            }
        }

        let lines: Vec<String> = branches
            .iter()
            .map(|b| format!("{:?}", model::detokenize(&b.out)))
            .collect();
        Ok(lines.join("\n    "))
    }
}

inferlet::export!(App);
