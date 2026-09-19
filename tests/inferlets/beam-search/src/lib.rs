//! Beam search, written as an inferlet.
//!
//! Example: `pie beam_search.wasm -- "prompt" [width] [max_tokens]`.
//!
//! The prompt is run once. Every step each beam proposes its `width` most
//! likely next tokens, the best `width` of all proposals survive, and each
//! survivor is a `fork` of its parent: the prompt's KV pages are shared by
//! every beam and never copied. All beams are submitted before any is
//! waited for, so each step is one batched model step, not `width` of them.

use inferlet::{Context, tokenizer};

struct Beam {
    ctx: Context,
    out: Vec<u32>,
    logp: f32,
    done: bool,
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or("The capital of France is", |s| s);
        let width = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(3);
        let max_tokens = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(16);
        let eos = tokenizer::eos_tokens();

        let mut ctx = Context::new();
        ctx.fill(prompt);
        let mut beams = vec![Beam {
            ctx,
            out: vec![],
            logp: 0.0,
            done: false,
        }];

        for _ in 0..max_tokens {
            // Every beam's proposals is a tuple:
            // (parent, next token or None if done, score).
            // Submit every live beam first, then wait: they run in one step.
            let mut submitted = vec![];
            for b in beams.iter_mut() {
                submitted.push(if b.done { None } else { Some(b.ctx.submit(width)?) });
            }
            let mut proposals = vec![];
            for (i, (b, pending)) in beams.iter().zip(submitted).enumerate() {
                let Some(pending) = pending else {
                    proposals.push((i, None, b.logp));
                    continue;
                };
                let d = pending.wait()?;
                for (t, p) in d.ids.iter().zip(&d.probs) {
                    proposals.push((i, Some(*t), b.logp + p.ln()));
                }
            }
            proposals.sort_by(|a, b| b.2.total_cmp(&a.2));
            proposals.truncate(width as usize);
            if proposals.iter().all(|p| p.1.is_none()) {
                break;
            }

            // Each survivor forks its parent; the old beams are dropped here,
            // which releases their hold on the shared pages.
            beams = proposals
                .into_iter()
                .map(|(i, token, logp)| {
                    let parent = &beams[i];
                    let mut beam = Beam {
                        ctx: parent.ctx.fork(),
                        out: parent.out.clone(),
                        logp,
                        done: parent.done,
                    };
                    match token {
                        Some(t) if !eos.contains(&t) => {
                            beam.out.push(t);
                            beam.ctx.fill_tokens(&[t]);
                        }
                        _ => beam.done = true,
                    }
                    beam
                })
                .collect();
        }

        let lines: Vec<String> = beams
            .iter()
            .map(|b| format!("{:.2} {:?}", b.logp, tokenizer::detokenize(&b.out)))
            .collect();
        Ok(lines.join("\n    "))
    }
}

inferlet::export!(App);
