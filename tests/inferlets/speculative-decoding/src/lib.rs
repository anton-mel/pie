//! Speculative decoding without a draft model (prompt lookup).
//!
//! Example: `pie speculative_decoding.wasm -- "prompt" [max_tokens] [draft_len]`.
//!
//! Each step guesses the next few tokens by finding the last two tokens
//! earlier in the text and copying what followed them there. The guesses go
//! through the model in one forward that returns a distribution after every
//! token, so all of them are checked at once. The longest prefix the model
//! agrees with is kept, the rest is rolled back. The output is exactly what
//! greedy decoding would produce, in fewer forwards.

use inferlet::{Context, greedy, tokenizer};

/// Guess up to `k` tokens that follow `seq`: find the last earlier place
/// where its final two tokens appeared and take what came after them.
fn lookup(seq: &[u32], k: usize) -> Vec<u32> {
    let n = seq.len();
    if n < 3 {
        return vec![];
    }
    let key = &seq[n - 2..];
    (0..n - 2)
        .rev()
        .find(|&i| &seq[i..i + 2] == key)
        .map_or(vec![], |i| seq[i + 2..(i + 2 + k).min(n)].to_vec())
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or("The capital of France is", |s| s);
        let max_tokens = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(64);
        let draft_len = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(4);
        let eos = tokenizer::eos_tokens();

        let mut ctx = Context::new();
        ctx.fill(prompt);
        let mut next = greedy(&ctx.forward(1)?);
        let mut out = vec![];
        let mut forwards = 1;

        'generate: while !eos.contains(&next) && out.len() < max_tokens {
            out.push(next);
            let mut seq = ctx.tokens.clone();
            seq.push(next);
            let draft = lookup(&seq, draft_len);

            // Row `i` predicts what follows token `i` of `[next] + draft`.
            ctx.fill_tokens(&[next]);
            ctx.fill_tokens(&draft);
            let rows = ctx.forward_all(1)?;
            forwards += 1;

            let accepted = (0..draft.len()).take_while(|&i| greedy(&rows[i]) == draft[i]).count();
            ctx.rollback(draft.len() - accepted);
            for &t in &draft[..accepted] {
                if eos.contains(&t) || out.len() >= max_tokens {
                    break 'generate;
                }
                out.push(t);
            }
            next = greedy(&rows[accepted]);
        }

        Ok(format!(
            "{:?}\n    {} tokens in {} forwards",
            tokenizer::detokenize(&out),
            out.len(),
            forwards
        ))
    }
}

inferlet::export!(App);
