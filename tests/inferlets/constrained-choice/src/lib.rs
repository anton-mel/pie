//! The answer must be one of a list of choices.
//!
//! Example: `pie constrained_choice.wasm -- "prompt" "choice|choice|choice"`.
//!
//! Each step, the inferlet works out which tokens could still lead to one of
//! the choices, given what it has generated so far, and allows only those.
//! Greedy decoding then cannot leave the list: the result is always exactly
//! one of the choices. A grammar works the same way, with a richer rule for
//! what is allowed next.

use inferlet::{Context, greedy, tokenizer};

const PROMPT: &str = "Review: \"The food was cold and the waiter ignored us for an hour.\"\n\
Is this review positive, negative or neutral? Answer:";

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let prompt = args.first().map_or(PROMPT, |s| s);
        let list = args.get(1).map_or("positive|negative|neutral", |s| s);
        // As they follow the prompt: after a space.
        let choices: Vec<Vec<u32>> = list.split('|').map(|c| tokenizer::tokenize(&format!(" {c}"))).collect();

        let mut ctx = Context::new();
        ctx.fill(prompt);
        let mut out: Vec<u32> = vec![];
        let mut first = None;
        loop {
            // Choices that start with what we have so far.
            let open: Vec<&Vec<u32>> = choices.iter().filter(|c| c.starts_with(&out)).collect();
            if open.iter().any(|c| c.len() == out.len()) {
                break;
            }
            // The next token of each of them is all that is allowed.
            let mut allowed: Vec<u32> = open.iter().map(|c| c[out.len()]).collect();
            allowed.sort();
            allowed.dedup();
            let d = ctx.forward_allowed(&allowed, allowed.len() as u32)?;
            first.get_or_insert_with(|| d.clone());
            let next = greedy(&d);
            out.push(next);
            ctx.fill_tokens(&[next]);
        }

        let d = first.unwrap();
        let odds: Vec<String> = d
            .ids
            .iter()
            .zip(&d.probs)
            .map(|(t, p)| format!("{:?} {:.2}", tokenizer::detokenize(&[*t]), p))
            .collect();
        Ok(format!(
            "{:?}\n    first token: {}",
            tokenizer::detokenize(&out),
            odds.join(", ")
        ))
    }
}

inferlet::export!(App);
