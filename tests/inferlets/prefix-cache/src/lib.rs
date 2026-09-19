//! Many inferlets, one long shared prompt, computed once.
//!
//! Example: `pie -i 4 --sequential prefix_cache.wasm -- "question" [max_tokens]`.
//!
//! Every request starts with the same system prompt. `Context::cached`
//! computes it once and publishes its KV. Every later inferlet opens the
//! published pages instead of running the prompt again, and only runs its
//! own question.

use inferlet::{Context, KvWorkingSet, greedy};

const SYSTEM: &str = "You are the help desk assistant of the Springfield Public Library. \
The library is open Monday to Friday from 9am to 8pm, Saturday from 10am to 6pm, and closed on Sunday. \
Members can borrow up to 10 books at a time for 3 weeks, and renew each book twice online or at the desk, \
unless another member has reserved it. Late books cost 25 cents per day, up to 10 dollars per book. \
DVDs can be borrowed for 1 week and cost 1 dollar per day when late. \
Membership is free for residents of Springfield; others pay 30 dollars a year. \
To join, bring a photo ID and a proof of address, such as a utility bill. \
Children under 12 need a parent or guardian to sign for them. \
The library has 20 computers with free internet for members, bookable for 2 hours a day, \
free wifi for everyone, and printing at 10 cents per page in black and white or 50 cents in color. \
Study rooms for up to 6 people can be booked up to one week ahead. \
Story time for children is every Wednesday at 10am, and the book club meets on the first Thursday of every month at 7pm. \
Lost cards can be replaced at the desk for 2 dollars. \
Answer questions about the library briefly and politely, in one or two sentences.\n\n";

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let question = args.first().map_or("How long can I keep a DVD?", |s| s);
        let max_tokens: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(32);
        let reused = KvWorkingSet::from_index(SYSTEM).is_some();

        let mut ctx = Context::cached(SYSTEM)?;
        let shared = ctx.tokens.len();
        ctx.fill(&format!("Question: {question}\nAnswer:"));
        let answer = ctx.generate(max_tokens, 1, greedy)?;

        let prefix = if reused { "reused" } else { "computed and published" };
        Ok(format!("{answer:?}\n    system prompt ({shared} tokens) {prefix}"))
    }
}

inferlet::export!(App);
