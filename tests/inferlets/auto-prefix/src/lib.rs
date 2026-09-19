//! Shared prompt prefixes found by their tokens, with no key.
//!
//! Example: `pie -i 3 --sequential auto_prefix.wasm -- "question" [max_tokens]`.
//!
//! Every request is the same long system prompt followed by its own
//! question. The inferlet does not publish anything or agree on a key: it
//! builds its prompt and asks for it with `Context::with_tokens`, and gets
//! back the pages of the longest prefix any inferlet computed before. Only
//! the rest of the prompt runs.

use inferlet::{Context, chat, greedy};

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

        let mut prompt = chat::prefix();
        prompt.extend(chat::system_user(SYSTEM.trim(), &format!("{question} /no_think")));
        let mut ctx = Context::with_tokens(&prompt);
        let reused = ctx.tokens.len();

        let answer = ctx.reply(max_tokens, 1, greedy)?.text;
        Ok(format!(
            "{answer:?}\n    reused {reused} of {} prompt tokens",
            prompt.len()
        ))
    }
}

inferlet::export!(App);
