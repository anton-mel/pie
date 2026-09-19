//! A conversation: each argument is one user message.
//!
//! Example: `pie chat.wasm -- "What is the capital of France?" "And its most famous tower?"`.
//!
//! The chat format comes from the host (`chat` in `wit/pie.wit`), so this
//! inferlet does not know how the model spells its turns. One context holds
//! the whole conversation: each reply only runs the new message.

use inferlet::{Context, Sampler};

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let turns = if args.is_empty() {
            vec![
                "What is the capital of France?".into(),
                "What is its most famous tower?".into(),
            ]
        } else {
            args
        };

        let mut ctx = Context::new();
        ctx.system("You are a helpful assistant. Answer in one short sentence.");
        // Qwen3's recommended settings for chat.
        let mut sampler = Sampler::new(0.6, 0.95).seed(1);
        let mut log = vec![];
        for message in &turns {
            ctx.user(message);
            let reply = ctx.reply(1024, 64, |d| sampler.sample(d))?;
            let thought = reply.thinking.map_or(0, |t| t.split_whitespace().count());
            log.push(format!(
                "user: {message}\n    assistant: {}  (thought for {thought} words)",
                reply.text
            ));
        }
        Ok(log.join("\n    "))
    }
}

inferlet::export!(App);
