//! An interactive chat, streamed.
//!
//! Example: `pie-client chat_session.wasm`, then type messages.
//!
//! Each message from the other side is one user turn. The answer is sent
//! back as it is generated, a few characters at a time, and the whole
//! conversation stays in one context until the other side is done.

use inferlet::{Context, Sampler, session};

struct App;

impl inferlet::Guest for App {
    fn run(_: Vec<String>) -> Result<String, String> {
        let mut ctx = Context::new();
        ctx.system("You are a helpful assistant. Answer briefly.");
        let mut sampler = Sampler::new(0.6, 0.95);
        let mut turns = 0;

        while let Some(message) = session::receive() {
            ctx.user(&message);
            let mut sent = 0;
            ctx.reply_streaming(
                1024,
                64,
                |d| sampler.sample(d),
                |answer| {
                    // Send only what is new since the last call.
                    if answer.len() > sent && answer.is_char_boundary(sent) {
                        session::send(&answer[sent..]);
                        sent = answer.len();
                    }
                },
            )?;
            session::send("\n");
            turns += 1;
        }
        Ok(format!("[{turns} turns]"))
    }
}

inferlet::export!(App);
