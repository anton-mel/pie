//! `reasoning`: follow the model's thinking as it is generated.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::reasoning::{self, Event};
use wasmtime::component::Resource;

/// NEW
/// What has been generated, and how much thinking was already reported.
#[derive(Default)]
pub struct ThinkingDecoder {
    tokens: Vec<u32>,
    started: bool,
    reported: usize,
    done: bool,
}

impl reasoning::Host for State {}

impl reasoning::HostDecoder for State {
    async fn new(&mut self) -> Resource<ThinkingDecoder> {
        self.table
            .push(ThinkingDecoder::default())
            .expect("resource table full")
    }

    async fn feed(&mut self, d: Resource<ThinkingDecoder>, tokens: Vec<u32>) -> Event {
        let Some((open, close)) = self.engine.template.thinking_markers() else {
            return Event::None;
        };
        let tokenizer = &self.engine.tokenizer;
        let Ok(d) = self.table.get_mut(&d) else {
            return Event::None;
        };
        if d.done {
            return Event::None;
        }
        d.tokens.extend(tokens);
        let text = tokenizer.decode(&d.tokens, false).unwrap_or_default();
        let Some(start) = text.find(open) else {
            return Event::None;
        };
        let thinking = &text[start + open.len()..];
        if let Some(end) = thinking.find(close) {
            d.done = true;
            return Event::Complete(thinking[..end].trim().to_string());
        }
        if !d.started {
            // Its text, if any came along, follows as deltas.
            d.started = true;
            return Event::Start;
        }
        // Hold back a character whose bytes have not all arrived yet.
        let so_far = thinking.trim_end_matches('\u{FFFD}');
        if so_far.len() <= d.reported {
            return Event::None;
        }
        let delta = so_far[d.reported..].to_string();
        d.reported = so_far.len();
        Event::Delta(delta)
    }

    async fn reset(&mut self, d: Resource<ThinkingDecoder>) {
        if let Ok(d) = self.table.get_mut(&d) {
            *d = ThinkingDecoder::default();
        }
    }

    async fn drop(&mut self, d: Resource<ThinkingDecoder>) -> wasmtime::Result<()> {
        self.table.delete(d)?;
        Ok(())
    }
}
