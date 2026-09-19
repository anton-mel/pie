//! `tools`: offer tools in the model's format, and spot its calls.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::tools::{self, Event, ToolCall};
use wasmtime::component::Resource;

/// What has been generated since the last call, and whether a call is open.
#[derive(Default)]
pub struct ToolDecoder {
    tokens: Vec<u32>,
    started: bool,
}

impl tools::Host for State {
    async fn equip(&mut self, tools: Vec<String>) -> Result<String, String> {
        self.engine
            .template
            .tools(&tools)
            .ok_or_else(|| "this model has no tool format here".into())
    }

    async fn answer(&mut self, values: Vec<String>) -> Result<Vec<u32>, String> {
        let text = self
            .engine
            .template
            .tool_results(&values)
            .ok_or("this model has no tool format here")?;
        Ok(self
            .engine
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default())
    }
}

impl tools::HostDecoder for State {
    async fn new(&mut self) -> Resource<ToolDecoder> {
        self.table.push(ToolDecoder::default()).expect("resource table full")
    }

    async fn feed(&mut self, d: Resource<ToolDecoder>, tokens: Vec<u32>) -> Result<Event, String> {
        let (open, close) = self
            .engine
            .template
            .tool_call_markers()
            .ok_or("this model has no tool format here")?;
        let tokenizer = &self.engine.tokenizer;
        let d = self.table.get_mut(&d).map_err(|e| e.to_string())?;
        d.tokens.extend(tokens);
        let text = tokenizer.decode(&d.tokens, false).unwrap_or_default();
        let Some(start) = text.find(open) else {
            return Ok(Event::None);
        };
        let body = &text[start + open.len()..];
        let Some(end) = body.find(close) else {
            let first = !d.started;
            d.started = true;
            return Ok(if first { Event::Start } else { Event::None });
        };
        let call: serde_json::Value =
            serde_json::from_str(body[..end].trim()).map_err(|e| format!("bad tool call: {e}"))?;
        d.tokens.clear();
        d.started = false;
        Ok(Event::Call(ToolCall {
            name: call["name"].as_str().unwrap_or_default().to_string(),
            arguments_json: call["arguments"].to_string(),
        }))
    }

    async fn reset(&mut self, d: Resource<ToolDecoder>) {
        if let Ok(d) = self.table.get_mut(&d) {
            *d = ToolDecoder::default();
        }
    }

    async fn drop(&mut self, d: Resource<ToolDecoder>) -> wasmtime::Result<()> {
        self.table.delete(d)?;
        Ok(())
    }
}
