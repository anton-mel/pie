//! A model that calls tools the inferlet runs.
//!
//! Example: `pie agent.wasm -- "question"`.
//!
//! The inferlet offers two tools in the model's own format (`equip`),
//! follows the model's thinking and spots its tool calls as tokens are
//! generated (the `reasoning` and `tools` decoders), runs the tools itself,
//! and gives the results back (`answer`), until the model answers without
//! calling anything.

use inferlet::reasoning::{self, Event as Thought};
use inferlet::tools::{self, Event as Tool, ToolCall};
use inferlet::{Context, chat, greedy, tokenizer};

const WEATHER: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#;
const CALCULATOR: &str = r#"{"type": "function", "function": {"name": "calculate", "description": "Compute a * b, a / b, a + b or a - b", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}"#;

/// The tools themselves: plain inferlet code.
fn run(call: &ToolCall) -> String {
    let args: serde_json::Value = serde_json::from_str(&call.arguments_json).unwrap_or_default();
    match call.name.as_str() {
        "get_weather" => match args["city"].as_str().unwrap_or_default() {
            "Paris" => "18°C, sunny".into(),
            "London" => "12°C, light rain".into(),
            "Tokyo" => "24°C, clear".into(),
            city => format!("no weather data for {city}"),
        },
        "calculate" => {
            let e = args["expression"].as_str().unwrap_or_default().replace(' ', "");
            let op = e
                .find(|c: char| "*/+".contains(c))
                .or_else(|| e[1..].find('-').map(|i| i + 1));
            let Some(i) = op else {
                return format!("cannot compute {e}");
            };
            let (a, b) = (e[..i].parse::<f64>(), e[i + 1..].parse::<f64>());
            let (Ok(a), Ok(b)) = (a, b) else {
                return format!("cannot compute {e}");
            };
            let v = match &e[i..i + 1] {
                "*" => a * b,
                "/" => a / b,
                "+" => a + b,
                _ => a - b,
            };
            format!("{v}")
        }
        name => format!("no tool named {name}"),
    }
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let question = args
            .first()
            .map_or("What is the weather in Paris, and what is 17 * 23?", |s| s);
        let mut ctx = Context::new();
        ctx.system("You are a helpful assistant. Use the tools when they help.");
        ctx.equip(&[WEATHER.into(), CALCULATOR.into()])?;
        ctx.user(question);
        let stop = chat::stop_tokens();
        let mut log = vec![];

        for round in 1..=6 {
            ctx.fill_tokens(&chat::cue());
            let (thinking, calling) = (reasoning::Decoder::new(), tools::Decoder::new());
            let (mut out, mut calls, mut thought) = (vec![], vec![], 0);
            while out.len() < 1024 {
                let next = greedy(&ctx.forward(1)?);
                if stop.contains(&next) {
                    break;
                }
                out.push(next);
                ctx.fill_tokens(&[next]);
                if let Thought::Complete(text) = thinking.feed(&[next]) {
                    thought = text.split_whitespace().count();
                }
                if let Tool::Call(call) = calling.feed(&[next])? {
                    calls.push(call);
                }
            }
            ctx.fill_tokens(&chat::seal());

            if calls.is_empty() {
                let text = tokenizer::detokenize(&out);
                let answer = text.rsplit("</think>").next().unwrap_or(&text).trim();
                log.push(format!("round {round}: thought {thought} words, answered: {answer}"));
                break;
            }
            let mut results = vec![];
            for call in &calls {
                let result = run(call);
                log.push(format!(
                    "round {round}: thought {thought} words, called {}({}) -> {result}",
                    call.name, call.arguments_json
                ));
                results.push(result);
            }
            ctx.fill_tokens(&tools::answer(&results)?);
        }
        Ok(log.join("\n    "))
    }
}

inferlet::export!(App);
