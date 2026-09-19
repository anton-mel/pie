//! Pull fields out of text as JSON that always fits a schema.
//!
//! Example: `pie json_extract.wasm -- "text" ['{"type":"object",...}']`.
//!
//! The model is asked for JSON, and a grammar made from the schema allows
//! at each step only the tokens that keep the output valid against it. The
//! result always parses, with the fields in the schema's order.

use inferlet::{Context, Matcher, greedy};

const TEXT: &str = "Alice Moreau is 34 years old, lives in Lyon and works as a nurse.";
const SCHEMA: &str = r#"{"type": "object", "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "city": {"type": "string"},
    "job": {"enum": ["doctor", "nurse", "teacher", "engineer", "other"]}
}}"#;

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let text = args.first().map_or(TEXT, |s| s);
        let schema = args.get(1).map_or(SCHEMA, |s| s);
        let matcher = Matcher::from_json_schema(schema)?;

        let mut ctx = Context::new();
        ctx.system("Extract the requested fields from the text as JSON.");
        ctx.user(&format!("Text: {text}\nSchema: {schema} /no_think"));
        // Open the reply as `reply` would, then write it under the grammar.
        ctx.fill_tokens(&inferlet::chat::cue());
        ctx.fill("<think>\n\n</think>\n\n");
        ctx.generate_matching(&matcher, 128, greedy)
    }
}

inferlet::export!(App);
