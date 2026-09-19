//! A JSON schema, as a regular expression that matches the JSON valid
//! against it. Supported: objects (every property, in order), arrays,
//! strings, integers, numbers, booleans, null and enums. A little whitespace
//! is allowed between elements, since models write `{"a": 1}` rather than
//! `{"a":1}`; bounded, so that a model cannot pad forever.

use anyhow::{Result, bail};
use serde_json::Value;

const STRING: &str = r#""(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*""#;
const INTEGER: &str = r"-?(?:0|[1-9][0-9]*)";
const NUMBER: &str = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";
const WS: &str = r"[ \n]{0,3}";

pub fn to_regex(schema: &Value) -> Result<String> {
    if let Some(options) = schema["enum"].as_array() {
        let options: Vec<String> = options.iter().map(|o| regex_escape(&o.to_string())).collect();
        return Ok(format!("(?:{})", options.join("|")));
    }
    Ok(match schema["type"].as_str() {
        Some("string") => STRING.into(),
        Some("integer") => INTEGER.into(),
        Some("number") => NUMBER.into(),
        Some("boolean") => "(?:true|false)".into(),
        Some("null") => "null".into(),
        Some("array") => {
            let item = to_regex(&schema["items"])?;
            format!(r"\[{WS}(?:{item}(?:{WS},{WS}{item})*)?{WS}\]")
        }
        Some("object") => {
            let Some(properties) = schema["properties"].as_object() else {
                bail!("an object schema needs properties");
            };
            let fields = properties
                .iter()
                .map(|(name, s)| {
                    let name = regex_escape(&Value::from(name.as_str()).to_string());
                    Ok(format!("{name}{WS}:{WS}{}", to_regex(s)?))
                })
                .collect::<Result<Vec<_>>>()?;
            format!(r"\{{{WS}{}{WS}\}}", fields.join(&format!("{WS},{WS}")))
        }
        other => bail!("unsupported schema type {other:?}"),
    })
}

fn regex_escape(literal: &str) -> String {
    let mut out = String::new();
    for c in literal.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}
