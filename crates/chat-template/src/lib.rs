//! How a conversation is written as text, per model family. A model's config
//! names its family (`model_type`), `for_model` picks the matching template,
//! and the tokenizer turns the template's text into tokens.

pub enum Role {
    System,
    User,
    Assistant,
}

pub trait Template: Send + Sync {
    /// The text a conversation starts with (a begin-of-text marker), if any.
    fn prefix(&self) -> &'static str;
    /// One whole message of `role`.
    fn turn(&self, role: Role, message: &str) -> String;
    /// A system prompt and the first user message. Most formats write them
    /// as two turns; some fold the system prompt into the user's message.
    fn system_user(&self, system: &str, user: &str) -> String {
        self.turn(Role::System, system) + &self.turn(Role::User, user)
    }
    /// Start the assistant's reply: what the model writes next is its answer.
    fn cue(&self) -> &'static str;
    /// Close the assistant's reply after it was generated.
    fn seal(&self) -> &'static str;

    /// NEW
    /// Text to add to the system prompt to offer `tools` (each a JSON
    /// function description), or none if this format has no tools here.
    fn tools(&self, _tools: &[String]) -> Option<String> {
        None
    }

    /// NEW
    /// The results of the tools called in one turn, written as the model
    /// expects to read them.
    fn tool_results(&self, _values: &[String]) -> Option<String> {
        None
    }

    /// NEW
    /// What a tool call starts and ends with in the model's output.
    fn tool_call_markers(&self) -> Option<(&'static str, &'static str)> {
        None
    }

    /// NEW
    /// What thinking starts and ends with in the model's output.
    fn thinking_markers(&self) -> Option<(&'static str, &'static str)> {
        None
    }
}

/// The template of a model family, by its config's `model_type`.
pub fn for_model(model_type: &str) -> Option<Box<dyn Template>> {
    Some(match model_type {
        "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe" => Box::new(ChatMl),
        "llama" => Box::new(Llama3),
        "gemma" | "gemma2" | "gemma3" | "gemma3_text" => Box::new(Gemma),
        "deepseek_v3" => Box::new(DeepSeek),
        _ => return None,
    })
}

/// The template whose markers appear in a model's own chat template text
/// (the Jinja in its `tokenizer_config.json`).
pub fn detect(chat_template: &str) -> Option<Box<dyn Template>> {
    Some(if chat_template.contains("<|im_start|>") {
        Box::new(ChatMl)
    } else if chat_template.contains("<|start_header_id|>") {
        Box::new(Llama3)
    } else if chat_template.contains("<start_of_turn>") {
        Box::new(Gemma)
    } else if chat_template.contains("<｜User｜>") {
        Box::new(DeepSeek)
    } else {
        return None;
    })
}

/// ChatML, used by Qwen: `<|im_start|>role\nmessage<|im_end|>\n`.
pub struct ChatMl;

impl Template for ChatMl {
    fn prefix(&self) -> &'static str {
        ""
    }
    fn turn(&self, role: Role, message: &str) -> String {
        let role = match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        format!("<|im_start|>{role}\n{message}<|im_end|>\n")
    }
    fn cue(&self) -> &'static str {
        "<|im_start|>assistant\n"
    }
    fn seal(&self) -> &'static str {
        "<|im_end|>\n"
    }
    /// UPDATED
    /// Tools and thinking as Qwen3's own template writes them.
    fn tools(&self, tools: &[String]) -> Option<String> {
        Some(format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments within \
             <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": \
             <args-json-object>}}\n</tool_call>",
            tools.join("\n")
        ))
    }
    fn tool_results(&self, values: &[String]) -> Option<String> {
        let responses: Vec<_> = values
            .iter()
            .map(|v| format!("<tool_response>\n{v}\n</tool_response>"))
            .collect();
        Some(format!("<|im_start|>user\n{}<|im_end|>\n", responses.join("\n")))
    }
    fn tool_call_markers(&self) -> Option<(&'static str, &'static str)> {
        Some(("<tool_call>", "</tool_call>"))
    }
    fn thinking_markers(&self) -> Option<(&'static str, &'static str)> {
        Some(("<think>", "</think>"))
    }
}

/// Llama 3: headers around each role, `<|eot_id|>` after each message.
pub struct Llama3;

impl Template for Llama3 {
    fn prefix(&self) -> &'static str {
        "<|begin_of_text|>"
    }
    fn turn(&self, role: Role, message: &str) -> String {
        let role = match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        format!("<|start_header_id|>{role}<|end_header_id|>\n\n{message}<|eot_id|>")
    }
    fn cue(&self) -> &'static str {
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    }
    fn seal(&self) -> &'static str {
        "<|eot_id|>"
    }
}

/// Gemma: `<start_of_turn>role\n…<end_of_turn>\n`, with the assistant called
/// `model`. Gemma has no system role: its template folds the system prompt
/// into the first user message (`system_user`); alone, it is a user turn.
pub struct Gemma;

impl Template for Gemma {
    fn prefix(&self) -> &'static str {
        "<bos>"
    }
    fn turn(&self, role: Role, message: &str) -> String {
        let role = match role {
            Role::System | Role::User => "user",
            Role::Assistant => "model",
        };
        format!("<start_of_turn>{role}\n{message}<end_of_turn>\n")
    }
    fn system_user(&self, system: &str, user: &str) -> String {
        self.turn(Role::User, &format!("{system}\n\n{user}"))
    }
    fn cue(&self) -> &'static str {
        "<start_of_turn>model\n"
    }
    fn seal(&self) -> &'static str {
        "<end_of_turn>\n"
    }
}

/// DeepSeek V3: the system prompt as plain text, then `<｜User｜>` and
/// `<｜Assistant｜>` markers, with an end-of-sentence marker after a reply.
pub struct DeepSeek;

impl Template for DeepSeek {
    fn prefix(&self) -> &'static str {
        "<｜begin▁of▁sentence｜>"
    }
    fn turn(&self, role: Role, message: &str) -> String {
        match role {
            Role::System => message.to_string(),
            Role::User => format!("<｜User｜>{message}"),
            Role::Assistant => format!("<｜Assistant｜>{message}<｜end▁of▁sentence｜>"),
        }
    }
    fn cue(&self) -> &'static str {
        "<｜Assistant｜>"
    }
    fn seal(&self) -> &'static str {
        "<｜end▁of▁sentence｜>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A system prompt, a user message, an answer and a second question, then
    /// the cue. The expected strings below are what each family's official
    /// chat template (from its Hugging Face `tokenizer_config.json`, rendered
    /// with Jinja) produces for the same conversation.
    fn render(t: &dyn Template) -> String {
        let mut s = t.prefix().to_string();
        s += &t.system_user("Be brief.", "Hi");
        s += &t.turn(Role::Assistant, "Hello!");
        s += &t.turn(Role::User, "Bye");
        s + t.cue()
    }

    #[test]
    fn chatml() {
        assert_eq!(
            render(&ChatMl),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3() {
        assert_eq!(
            render(&Llama3),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn gemma() {
        assert_eq!(
            render(&Gemma),
            "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n\
             <start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn deepseek() {
        assert_eq!(
            render(&DeepSeek),
            "<｜begin▁of▁sentence｜>Be brief.<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜><｜User｜>Bye<｜Assistant｜>"
        );
    }

    #[test]
    fn detects_from_template_text() {
        let text = |t: &dyn Template| render(t);
        assert_eq!(text(&*detect("{{ '<|im_start|>' + role }}").unwrap()), render(&ChatMl));
        assert_eq!(text(&*detect("<|start_header_id|>").unwrap()), render(&Llama3));
        assert!(detect("{{ messages }}").is_none());
    }

    /// What Qwen3's official template writes for a system prompt with one
    /// tool, a call and its result.
    #[test]
    fn chatml_tools() {
        let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#;
        let t = ChatMl;
        let mut s = t.turn(Role::System, &format!("Be brief.{}", t.tools(&[tool.into()]).unwrap()));
        s += &t.turn(Role::User, "Weather in Paris?");
        s += &t.turn(
            Role::Assistant,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        );
        s += &t.tool_results(&["18C, sunny".into()]).unwrap();
        s += t.cue();
        let expected = format!(
            "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\nWeather in Paris?<|im_end|>\n\
             <|im_start|>assistant\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Paris\"}}}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn families() {
        assert!(for_model("qwen3").is_some());
        assert!(for_model("llama").is_some());
        assert!(for_model("gemma3").is_some());
        assert!(for_model("no-such-family").is_none());
    }
}
