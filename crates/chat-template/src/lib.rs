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
    fn families() {
        assert!(for_model("qwen3").is_some());
        assert!(for_model("llama").is_some());
        assert!(for_model("gemma3").is_some());
        assert!(for_model("no-such-family").is_none());
    }
}
