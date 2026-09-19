//! `chat`: the model's chat format.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::chat;

/// The chat format of Qwen models (ChatML). Other model families write
/// their turns differently; the reference reads each model's own template.
impl chat::Host for State {
    async fn system(&mut self, message: String) -> Vec<u32> {
        self.turn("system", &message)
    }

    async fn user(&mut self, message: String) -> Vec<u32> {
        self.turn("user", &message)
    }

    async fn assistant(&mut self, message: String) -> Vec<u32> {
        self.turn("assistant", &message)
    }

    async fn cue(&mut self) -> Vec<u32> {
        self.encode("<|im_start|>assistant\n")
    }

    async fn seal(&mut self) -> Vec<u32> {
        self.encode("<|im_end|>\n")
    }

    async fn stop_tokens(&mut self) -> Vec<u32> {
        self.engine.eos.clone()
    }
}

impl State {
    fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.engine.tokenizer.encode(text, false);
        encoding.map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }

    fn turn(&self, role: &str, message: &str) -> Vec<u32> {
        self.encode(&format!("<|im_start|>{role}\n{message}<|im_end|>\n"))
    }
}
