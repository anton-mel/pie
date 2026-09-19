//! `chat`: the model's chat format, from its template (`chat-template`).

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::chat;
use chat_template::Role;

/// Renders through the model's template instead of writing ChatML by hand.
impl chat::Host for State {
    async fn prefix(&mut self) -> Vec<u32> {
        self.encode(self.engine.template.prefix())
    }

    async fn system(&mut self, message: String) -> Vec<u32> {
        self.encode(&self.engine.template.turn(Role::System, &message))
    }

    async fn user(&mut self, message: String) -> Vec<u32> {
        self.encode(&self.engine.template.turn(Role::User, &message))
    }

    async fn assistant(&mut self, message: String) -> Vec<u32> {
        self.encode(&self.engine.template.turn(Role::Assistant, &message))
    }

    async fn system_user(&mut self, system: String, user: String) -> Vec<u32> {
        self.encode(&self.engine.template.system_user(&system, &user))
    }

    async fn cue(&mut self) -> Vec<u32> {
        self.encode(self.engine.template.cue())
    }

    async fn seal(&mut self) -> Vec<u32> {
        self.encode(self.engine.template.seal())
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
}
