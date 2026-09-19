//! `tokenizer`: text to tokens and back.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::tokenizer;

impl tokenizer::Host for State {
    async fn tokenize(&mut self, text: String) -> Vec<u32> {
        self.engine
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    async fn detokenize(&mut self, tokens: Vec<u32>) -> String {
        self.engine.tokenizer.decode(&tokens, false).unwrap_or_default()
    }

    async fn eos_tokens(&mut self) -> Vec<u32> {
        self.engine.eos.clone()
    }
}
