//! `model`: facts about the model the engine serves.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::model;

impl model::Host for State {
    async fn kv_page_size(&mut self) -> u32 {
        self.engine.page_size
    }
}
