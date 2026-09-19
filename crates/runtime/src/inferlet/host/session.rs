//! `session`: messages with whoever started the inferlet.

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::session;

impl session::Host for State {
    async fn send(&mut self, message: String) {
        let _ = self.session.out.send(message);
    }

    async fn receive(&mut self) -> Option<String> {
        self.session.inbox.lock().await.recv().await
    }
}
