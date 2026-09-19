//! What clients and `pie --serve` say to each other: the one public,
//! versioned interface, shared by the gateway and the client. Every message
//! is one websocket text frame holding JSON; a program's wasm travels as one
//! binary frame right after the `Install` that announces it.

use serde::{Deserialize, Serialize};

/// Bumped whenever a message changes shape.
pub const VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Install a program so it can be launched by name. Its wasm follows as
    /// the next frame, a binary one.
    Install { manifest: Manifest },
    /// Start an installed program. The connection is then its session.
    Launch { program: String, args: Vec<String> },
    /// A message for the running program (`session.receive`).
    Message { text: String },
    /// No more messages: the program's next `session.receive` returns none.
    Close,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// The first message on every connection.
    Hello {
        version: u32,
    },
    Installed {
        program: String,
        version: String,
    },
    /// A message from the running program (`session.send`).
    Message {
        text: String,
    },
    /// The program returned.
    Result {
        value: String,
    },
    Error {
        message: String,
    },
}

/// A program's manifest: the `Pie.toml` next to its sources.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub package: Package,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("a manifest is always valid TOML")
    }
}
