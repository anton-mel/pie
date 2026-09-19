//! What clients and `pie --serve` say to each other: the one public,
//! versioned interface, shared by the gateway and the client. Every message
//! is one websocket text frame holding JSON; a program's wasm travels as one
//! binary frame right after the `Install` that announces it.

use serde::{Deserialize, Serialize};

/// UPDATED
/// Bumped whenever a message changes shape.
pub const VERSION: u32 = 2;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
/// UPDATED
pub enum ClientMessage {
    /// Install a program so it can be launched by name. Its wasm follows as
    /// the next frame, a binary one.
    Install { manifest: Manifest },
    /// Start an installed program as a new process, and attach to it.
    Launch { program: String, args: Vec<String> },
    /// Attach to a process: receive what it sent while nobody was attached,
    /// then what it sends next. Closing the connection only detaches: the
    /// process keeps running.
    Attach { process: u64 },
    /// List the processes.
    List,
    /// Stop a process.
    Kill { process: u64 },
    /// A message for the attached process (`session.receive`).
    Message { text: String },
    /// No more messages: the process's next `session.receive` returns none.
    Close,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
/// UPDATED
pub enum ServerMessage {
    /// The first message on every connection.
    Hello {
        version: u32,
    },
    Installed {
        program: String,
        version: String,
    },
    /// A process was started, and this connection is attached to it.
    Launched {
        process: u64,
    },
    Processes {
        processes: Vec<ProcessInfo>,
    },
    Killed {
        process: u64,
    },
    /// A message from the attached process (`session.send`).
    Message {
        text: String,
    },
    /// The attached process returned.
    Result {
        value: String,
    },
    Error {
        message: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
/// NEW
pub struct ProcessInfo {
    pub process: u64,
    pub program: String,
    pub running: bool,
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
