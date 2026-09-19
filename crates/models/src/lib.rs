//! The models the engine can run: for now one, the Qwen2/Qwen3 family.

mod qwen;

pub use qwen::{Config, Model, Seq};
