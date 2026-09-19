//! The models the engine can run: for now one, the Qwen2/Qwen3 family.

#[cfg(feature = "metal")]
mod paged_attention;
mod qwen;

pub use qwen::{Config, Model};

/// Whether a model family, as its config names it (`model_type`), can run.
pub fn supports(model_type: &str) -> bool {
    matches!(model_type, "qwen2" | "qwen3")
}
