//! The models the engine can run: the families `description` describes,
//! all run by the one transformer in `transformer`.

mod description;
#[cfg(feature = "metal")]
mod paged_attention;
mod transformer;

pub use description::{Description, describe, supports};
pub use transformer::Model;
