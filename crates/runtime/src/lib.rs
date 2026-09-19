//! The Pie runtime: runs inferlets (`inferlet`) against one model, with a
//! KV page pool and a queue of forwards (`engine`), a scheduler that turns
//! the queue into model steps (`scheduler`), a planner for when pages run
//! out (`planner`), and shared prompt prefixes (`store`). Serving remote clients is the gateway's job.

pub mod engine;
pub mod inferlet;
pub mod planner;
pub mod scheduler;
pub mod store;
