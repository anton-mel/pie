//! The worker: the role that owns the GPU. It loads the model, opens it as
//! an engine, and builds the runtime on top. Whoever starts `pie` gets back
//! a running host, and does not deal with checkpoints or devices.

mod weights;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use runtime::inferlet::Host;
use std::sync::Arc;
use std::time::Instant;

/// What the worker needs to know to start.
pub struct Config {
    /// Hugging Face model id or local directory.
    pub model: String,
    pub kv_pages: u32,
    pub page_size: usize,
    /// Most tokens in one model step.
    pub step_tokens: usize,
    /// Run on the CPU instead of the GPU.
    pub cpu: bool,
    /// NEW
    /// What inferlets may reach besides the model.
    pub policy: runtime::inferlet::Policy,
}

/// Load the model and start the runtime on it.
pub fn start(config: &Config) -> Result<Arc<Host>> {
    let device = if config.cpu {
        Device::Cpu
    } else {
        Device::metal_if_available(0)?
    };
    let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
    let files = weights::Files::find(&config.model)?;

    let t = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
    let model = models::Model::load(&files.config, vb, config.kv_pages as usize, config.page_size)?;
    eprintln!("loaded {} on {:?} in {:.1?}", config.model, device, t.elapsed());

    let engine: Box<dyn engine::Engine> = Box::new(model);
    let runtime = runtime::engine::Engine::new(engine, files.tokenizer, files.eos, config.kv_pages, config.step_tokens);
    Host::new(Arc::new(runtime), config.policy.clone()).map(Arc::new)
}
