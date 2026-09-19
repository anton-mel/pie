//! The worker: the role that owns the GPU. It loads the model, opens it as
//! an engine, and builds the runtime on top. Whoever starts `pie` gets back
//! a running host, and does not deal with checkpoints or devices.

pub mod weights;

use anyhow::{Context, Result};
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
    /// What inferlets may reach besides the model.
    pub policy: runtime::inferlet::Policy,
}

/// Refuses an unsupported model from its config, before fetching weights.
/// Load the model and start the runtime on it.
pub fn start(config: &Config) -> Result<Arc<Host>> {
    let device = if config.cpu {
        Device::Cpu
    } else {
        Device::metal_if_available(0)?
    };
    let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
    let model_type = weights::model_type(&config.model)?;
    anyhow::ensure!(
        models::supports(&model_type),
        "{} is a {model_type:?} model, which this engine cannot run",
        config.model
    );
    let files = weights::Files::find(&config.model)?;

    let t = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
    let model = models::Model::load(&files.description, vb, config.kv_pages as usize, config.page_size)?;
    eprintln!("loaded {} on {:?} in {:.1?}", config.model, device, t.elapsed());

    // The model's own template text says which format it is; the family is
    // only a fallback (a "llama" model can speak ChatML).
    let template = files
        .chat_template
        .as_deref()
        .and_then(chat_template::detect)
        .or_else(|| chat_template::for_model(&files.model_type))
        .with_context(|| format!("no chat template for {}", config.model))?;
    let engine: Box<dyn engine::Engine> = Box::new(model);
    let runtime = runtime::engine::Engine::new(
        engine,
        files.tokenizer,
        template,
        files.eos,
        config.kv_pages,
        config.step_tokens,
    );
    Host::new(Arc::new(runtime), config.policy.clone()).map(Arc::new)
}
