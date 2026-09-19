//! What differs between model families, said once per family. The
//! transformer in `transformer.rs` is written once and follows the
//! description, the way the reference describes every family once and runs
//! them all with one engine.

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// One model, as the transformer needs to know it.
pub struct Description {
    pub family: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub norm_eps: f64,
    /// Biases on the query, key and value projections (Qwen2).
    pub qkv_bias: bool,
    /// An RMS norm on each query and key head (Qwen3).
    pub qk_norm: bool,
    /// The output head reuses the embedding matrix.
    pub tie_embeddings: bool,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
}

/// Llama 3's frequency scaling: long wavelengths are stretched by `factor`,
/// short ones kept, and the ones between blended.
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_context: f64,
}

/// The families the transformer can run, by their config's `model_type`.
const FAMILIES: [&str; 3] = ["qwen2", "qwen3", "llama"];

pub fn supports(model_type: &str) -> bool {
    FAMILIES.contains(&model_type)
}

/// Read a model's description from its `config.json`.
pub fn describe(config: &Value) -> Result<Description> {
    let family = config["model_type"].as_str().unwrap_or_default().to_string();
    let int = |key: &str| {
        config[key]
            .as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config has no {key}"))
    };
    let float = |key: &str, default: f64| config[key].as_f64().unwrap_or(default);
    let flag = |key: &str| config[key].as_bool().unwrap_or(false);

    let (qkv_bias, qk_norm) = match family.as_str() {
        "qwen2" => (true, false),
        "qwen3" => (false, true),
        "llama" => (flag("attention_bias"), false),
        other => bail!("no description for {other:?} models"),
    };
    let rope_scaling = match &config["rope_scaling"] {
        s if s["rope_type"] == "llama3" => Some(RopeScaling {
            factor: s["factor"].as_f64().context("rope_scaling.factor")?,
            low_freq_factor: s["low_freq_factor"].as_f64().unwrap_or(1.0),
            high_freq_factor: s["high_freq_factor"].as_f64().unwrap_or(4.0),
            original_context: s["original_max_position_embeddings"].as_f64().unwrap_or(8192.0),
        }),
        Value::Null => None,
        other => bail!("unsupported rope_scaling {other}"),
    };
    let (hidden, heads) = (int("hidden_size")?, int("num_attention_heads")?);
    Ok(Description {
        family,
        hidden,
        intermediate: int("intermediate_size")?,
        layers: int("num_hidden_layers")?,
        heads,
        kv_heads: int("num_key_value_heads")?,
        head_dim: int("head_dim").unwrap_or(hidden / heads),
        vocab: int("vocab_size")?,
        norm_eps: float("rms_norm_eps", 1e-6),
        qkv_bias,
        qk_norm,
        tie_embeddings: flag("tie_word_embeddings"),
        rope_theta: float("rope_theta", 10000.0),
        rope_scaling,
    })
}

impl Description {
    /// The rotary inverse frequencies, scaled if the family scales them.
    pub fn inv_freq(&self) -> Vec<f32> {
        let d = self.head_dim as f64;
        (0..self.head_dim / 2)
            .map(|i| {
                let f = 1.0 / self.rope_theta.powf(2.0 * i as f64 / d);
                let Some(s) = &self.rope_scaling else {
                    return f as f32;
                };
                let wavelen = 2.0 * std::f64::consts::PI / f;
                let (low, high) = (
                    s.original_context / s.low_freq_factor,
                    s.original_context / s.high_freq_factor,
                );
                let f = if wavelen < high {
                    f
                } else if wavelen > low {
                    f / s.factor
                } else {
                    let smooth =
                        (s.original_context / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
                    (1.0 - smooth) * f / s.factor + smooth * f
                };
                f as f32
            })
            .collect()
    }
}
