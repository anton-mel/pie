//! Where a model's files are: a local directory, or the Hugging Face hub
//! (downloaded the first time, then read from its cache).

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct Files {
    pub config: models::Config,
    /// NEW
    /// The model family, as its config names it (`model_type`).
    pub model_type: String,
    pub weights: Vec<PathBuf>,
    pub tokenizer: Tokenizer,
    pub eos: Vec<u32>,
}

impl Files {
    pub fn find(model: &str) -> Result<Self> {
        let file = |name: &str| -> Result<PathBuf> {
            let local = PathBuf::from(model).join(name);
            if local.exists() {
                return Ok(local);
            }
            let repo = hf_hub::api::sync::Api::new()?.model(model.to_string());
            repo.get(name).with_context(|| format!("fetching {name} from {model}"))
        };
        let json = |name: &str| -> Result<Value> { Ok(serde_json::from_slice(&std::fs::read(file(name)?)?)?) };

        let config = json("config.json")?;
        let weights = match file("model.safetensors.index.json") {
            Ok(_) => {
                let index = json("model.safetensors.index.json")?;
                let mut shards: Vec<String> = index["weight_map"]
                    .as_object()
                    .context("weight_map")?
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                shards.sort();
                shards.dedup();
                shards.iter().map(|s| file(s)).collect::<Result<Vec<_>>>()?
            }
            Err(_) => vec![file("model.safetensors")?],
        };
        let tokenizer = Tokenizer::from_file(file("tokenizer.json")?).map_err(anyhow::Error::msg)?;
        let eos_value = json("generation_config.json")
            .map(|g| g["eos_token_id"].clone())
            .unwrap_or(config["eos_token_id"].clone());
        let eos: Vec<u32> = match &eos_value {
            Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
            v => v.as_u64().map(|v| v as u32).into_iter().collect(),
        };
        Ok(Self {
            model_type: config["model_type"].as_str().unwrap_or_default().to_string(),
            config: serde_json::from_value(config)?,
            weights,
            tokenizer,
            eos,
        })
    }
}
