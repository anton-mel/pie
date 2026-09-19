//! `pie model`: models imported once into a local catalog
//! (`~/.pie-tutorial/models.toml`). Importing fetches a model's files and
//! checks it can run: the engine must support its family, and there must be
//! a chat template for it. After that, `--model` takes its short name.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Entry {
    /// The short name `--model` takes.
    pub name: String,
    /// A Hugging Face id or a directory.
    pub source: String,
    pub model_type: String,
}

#[derive(Serialize, Deserialize, Default)]
struct Catalog {
    #[serde(default)]
    model: Vec<Entry>,
}

fn path() -> Result<std::path::PathBuf> {
    Ok(bootstrap::home()?.join("models.toml"))
}

pub fn list() -> Result<Vec<Entry>> {
    let path = path()?;
    if !path.exists() {
        return Ok(vec![]);
    }
    let catalog: Catalog = toml::from_str(&std::fs::read_to_string(path)?)?;
    Ok(catalog.model)
}

/// Fetch `source`, check it can run, and add it to the catalog.
pub fn import(source: &str) -> Result<Entry> {
    // Check from the config alone, before fetching any weights.
    let model_type = worker::weights::model_type(source)?;
    ensure!(
        models::supports(&model_type),
        "{source} is a {model_type:?} model, which this engine cannot run"
    );
    ensure!(
        chat_template::for_model(&model_type).is_some(),
        "there is no chat template for {model_type:?} models"
    );
    worker::weights::Files::find(source)?;
    let name = source.trim_end_matches('/').rsplit('/').next().unwrap_or(source);
    let entry = Entry {
        name: name.to_string(),
        source: source.to_string(),
        model_type,
    };
    let mut model = list()?;
    model.retain(|e| e.name != entry.name);
    model.push(entry.clone());
    std::fs::create_dir_all(bootstrap::home()?)?;
    std::fs::write(path()?, toml::to_string(&Catalog { model })?)?;
    Ok(entry)
}

/// What to load for `model`: a catalog entry's source, or `model` itself.
pub fn resolve(model: &str) -> Result<String> {
    Ok(list()?
        .into_iter()
        .find(|e| e.name == model)
        .map_or(model.to_string(), |e| e.source))
}
