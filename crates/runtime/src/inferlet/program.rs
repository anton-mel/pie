//! Installed programs: each inferlet's wasm and manifest, kept as
//! `<dir>/<name>/<version>.wasm` and `.toml` like the reference, so a
//! program is sent once and then started by name (its latest version).

use super::Host;
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use wasmtime::component::Component;

pub struct Programs {
    dir: PathBuf,
    /// Programs compiled since the server started.
    compiled: Mutex<HashMap<String, Component>>,
}

impl Programs {
    pub fn open(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self {
            dir,
            compiled: Mutex::default(),
        })
    }

    /// Check that `wasm` is an inferlet, then keep it and its manifest as
    /// `name` at `version`, replacing that version if it was there.
    pub fn install(&self, host: &Host, name: &str, version: &str, manifest: &str, wasm: &[u8]) -> Result<()> {
        let plain = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
        ensure!(plain(name) && !name.starts_with('.'), "bad program name {name:?}");
        ensure!(plain(version) && !version.starts_with('.'), "bad version {version:?}");
        let component = host.compile(wasm)?;
        let dir = self.dir.join(name);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(format!("{version}.wasm")), wasm)?;
        std::fs::write(dir.join(format!("{version}.toml")), manifest)?;
        // A new install is the latest version: drop the compiled older one.
        self.compiled.lock().unwrap().insert(name.to_string(), component);
        Ok(())
    }

    /// The program installed as `name`, compiled.
    pub fn get(&self, host: &Host, name: &str) -> Result<Component> {
        if let Some(component) = self.compiled.lock().unwrap().get(name) {
            return Ok(component.clone());
        }
        let path = self
            .latest(name)
            .with_context(|| format!("program {name:?} is not installed"))?;
        let component = host.load(path.to_str().context("program path")?)?;
        self.compiled
            .lock()
            .unwrap()
            .insert(name.to_string(), component.clone());
        Ok(component)
    }

    /// The wasm of the highest version installed as `name`.
    fn latest(&self, name: &str) -> Option<PathBuf> {
        let key = |p: &PathBuf| -> Vec<u64> {
            let stem = p.file_stem().unwrap_or_default().to_string_lossy();
            stem.split('.').map(|n| n.parse().unwrap_or(0)).collect()
        };
        std::fs::read_dir(self.dir.join(name))
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wasm"))
            .max_by_key(key)
    }
}
