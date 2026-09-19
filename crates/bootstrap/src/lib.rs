//! What every `pie` command starts from: where its files live, its config
//! file, and the `/metrics` endpoint.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Where `pie` keeps its files: `~/.pie-tutorial`.
pub fn home() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".pie-tutorial"))
}

/// The config file. Every field has a default, and command-line flags
/// override what is here.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    /// A model from `pie model list`, a Hugging Face id, or a directory.
    pub model: String,
    pub kv_pages: u32,
    pub page_size: usize,
    /// Most tokens in one model step.
    pub step_tokens: usize,
    pub cpu: bool,
    /// Where `pie serve` listens.
    pub addr: String,
    /// Where `pie serve` serves `/metrics`, if anywhere.
    pub metrics: Option<String>,
    pub sandbox: Sandbox,
}

/// What inferlets may reach besides the model (chapter 19).
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(default)]
pub struct Sandbox {
    pub dir: Option<PathBuf>,
    pub writable: bool,
    pub connect: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen3-0.6B".into(),
            kv_pages: 1024,
            page_size: 16,
            step_tokens: 256,
            cpu: false,
            addr: "127.0.0.1:9123".into(),
            metrics: None,
            sandbox: Sandbox::default(),
        }
    }
}

impl Config {
    pub fn path() -> Result<PathBuf> {
        Ok(home()?.join("config.toml"))
    }

    /// The config file, or the defaults if there is none.
    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        toml::from_str(&text).with_context(|| format!("reading {}", path.display()))
    }

    /// Write the defaults to the config file. Refuses to overwrite one
    /// unless `force`.
    pub fn init(force: bool) -> Result<PathBuf> {
        let path = Self::path()?;
        if path.exists() && !force {
            bail!("{} exists (use --force to overwrite it)", path.display());
        }
        std::fs::create_dir_all(home()?)?;
        std::fs::write(&path, toml::to_string(&Self::default())?)?;
        Ok(path)
    }
}

/// Serve `GET /metrics` on `addr`, answering with what `render` returns.
pub async fn serve_metrics(addr: String, render: impl Fn() -> String + Send + Sync + 'static) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("metrics on http://{addr}/metrics");
    loop {
        let (mut conn, _) = listener.accept().await?;
        let mut request = [0u8; 1024];
        let n = conn.read(&mut request).await.unwrap_or(0);
        let request = String::from_utf8_lossy(&request[..n]);
        let response = if request.starts_with("GET /metrics") {
            let body = render();
            format!(
                "HTTP/1.0 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
        } else {
            "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n".into()
        };
        let _ = conn.write_all(response.as_bytes()).await;
    }
}
