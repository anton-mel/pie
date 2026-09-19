//! The `pie` binary: load the model, start the engine, and run N copies of
//! an inferlet at once (fixed at launch for now), printing each one's output.

mod engine;
mod host;
mod model;
mod planner;
mod scheduler;
mod server;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Parse args.
#[derive(Parser)]
struct Args {
    /// Path to the inferlet (.wasm component). Not needed with `--serve`.
    inferlet: Option<String>,
    /// Serve inferlets sent by `pie-client` on this address, instead of
    /// running one.
    #[arg(long)]
    serve: Option<String>,
    /// Hugging Face model id or local directory.
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    /// How many copies of the inferlet to run at once.
    #[arg(short, long, default_value_t = 1)]
    instances: usize,
    #[arg(long, default_value_t = 1024)]
    kv_pages: u32, // configured based on your PC
    #[arg(long, default_value_t = 16)]
    page_size: usize,
    /// Most tokens in one model step. Longer prefills are split.
    #[arg(long, default_value_t = 256)]
    step_tokens: usize,
    #[arg(long)]
    cpu: bool,
    /// Run the instances one after another instead of all at once.
    #[arg(short, long)]
    sequential: bool,
    /// Arguments passed to the inferlet.
    #[arg(last = true)]
    args: Vec<String>,
    // passed via `--`
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::metal_if_available(0)?
    };
    let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };

    let file = |name: &str| -> Result<PathBuf> {
        let local = PathBuf::from(&args.model).join(name);
        if local.exists() {
            return Ok(local);
        }
        let repo = hf_hub::api::sync::Api::new()?.model(args.model.clone());
        repo.get(name)
            .with_context(|| format!("fetching {name} from {}", args.model))
    };
    let json = |name: &str| -> Result<serde_json::Value> { Ok(serde_json::from_slice(&std::fs::read(file(name)?)?)?) };

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
    let tokenizer = tokenizers::Tokenizer::from_file(file("tokenizer.json")?).map_err(anyhow::Error::msg)?;
    let eos_value = json("generation_config.json")
        .map(|g| g["eos_token_id"].clone())
        .unwrap_or(config["eos_token_id"].clone());
    let eos: Vec<u32> = match &eos_value {
        serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        v => v.as_u64().map(|v| v as u32).into_iter().collect(),
    };

    let t = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let model = model::Model::load(
        &serde_json::from_value(config)?,
        vb,
        args.kv_pages as usize,
        args.page_size,
    )?;
    eprintln!("loaded {} on {:?} in {:.1?}", args.model, device, t.elapsed());

    let engine = Arc::new(engine::Engine::new(
        model,
        tokenizer,
        eos,
        args.kv_pages,
        args.step_tokens,
    ));
    let host = Arc::new(host::Host::new(engine)?);
    if let Some(addr) = &args.serve {
        return server::serve(host, addr).await;
    }
    let component = host.load(args.inferlet.as_deref().context("give an inferlet or --serve")?)?;
    let session = terminal();

    let t = Instant::now();
    let mut running = vec![];
    let mut done = vec![];
    for _ in 0..args.instances {
        let (host, component, inferlet_args) = (host.clone(), component.clone(), args.args.clone());
        let session = session.clone();
        let run = tokio::spawn(async move {
            let started = Instant::now();
            let result = host.run(&component, inferlet_args, session).await;
            (started.elapsed(), result)
        });
        // Sequential: finish this one before starting the next.
        if args.sequential {
            done.push(run.await?)
        } else {
            running.push(run)
        }
    }
    for run in running {
        done.push(run.await?);
    }
    for (i, (took, result)) in done.into_iter().enumerate() {
        match result? {
            Ok(out) => println!("[{i}] ({took:.1?}) {out}"),
            Err(e) => println!("[{i}] error: {e}"),
        }
    }
    eprintln!("{} run(s) in {:.1?}", args.instances, t.elapsed());
    Ok(())
}

/// The session of a local run: messages are printed as they come, and each
/// line typed on stdin is a message for the inferlet.
fn terminal() -> host::Session {
    use std::io::{BufRead, Write};
    let (out, mut outbox) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(message) = outbox.recv().await {
            print!("{message}");
            let _ = std::io::stdout().flush();
        }
    });
    let (to_inbox, inbox) = tokio::sync::mpsc::unbounded_channel();
    // A plain thread: reading stdin blocks, and must not keep `pie` alive.
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if to_inbox.send(line).is_err() {
                break;
            }
        }
    });
    host::Session {
        out,
        inbox: std::sync::Arc::new(tokio::sync::Mutex::new(inbox)),
    }
}
