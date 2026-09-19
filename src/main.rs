//! The `pie` binary: start a worker, then run N copies of an inferlet at
//! once and print each one's output, or serve clients.

use anyhow::{Context, Result};
use clap::Parser;
use std::time::Instant;

/// Parse args.
#[derive(Parser)]
struct Args {
    /// Path to the inferlet (.wasm component). Not needed with `--serve`.
    inferlet: Option<String>,
    /// UPDATED
    /// Serve programs to `pie-client` on this address, instead of running
    /// one.
    #[arg(long)]
    serve: Option<String>,
    /// NEW
    /// Where installed programs are kept, with `--serve`. Defaults to
    /// `~/.pie-tutorial/programs`.
    #[arg(long)]
    programs: Option<std::path::PathBuf>,
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
    let host = worker::start(&worker::Config {
        model: args.model.clone(),
        kv_pages: args.kv_pages,
        page_size: args.page_size,
        step_tokens: args.step_tokens,
        cpu: args.cpu,
    })?;
    if let Some(addr) = &args.serve {
        let home = std::env::var("HOME").context("HOME is not set")?;
        let dir = args
            .programs
            .clone()
            .unwrap_or_else(|| format!("{home}/.pie-tutorial/programs").into());
        let programs = runtime::inferlet::Programs::open(dir)?;
        return gateway::serve(host, std::sync::Arc::new(programs), addr).await;
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
fn terminal() -> runtime::inferlet::Session {
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
    runtime::inferlet::Session {
        out,
        inbox: std::sync::Arc::new(tokio::sync::Mutex::new(inbox)),
    }
}
