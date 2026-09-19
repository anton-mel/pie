//! The `pie` binary. `pie run` runs an inferlet next to the model and prints
//! what it returns; `pie serve` serves programs to `pie-client`; `pie model`
//! and `pie config` look after the model catalog and the config file.

mod catalog;

use anyhow::{Context, Result};
use bootstrap::Config;
use clap::{Args, Parser, Subcommand};
use runtime::inferlet::Host;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pie")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Subcommands, as in the reference.
#[derive(Subcommand)]
enum Command {
    /// Run an inferlet next to the model, and print what it returns.
    Run {
        /// Path to the inferlet (.wasm component).
        inferlet: String,
        /// How many copies of the inferlet to run at once.
        #[arg(short, long, default_value_t = 1)]
        instances: usize,
        /// Run the copies one after another instead of all at once.
        #[arg(short, long)]
        sequential: bool,
        #[command(flatten)]
        engine: EngineArgs,
        /// Arguments passed to the inferlet.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Serve programs to `pie-client`.
    Serve {
        /// Where to listen (default from the config: 127.0.0.1:9123).
        #[arg(long)]
        addr: Option<String>,
        /// Where installed programs are kept (default
        /// `~/.pie-tutorial/programs`).
        #[arg(long)]
        programs: Option<PathBuf>,
        /// Also serve `/metrics` on this address.
        #[arg(long)]
        metrics: Option<String>,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// NEW
    /// Route clients to workers (`pie worker`) that register here.
    Gateway {
        #[arg(long, default_value = "127.0.0.1:9123")]
        addr: String,
    },
    /// NEW
    /// Load the model and serve programs, registered with a gateway.
    Worker {
        /// The gateway to register with.
        #[arg(long)]
        gateway: String,
        /// Where this worker serves; the gateway connects here.
        #[arg(long)]
        addr: String,
        #[arg(long)]
        programs: Option<PathBuf>,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// The local model catalog.
    #[command(subcommand)]
    Model(ModelCommand),
    /// The config file (`~/.pie-tutorial/config.toml`).
    #[command(subcommand)]
    Config(ConfigCommand),
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Fetch a model and check the engine can run it.
    Import { model: String },
    /// List imported models.
    List,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Write the defaults to the config file.
    Init {
        #[arg(long)]
        force: bool,
    },
    /// Print the config in effect.
    Show,
}

/// Overrides of the config file, for one command.
#[derive(Args)]
struct EngineArgs {
    /// A model from `pie model list`, a Hugging Face id, or a directory.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    kv_pages: Option<u32>,
    #[arg(long)]
    page_size: Option<usize>,
    /// Most tokens in one model step. Longer prefills are split.
    #[arg(long)]
    step_tokens: Option<usize>,
    #[arg(long)]
    cpu: bool,
    /// A directory inferlets may read, as `/data`.
    #[arg(long)]
    allow_dir: Option<PathBuf>,
    /// Let inferlets also write in `--allow-dir`.
    #[arg(long)]
    allow_write: bool,
    /// An address (host:port) inferlets may open TCP connections to.
    #[arg(long)]
    allow_connect: Vec<String>,
}

impl EngineArgs {
    /// The config file, with these flags on top.
    fn config(&self) -> Result<Config> {
        let mut c = Config::load()?;
        c.model = self.model.clone().unwrap_or(c.model);
        c.kv_pages = self.kv_pages.unwrap_or(c.kv_pages);
        c.page_size = self.page_size.unwrap_or(c.page_size);
        c.step_tokens = self.step_tokens.unwrap_or(c.step_tokens);
        c.cpu |= self.cpu;
        c.sandbox.dir = self.allow_dir.clone().or(c.sandbox.dir);
        c.sandbox.writable |= self.allow_write;
        c.sandbox.connect.extend(self.allow_connect.iter().cloned());
        Ok(c)
    }
}

/// Start a worker as `config` says.
fn start(config: &Config) -> Result<Arc<Host>> {
    let connect = config
        .sandbox
        .connect
        .iter()
        .map(|a| std::net::ToSocketAddrs::to_socket_addrs(a).with_context(|| format!("bad address {a}")))
        .collect::<Result<Vec<_>>>()?;
    worker::start(&worker::Config {
        model: catalog::resolve(&config.model)?,
        kv_pages: config.kv_pages,
        page_size: config.page_size,
        step_tokens: config.step_tokens,
        cpu: config.cpu,
        policy: runtime::inferlet::Policy {
            dir: config.sandbox.dir.clone(),
            writable: config.sandbox.writable,
            connect: connect.into_iter().flatten().collect(),
        },
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            inferlet,
            instances,
            sequential,
            engine,
            args,
        } => run(start(&engine.config()?)?, &inferlet, instances, sequential, args).await,
        Command::Serve {
            addr,
            programs,
            metrics,
            engine,
        } => {
            let config = engine.config()?;
            let host = start(&config)?;
            if let Some(metrics) = metrics.or(config.metrics.clone()) {
                let h = host.clone();
                tokio::spawn(bootstrap::serve_metrics(metrics, move || h.metrics()));
            }
            let dir = match programs {
                Some(dir) => dir,
                None => bootstrap::home()?.join("programs"),
            };
            let programs = runtime::inferlet::Programs::open(dir)?;
            gateway::serve(host, Arc::new(programs), &addr.unwrap_or(config.addr), 1).await
        }
        Command::Gateway { addr } => gateway::route::route(&addr).await,
        Command::Worker {
            gateway,
            addr,
            programs,
            engine,
        } => {
            let host = start(&engine.config()?)?;
            let dir = match programs {
                Some(dir) => dir,
                None => bootstrap::home()?.join("programs"),
            };
            let programs = runtime::inferlet::Programs::open(dir)?;
            let id = gateway::route::register(&gateway, &addr).await?;
            eprintln!("registered with {gateway} as worker {id}");
            gateway::serve(host, Arc::new(programs), &addr, (id as u64) << 32).await
        }
        Command::Model(ModelCommand::Import { model }) => {
            let entry = catalog::import(&model)?;
            println!("imported {} ({}) as {}", entry.source, entry.model_type, entry.name);
            Ok(())
        }
        Command::Model(ModelCommand::List) => {
            for e in catalog::list()? {
                println!("{:<20} {:<10} {}", e.name, e.model_type, e.source);
            }
            Ok(())
        }
        Command::Config(ConfigCommand::Init { force }) => {
            println!("wrote {}", Config::init(force)?.display());
            Ok(())
        }
        Command::Config(ConfigCommand::Show) => {
            print!("{}", toml::to_string(&Config::load()?)?);
            Ok(())
        }
    }
}

/// Run `instances` copies of an inferlet and print what each returns.
async fn run(host: Arc<Host>, inferlet: &str, instances: usize, sequential: bool, args: Vec<String>) -> Result<()> {
    let component = host.load(inferlet)?;
    let session = terminal();

    let t = Instant::now();
    let mut running = vec![];
    let mut done = vec![];
    for _ in 0..instances {
        let (host, component, inferlet_args) = (host.clone(), component.clone(), args.clone());
        let session = session.clone();
        let run = tokio::spawn(async move {
            let started = Instant::now();
            let result = host.run(&component, inferlet_args, session).await;
            (started.elapsed(), result)
        });
        // Sequential: finish this one before starting the next.
        if sequential {
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
    eprintln!("{instances} run(s) in {:.1?}", t.elapsed());
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
