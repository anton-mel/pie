//! `pie-client`: install programs on a running `pie --serve`, run them by
//! name as processes, and attach to, list or kill processes.
//!
//! While attached, each line typed on stdin is a message for the process,
//! and its messages are printed as they arrive, then its result. Leaving
//! (Ctrl-C) detaches; the process keeps running. The messages themselves are
//! in `client-api`.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use client_api::{ClientMessage, Manifest, ServerMessage, VERSION};
use futures_util::{SinkExt, StreamExt};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tokio_tungstenite::tungstenite::Message;

#[derive(Parser)]
struct Args {
    /// Address of `pie --serve`.
    #[arg(long, default_value = "127.0.0.1:9123")]
    server: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Install a program, so it can be run by the name in its manifest.
    Install {
        /// The inferlet (.wasm component).
        wasm: PathBuf,
        /// Its manifest (Pie.toml).
        manifest: PathBuf,
    },
    /// Run an installed program as a new process, attached to it.
    Run {
        program: String,
        /// Arguments passed to the program.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Attach to a process: see what it sent meanwhile, and talk to it.
    Attach { process: u64 },
    /// List the processes.
    Ps,
    /// Stop a process.
    Kill { process: u64 },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let url = format!("ws://{}", args.server);
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("connecting to {url}"))?;
    let (mut tx, mut rx) = ws.split();
    let mut next = async || -> Result<ServerMessage> {
        loop {
            match rx.next().await {
                Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                _ => bail!("the server closed the connection"),
            }
        }
    };
    match next().await? {
        ServerMessage::Hello { version } if version == VERSION => {}
        other => bail!("expected protocol version {VERSION}, got {other:?}"),
    }
    let send = |m: &ClientMessage| Message::text(serde_json::to_string(m).unwrap());

    match args.command {
        Command::Install { wasm, manifest } => {
            let text = std::fs::read_to_string(&manifest).with_context(|| format!("reading {}", manifest.display()))?;
            let manifest = Manifest::parse(&text)?;
            let wasm = std::fs::read(&wasm).with_context(|| format!("reading {}", wasm.display()))?;
            tx.send(send(&ClientMessage::Install { manifest })).await?;
            tx.send(Message::binary(wasm)).await?;
            match next().await? {
                ServerMessage::Installed { program, version } => println!("installed {program} {version}"),
                ServerMessage::Error { message } => bail!("{message}"),
                other => bail!("unexpected {other:?}"),
            }
        }
        Command::Ps => {
            tx.send(send(&ClientMessage::List)).await?;
            let ServerMessage::Processes { processes } = next().await? else {
                bail!("unexpected reply");
            };
            for p in processes {
                let state = if p.running { "running" } else { "ended" };
                println!("{:>5}  {:<8} {}", p.process, state, p.program);
            }
        }
        Command::Kill { process } => {
            tx.send(send(&ClientMessage::Kill { process })).await?;
            match next().await? {
                ServerMessage::Killed { process } => println!("killed {process}"),
                ServerMessage::Error { message } => bail!("{message}"),
                other => bail!("unexpected {other:?}"),
            }
        }
        Command::Run { .. } | Command::Attach { .. } => {
            let request = match args.command {
                Command::Run { program, args } => ClientMessage::Launch { program, args },
                Command::Attach { process } => ClientMessage::Attach { process },
                _ => unreachable!(),
            };
            tx.send(send(&request)).await?;
            // Forward stdin, one message per line, then say there are no more.
            let (lines, mut to_send) = tokio::sync::mpsc::unbounded_channel();
            std::thread::spawn(move || {
                for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                    let _ = lines.send(ClientMessage::Message { text: line });
                }
                let _ = lines.send(ClientMessage::Close);
            });
            tokio::spawn(async move {
                while let Some(message) = to_send.recv().await {
                    if tx.send(send(&message)).await.is_err() {
                        break;
                    }
                }
            });
            loop {
                match next().await? {
                    ServerMessage::Launched { process } => eprintln!("process {process}"),
                    ServerMessage::Message { text } => {
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    ServerMessage::Result { value } => {
                        println!("{value}");
                        std::process::exit(0);
                    }
                    ServerMessage::Error { message } => bail!("{message}"),
                    other => bail!("unexpected {other:?}"),
                }
            }
        }
    }
    Ok(())
}
