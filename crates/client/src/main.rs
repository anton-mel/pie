//! `pie-client`: install programs on a running `pie --serve`, and run them
//! by name.
//!
//! While a program runs, each line typed on stdin is a message for it, and
//! its messages are printed as they arrive, then its result. The messages
//! themselves are in `client-api`.

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
    /// Run an installed program.
    Run {
        program: String,
        /// Arguments passed to the program.
        #[arg(last = true)]
        args: Vec<String>,
    },
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
        Command::Run { program, args } => {
            tx.send(send(&ClientMessage::Launch { program, args })).await?;
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
