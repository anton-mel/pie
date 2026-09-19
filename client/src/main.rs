//! `pie-client`: send an inferlet to a running `pie --serve` and talk to it.
//!
//! Each line typed on stdin is a message for the inferlet; its messages are
//! printed as they arrive, then its result. The protocol is in
//! `runtime/src/server.rs`.

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpStream};

#[derive(Parser)]
struct Args {
    /// Path to the inferlet (.wasm component).
    inferlet: String,
    /// Address of `pie --serve`.
    #[arg(long, default_value = "127.0.0.1:9123")]
    server: String,
    /// Arguments passed to the inferlet.
    #[arg(last = true)]
    args: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let wasm = std::fs::read(&args.inferlet).with_context(|| format!("reading {}", args.inferlet))?;
    let mut conn = TcpStream::connect(&args.server).with_context(|| format!("connecting to {}", args.server))?;
    writeln!(conn, "{}", json!({ "args": args.args, "wasm": wasm.len() }))?;
    conn.write_all(&wasm)?;

    // Forward stdin, one message per line; closing stdin ends the messages.
    let mut to_server = conn.try_clone()?;
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if writeln!(to_server, "{}", Value::String(line)).is_err() {
                return;
            }
        }
        let _ = to_server.shutdown(Shutdown::Write);
    });

    for line in BufReader::new(conn).lines() {
        let value: Value = serde_json::from_str(&line?)?;
        if let Some(message) = value["message"].as_str() {
            print!("{message}");
            std::io::stdout().flush()?;
        } else if let Some(result) = value["result"].as_str() {
            println!("{result}");
            std::process::exit(0);
        } else if let Some(error) = value["error"].as_str() {
            bail!("{error}");
        }
    }
    bail!("the server closed the connection")
}
