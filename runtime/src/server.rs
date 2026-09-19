//! `pie --serve`: run inferlets that clients send over the network.
//!
//! One TCP connection is one inferlet. Every line is one JSON value:
//!
//! ```text
//! client -> server   {"args": [...], "wasm": N}, then N bytes of wasm,
//!                    then one JSON string per message for the inferlet
//! server -> client   {"message": "..."} for every `session.send`,
//!                    then {"result": "..."} or {"error": "..."}
//! ```
//!
//! All inferlets share the one engine, so requests from different clients
//! are batched together like local ones.

use crate::host::{Host, Session};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream, tcp::OwnedWriteHalf};
use tokio::sync::{Mutex, mpsc};

#[derive(Deserialize)]
struct Header {
    args: Vec<String>,
    /// Size of the wasm that follows, in bytes.
    wasm: usize,
}

pub async fn serve(host: Arc<Host>, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    eprintln!("serving on {addr}");
    loop {
        let (conn, peer) = listener.accept().await?;
        let host = host.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(host, conn).await {
                eprintln!("{peer}: {e}");
            }
        });
    }
}

async fn handle(host: Arc<Host>, conn: TcpStream) -> Result<()> {
    let (read, mut write) = conn.into_split();
    let mut read = BufReader::new(read);
    let mut line = String::new();
    read.read_line(&mut line).await?;
    let header: Header = serde_json::from_str(&line)?;
    let mut wasm = vec![0; header.wasm];
    read.read_exact(&mut wasm).await?;
    let component = match host.compile(&wasm) {
        Ok(c) => c,
        Err(e) => return send(&mut write, json!({ "error": e.to_string() })).await,
    };

    // Every further line from the client is a message for the inferlet.
    let (to_inbox, inbox) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut lines = read.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(message) = serde_json::from_str::<String>(&line)
                && to_inbox.send(message).is_err()
            {
                break;
            }
        }
    });

    let (out, mut outbox) = mpsc::unbounded_channel();
    let session = Session {
        out,
        inbox: Arc::new(Mutex::new(inbox)),
    };
    // Relay the inferlet's messages while it runs. If the client goes away,
    // sending fails and the inferlet is dropped with its pages.
    let run = host.run(&component, header.args, session);
    tokio::pin!(run);
    let result = loop {
        tokio::select! {
            Some(message) = outbox.recv() => send(&mut write, json!({ "message": message })).await?,
            result = &mut run => break result,
        }
    };
    while let Ok(message) = outbox.try_recv() {
        send(&mut write, json!({ "message": message })).await?;
    }
    let end = match result {
        Ok(Ok(r)) => json!({ "result": r }),
        Ok(Err(e)) => json!({ "error": e }),
        Err(e) => json!({ "error": e.to_string() }),
    };
    send(&mut write, end).await
}

async fn send(write: &mut OwnedWriteHalf, value: Value) -> Result<()> {
    write.write_all(format!("{value}\n").as_bytes()).await?;
    Ok(())
}
