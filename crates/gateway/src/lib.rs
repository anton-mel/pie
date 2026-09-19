//! The gateway: where clients connect. It speaks `client-api` over
//! websockets, installs programs, starts them as processes, and attaches
//! connections to processes: the client's messages go to the process, and
//! its messages come back. A process outlives the connection that started it.
//!
//! All programs run in the one runtime, so requests from different clients
//! are batched together like local ones.

use anyhow::{Result, bail};
use client_api::{ClientMessage, ProcessInfo, ServerMessage, VERSION};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use runtime::inferlet::{Event, Host, Processes, Programs};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

type Tx = SplitSink<WebSocketStream<TcpStream>, Message>;
type Rx = SplitStream<WebSocketStream<TcpStream>>;

pub async fn serve(host: Arc<Host>, programs: Arc<Programs>, addr: &str) -> Result<()> {
    let processes = Arc::new(Processes::new(host.clone()));
    let listener = TcpListener::bind(addr).await?;
    eprintln!("serving on ws://{addr}");
    loop {
        let (conn, peer) = listener.accept().await?;
        let (host, programs, processes) = (host.clone(), programs.clone(), processes.clone());
        tokio::spawn(async move {
            if let Err(e) = handle(host, programs, processes, conn).await {
                eprintln!("{peer}: {e}");
            }
        });
    }
}

async fn handle(host: Arc<Host>, programs: Arc<Programs>, processes: Arc<Processes>, conn: TcpStream) -> Result<()> {
    let (mut tx, mut rx) = tokio_tungstenite::accept_async(conn).await?.split();
    send(&mut tx, ServerMessage::Hello { version: VERSION }).await?;

    // Answer requests until the client launches or attaches to a process.
    let process = loop {
        let Some(request) = next(&mut rx).await else {
            return Ok(());
        };
        let reply = match request {
            ClientMessage::Install { manifest } => {
                let Some(Ok(Message::Binary(wasm))) = rx.next().await else {
                    bail!("expected the program's wasm after install");
                };
                let package = &manifest.package;
                match programs.install(&host, &package.name, &package.version, &manifest.to_toml(), &wasm) {
                    Ok(()) => ServerMessage::Installed {
                        program: package.name.clone(),
                        version: package.version.clone(),
                    },
                    Err(e) => ServerMessage::Error { message: e.to_string() },
                }
            }
            ClientMessage::Launch { program, args } => match programs.get(&host, &program) {
                Ok(component) => {
                    let process = processes.spawn(component, &program, args);
                    send(&mut tx, ServerMessage::Launched { process: process.id }).await?;
                    break process;
                }
                Err(e) => ServerMessage::Error { message: e.to_string() },
            },
            ClientMessage::Attach { process } => match processes.get(process) {
                Some(process) => break process,
                None => ServerMessage::Error {
                    message: format!("no process {process}"),
                },
            },
            ClientMessage::List => ServerMessage::Processes {
                processes: processes
                    .list()
                    .into_iter()
                    .map(|p| ProcessInfo {
                        process: p.id,
                        program: p.program,
                        running: p.running,
                    })
                    .collect(),
            },
            ClientMessage::Kill { process } => match processes.kill(process) {
                true => ServerMessage::Killed { process },
                false => ServerMessage::Error {
                    message: format!("no process {process}"),
                },
            },
            ClientMessage::Message { .. } | ClientMessage::Close => ServerMessage::Error {
                message: "not attached to a process".into(),
            },
        };
        send(&mut tx, reply).await?;
    };

    // Attached: the client's messages go to the process, until it leaves...
    let p = process.clone();
    let mut input = tokio::spawn(async move {
        while let Some(message) = next(&mut rx).await {
            match message {
                ClientMessage::Message { text } => {
                    p.send(text);
                }
                ClientMessage::Close => p.close_input(),
                _ => break,
            }
        }
    });
    // ...and its events come back. If the client leaves, the process is only
    // detached, and what was on its way to the client is kept.
    let (attachment, mut events) = process.attach();
    loop {
        tokio::select! {
            Some(event) = events.recv() => {
                let (message, ended) = match &event {
                    Event::Message(text) => (ServerMessage::Message { text: text.clone() }, false),
                    Event::Exited(Ok(value)) => (ServerMessage::Result { value: value.clone() }, true),
                    Event::Exited(Err(message)) => (ServerMessage::Error { message: message.clone() }, true),
                };
                if send(&mut tx, message).await.is_err() {
                    let mut undelivered = vec![event];
                    while let Ok(e) = events.try_recv() {
                        undelivered.push(e);
                    }
                    process.detach(attachment, undelivered);
                    break;
                }
                if ended {
                    processes.reap(process.id);
                    break;
                }
            }
            _ = &mut input => {
                let mut undelivered = vec![];
                while let Ok(e) = events.try_recv() {
                    undelivered.push(e);
                }
                process.detach(attachment, undelivered);
                break;
            }
        }
    }
    input.abort();
    Ok(())
}

/// The next message from the client; none once it has left.
async fn next(rx: &mut Rx) -> Option<ClientMessage> {
    loop {
        match rx.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).ok(),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            _ => return None,
        }
    }
}

async fn send(tx: &mut Tx, message: ServerMessage) -> Result<()> {
    tx.send(Message::text(serde_json::to_string(&message)?)).await?;
    Ok(())
}
