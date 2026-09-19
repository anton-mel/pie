//! The gateway: where clients connect. It speaks `client-api` over
//! websockets, installs programs, and starts them by name. Each connection
//! that launches a program is that program's session: the client's messages
//! go to it, and its messages go back.
//!
//! All programs run in the one runtime, so requests from different clients
//! are batched together like local ones.

use anyhow::{Result, bail};
use client_api::{ClientMessage, ServerMessage, VERSION};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use runtime::inferlet::{Host, Programs, Session};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

type Tx = SplitSink<WebSocketStream<TcpStream>, Message>;
type Rx = SplitStream<WebSocketStream<TcpStream>>;

pub async fn serve(host: Arc<Host>, programs: Arc<Programs>, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    eprintln!("serving on ws://{addr}");
    loop {
        let (conn, peer) = listener.accept().await?;
        let (host, programs) = (host.clone(), programs.clone());
        tokio::spawn(async move {
            if let Err(e) = handle(host, programs, conn).await {
                eprintln!("{peer}: {e}");
            }
        });
    }
}

async fn handle(host: Arc<Host>, programs: Arc<Programs>, conn: TcpStream) -> Result<()> {
    let (mut tx, mut rx) = tokio_tungstenite::accept_async(conn).await?.split();
    send(&mut tx, ServerMessage::Hello { version: VERSION }).await?;

    // Install programs until one is launched.
    let (component, args) = loop {
        match next(&mut rx).await? {
            ClientMessage::Install { manifest } => {
                let Some(Ok(Message::Binary(wasm))) = rx.next().await else {
                    bail!("expected the program's wasm after install");
                };
                let package = &manifest.package;
                let reply = match programs.install(&host, &package.name, &package.version, &manifest.to_toml(), &wasm) {
                    Ok(()) => ServerMessage::Installed {
                        program: package.name.clone(),
                        version: package.version.clone(),
                    },
                    Err(e) => ServerMessage::Error { message: e.to_string() },
                };
                send(&mut tx, reply).await?;
            }
            ClientMessage::Launch { program, args } => match programs.get(&host, &program) {
                Ok(component) => break (component, args),
                Err(e) => return send(&mut tx, ServerMessage::Error { message: e.to_string() }).await,
            },
            ClientMessage::Message { .. } | ClientMessage::Close => bail!("no program launched"),
        }
    };

    // The client's messages are the program's inbox, until it closes.
    let (to_inbox, inbox) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok(ClientMessage::Message { text }) = next(&mut rx).await {
            if to_inbox.send(text).is_err() {
                break;
            }
        }
    });

    let (out, mut outbox) = mpsc::unbounded_channel();
    let session = Session {
        out,
        inbox: Arc::new(Mutex::new(inbox)),
    };
    // Relay the program's messages while it runs. If the client goes away,
    // sending fails and the program is dropped with its pages.
    let run = host.run(&component, args, session);
    tokio::pin!(run);
    let result = loop {
        tokio::select! {
            Some(text) = outbox.recv() => send(&mut tx, ServerMessage::Message { text }).await?,
            result = &mut run => break result,
        }
    };
    while let Ok(text) = outbox.try_recv() {
        send(&mut tx, ServerMessage::Message { text }).await?;
    }
    let end = match result {
        Ok(Ok(value)) => ServerMessage::Result { value },
        Ok(Err(message)) => ServerMessage::Error { message },
        Err(e) => ServerMessage::Error { message: e.to_string() },
    };
    send(&mut tx, end).await
}

/// The next message from the client; an error once it is gone.
async fn next(rx: &mut Rx) -> Result<ClientMessage> {
    loop {
        match rx.next().await {
            Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            _ => bail!("client closed the connection"),
        }
    }
}

async fn send(tx: &mut Tx, message: ServerMessage) -> Result<()> {
    tx.send(Message::text(serde_json::to_string(&message)?)).await?;
    Ok(())
}
