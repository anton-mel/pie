//! The gateway in front of several workers: the controller's registry and
//! the routing, in one process, as the reference runs them on one machine.
//!
//! A worker registers with `Register` and stays connected; when that
//! connection drops, the worker is gone. Each client connection is routed:
//! `Install` goes to every worker, `Launch` to the worker with the fewest
//! sessions, `Attach` and `Kill` to the worker whose range holds the
//! process id, and `List` merges every worker's list. Once a session is
//! placed, its frames are forwarded both ways as they are.

use anyhow::{Context, Result, bail};
use client_api::{ClientMessage, ServerMessage, VERSION};
use futures_util::{SinkExt, StreamExt};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Client = WebSocketStream<TcpStream>;
type Upstream = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct Worker {
    addr: String,
    /// Sessions routed to it and still open.
    sessions: usize,
}

#[derive(Default)]
struct Registry {
    workers: BTreeMap<u32, Worker>,
    next: u32,
}

pub async fn route(addr: &str) -> Result<()> {
    let registry = Arc::new(Mutex::new(Registry {
        next: 1,
        ..Default::default()
    }));
    let listener = TcpListener::bind(addr).await?;
    eprintln!("gateway on ws://{addr}");
    loop {
        let (conn, peer) = listener.accept().await?;
        let registry = registry.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(registry, conn).await {
                eprintln!("{peer}: {e}");
            }
        });
    }
}

async fn handle(registry: Arc<Mutex<Registry>>, conn: TcpStream) -> Result<()> {
    let mut client = tokio_tungstenite::accept_async(conn).await?;
    send(&mut client, &ServerMessage::Hello { version: VERSION }).await?;
    loop {
        let Some(request) = next(&mut client).await else {
            return Ok(());
        };
        let reply = match request {
            ClientMessage::Register { addr } => return keep_registered(registry, client, addr).await,
            ClientMessage::Install { manifest } => {
                let Some(Ok(Message::Binary(wasm))) = client.next().await else {
                    bail!("expected the program's wasm after install");
                };
                let mut reply = None;
                for addr in addrs(&registry) {
                    let mut worker = connect(&addr).await?;
                    send(
                        &mut worker,
                        &ClientMessage::Install {
                            manifest: manifest.clone(),
                        },
                    )
                    .await?;
                    worker.send(Message::Binary(wasm.clone())).await?;
                    match reply_of(&mut worker).await? {
                        ok @ ServerMessage::Installed { .. } => reply = reply.or(Some(ok)),
                        error => {
                            reply = Some(error);
                            break;
                        }
                    }
                }
                reply.unwrap_or(ServerMessage::Error {
                    message: "no workers".into(),
                })
            }
            ClientMessage::List => {
                let mut processes = vec![];
                for addr in addrs(&registry) {
                    let mut worker = connect(&addr).await?;
                    send(&mut worker, &ClientMessage::List).await?;
                    if let ServerMessage::Processes { processes: p } = reply_of(&mut worker).await? {
                        processes.extend(p);
                    }
                }
                ServerMessage::Processes { processes }
            }
            ClientMessage::Kill { process } => match addr_of(&registry, process) {
                Some(addr) => {
                    let mut worker = connect(&addr).await?;
                    send(&mut worker, &ClientMessage::Kill { process }).await?;
                    reply_of(&mut worker).await?
                }
                None => ServerMessage::Error {
                    message: format!("no process {process}"),
                },
            },
            ClientMessage::Launch { program, args } => {
                let Some((id, addr)) = least_busy(&registry) else {
                    send(
                        &mut client,
                        &ServerMessage::Error {
                            message: "no workers".into(),
                        },
                    )
                    .await?;
                    continue;
                };
                let result = session(client, &addr, ClientMessage::Launch { program, args }).await;
                if let Some(w) = registry.lock().unwrap().workers.get_mut(&id) {
                    w.sessions -= 1;
                }
                return result;
            }
            ClientMessage::Attach { process } => match addr_of(&registry, process) {
                Some(addr) => return session(client, &addr, ClientMessage::Attach { process }).await,
                None => ServerMessage::Error {
                    message: format!("no process {process}"),
                },
            },
            ClientMessage::Message { .. } | ClientMessage::Close => ServerMessage::Error {
                message: "not attached to a process".into(),
            },
        };
        send(&mut client, &reply).await?;
    }
}

/// A worker has registered: give it an id, and forget it when it hangs up.
async fn keep_registered(registry: Arc<Mutex<Registry>>, mut conn: Client, addr: String) -> Result<()> {
    let id = {
        let mut r = registry.lock().unwrap();
        let id = r.next;
        r.next += 1;
        r.workers.insert(
            id,
            Worker {
                addr: addr.clone(),
                sessions: 0,
            },
        );
        id
    };
    eprintln!("worker {id} registered at {addr}");
    send(&mut conn, &ServerMessage::Registered { worker: id }).await?;
    while let Some(Ok(_)) = conn.next().await {}
    registry.lock().unwrap().workers.remove(&id);
    eprintln!("worker {id} left");
    Ok(())
}

/// Place a session on a worker, then forward frames both ways until either
/// side hangs up.
async fn session(client: Client, addr: &str, first: ClientMessage) -> Result<()> {
    let mut worker = connect(addr).await?;
    send(&mut worker, &first).await?;
    let (mut client_tx, mut client_rx) = client.split();
    let (mut worker_tx, mut worker_rx) = worker.split();
    let up = async {
        while let Some(Ok(frame)) = client_rx.next().await {
            if worker_tx.send(frame).await.is_err() {
                break;
            }
        }
    };
    let down = async {
        while let Some(Ok(frame)) = worker_rx.next().await {
            if client_tx.send(frame).await.is_err() {
                break;
            }
        }
    };
    tokio::select! {
        _ = up => {}
        _ = down => {}
    }
    Ok(())
}

fn addrs(registry: &Mutex<Registry>) -> Vec<String> {
    registry
        .lock()
        .unwrap()
        .workers
        .values()
        .map(|w| w.addr.clone())
        .collect()
}

/// The worker whose id range holds `process`.
fn addr_of(registry: &Mutex<Registry>, process: u64) -> Option<String> {
    let id = (process >> 32) as u32;
    registry.lock().unwrap().workers.get(&id).map(|w| w.addr.clone())
}

/// The worker with the fewest open sessions, now with one more.
fn least_busy(registry: &Mutex<Registry>) -> Option<(u32, String)> {
    let mut r = registry.lock().unwrap();
    let (&id, worker) = r.workers.iter_mut().min_by_key(|(_, w)| w.sessions)?;
    worker.sessions += 1;
    Some((id, worker.addr.clone()))
}

/// Open a connection to a worker and read its greeting.
async fn connect(addr: &str) -> Result<Upstream> {
    let (mut worker, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
        .await
        .with_context(|| format!("connecting to worker at {addr}"))?;
    match reply_of(&mut worker).await? {
        ServerMessage::Hello { version } if version == VERSION => Ok(worker),
        other => bail!("worker at {addr} said {other:?}"),
    }
}

async fn reply_of<S>(ws: &mut WebSocketStream<S>) -> Result<ServerMessage>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    loop {
        match ws.next().await {
            Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            _ => bail!("the worker closed the connection"),
        }
    }
}

async fn next(client: &mut Client) -> Option<ClientMessage> {
    loop {
        match client.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).ok(),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            _ => return None,
        }
    }
}

async fn send<S, M>(ws: &mut WebSocketStream<S>, message: &M) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    M: serde::Serialize,
{
    ws.send(Message::text(serde_json::to_string(message)?)).await?;
    Ok(())
}

/// From a worker: register with the gateway at `gateway` as serving at
/// `addr`, and stay registered (the connection is kept open in the
/// background). Returns the worker's id.
pub async fn register(gateway: &str, addr: &str) -> Result<u32> {
    let mut conn = connect(gateway).await?;
    send(&mut conn, &ClientMessage::Register { addr: addr.to_string() }).await?;
    let ServerMessage::Registered { worker } = reply_of(&mut conn).await? else {
        bail!("the gateway did not register this worker");
    };
    tokio::spawn(async move { while let Some(Ok(_)) = conn.next().await {} });
    Ok(worker)
}
