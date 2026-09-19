//! Answer a question about a document the inferlet fetches itself.
//!
//! Example: `pie --allow-dir docs ask_docs.wasm -- /data/notes.txt "question"`,
//! or `pie --allow-connect 127.0.0.1:8000 ask_docs.wasm -- http://127.0.0.1:8000/notes.txt "question"`.
//!
//! The document is read with ordinary file and socket calls. Whether they
//! succeed is not up to this inferlet: the runtime's sandbox policy decides
//! which directory and which addresses it may reach.

use inferlet::{Context, greedy};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

/// A file under `/data`, or `http://IP:PORT/path` (an address, not a name:
/// looking names up is not granted).
fn fetch(source: &str) -> Result<String, String> {
    let Some(rest) = source.strip_prefix("http://") else {
        return std::fs::read_to_string(source).map_err(|e| format!("reading {source}: {e}"));
    };
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let addr: SocketAddr = host.parse().map_err(|e| format!("{host}: {e}"))?;
    let mut conn = TcpStream::connect(addr).map_err(|e| format!("connecting to {addr}: {e}"))?;
    write!(conn, "GET /{path} HTTP/1.0\r\nHost: {host}\r\n\r\n").map_err(|e| e.to_string())?;
    let mut response = String::new();
    conn.read_to_string(&mut response).map_err(|e| e.to_string())?;
    let (head, body) = response.split_once("\r\n\r\n").ok_or("not an HTTP response")?;
    match head.lines().next() {
        Some(status) if status.contains(" 200 ") => Ok(body.to_string()),
        status => Err(format!("{source}: {}", status.unwrap_or("no status"))),
    }
}

struct App;

impl inferlet::Guest for App {
    fn run(args: Vec<String>) -> Result<String, String> {
        let source = args.first().ok_or("give a document: /data/<file> or http://IP:PORT/<path>")?;
        let question = args.get(1).map_or("What is this document about?", |s| s);
        let document = fetch(source)?;

        let mut ctx = Context::new();
        ctx.system("Answer the question using only the document. Answer in one short sentence.");
        ctx.user(&format!("Document:\n{document}\n\nQuestion: {question} /no_think"));
        let reply = ctx.reply(256, 1, greedy)?;
        Ok(format!("{}\n    ({} bytes from {source})", reply.text, document.len()))
    }
}

inferlet::export!(App);
