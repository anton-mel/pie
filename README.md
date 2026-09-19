# Chapter #12: Server and Client

Until chapter 11, `pie` ran one inferlet file and printed what it returned.
A serving system is long-running: clients send it programs over the
network, and talk to them while they run.

In chapter 12 `pie --serve ADDR` does that (`runtime/src/server.rs`).
`pie-client` (`client/`) sends it an inferlet with its arguments, and after
that every line typed on stdin is a message for the inferlet. The inferlet
talks back through `session` (`wit/pie.wit`): `send` a message, `receive` the
next one. Every client's inferlet runs in the same engine, so their forwards
are batched together (chapter 4), even when they run different programs.

`examples/chat-session` is an interactive chat: each message is one user
turn, and the answer is sent back while it is being generated, a few
characters per message (`Context::reply_streaming`). Run locally with
`pie`, the same inferlet reads messages from stdin and prints what it sends.

> [!NOTE]
> The protocol is one TCP connection per inferlet, one JSON value per line.
> The current Pie has a gateway that clients reach over websockets, with
> authentication and file transfer, in front of workers that can be on
> other machines.

## Read Order

Read `session` in `wit/pie.wit`, then `Session` and `session::Host` in
`runtime/src/host.rs`. Then `runtime/src/server.rs` from the top, and
`client/src/main.rs`. Finally `Context::reply_streaming` in
`inferlet/src/lib.rs` and `examples/chat-session`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p pie-client
cargo build --release -p chat-session -p beam-search --target wasm32-wasip2

./target/release/pie --serve 127.0.0.1:9123

# in other terminals, at the same time
./target/release/pie-client target/wasm32-wasip2/release/chat_session.wasm
./target/release/pie-client target/wasm32-wasip2/release/beam_search.wasm -- "The capital of France is" 4 16
```
