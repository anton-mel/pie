# Chapter #17: Protocol and Programs

Until chapter 16, `pie --serve` was a TCP server inside the runtime, with a
line-based JSON protocol written out by hand on both sides, and every client
uploaded its whole inferlet on every connection.

In chapter 17 clients talk to a gateway, over a protocol both sides share,
and start programs by name, as in the reference.

`crates/client-api` is that protocol: the only public interface, and it has
a version. A client sends `Install`, `Launch`, `Message` or `Close`; the
server answers `Hello`, `Installed`, `Message`, `Result` or `Error`. They are
typed Rust enums, sent as JSON over websockets, and the gateway and the
client both use the same crate, so they cannot drift apart.

Every inferlet in `tests/inferlets` now has a manifest, `Pie.toml`, with its
name, version and description. `pie-client install` sends a program once;
the runtime checks that it compiles and keeps it as
`<name>/<version>.wasm` in a program directory
(`crates/runtime/src/inferlet/program.rs`, by default
`~/.pie-tutorial/programs`), where it stays across restarts.
`pie-client run <name>` then starts the latest version by name.

`crates/gateway` is the server, moved out of the runtime: it greets each
client with the protocol version, installs programs, launches them, and
relays the session's messages both ways.

```
pie-client ──websocket──▶ gateway ──▶ runtime (programs, inferlets, …)
          (client-api)
```

> [!NOTE]
> The reference's gateway also decides whether to admit each request,
> moves files, and routes sessions to one of several workers, and programs
> can be fetched from a registry and declare typed parameters. Here,
> arguments are still a list of strings.

## Read Order

Read `crates/client-api/src/lib.rs`, then `crates/gateway/src/lib.rs`, then
`Programs` in `crates/runtime/src/inferlet/program.rs`. Finally
`crates/client/src/main.rs` and a `Pie.toml` in `tests/inferlets`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p client
cargo build --release -p text-completion -p chat-session --target wasm32-wasip2

./target/release/pie --serve 127.0.0.1:9123

# in another terminal: install once, then run by name
./target/release/pie-client install target/wasm32-wasip2/release/chat_session.wasm tests/inferlets/chat-session/Pie.toml
./target/release/pie-client run chat-session
```
