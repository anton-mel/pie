# Chapter #18: Processes

Until chapter 17, a program lived exactly as long as the connection that
launched it: if the client went away, the program was dropped. Nothing
could be long-running, and nobody else could look at a program while it ran.

In chapter 18 a launched program is a process with an id
(`crates/runtime/src/inferlet/process.rs`), and it outlives the client that
started it. What it sends while nobody listens is kept. Any client can
attach to it later: it first gets what it missed, then what comes next, and
can send it messages. A client leaving only detaches. How a process ends,
whether it returned, failed or was killed, is recorded in one place, and
killing one drops its instance and every page it held.

The protocol is now version 2 (`crates/client-api`): `Launch` answers with
the process id, and there are `Attach`, `List` and `Kill`. `pie-client` has
matching commands:

```
pie-client run decode-latency -- 400    # prints "process 1", then Ctrl-C
pie-client ps                           #     1  running  decode-latency
pie-client attach 1                     # the result it produced meanwhile
pie-client kill 1
```

A chat can outlive its client too: one client starts `chat-session` and
leaves, another attaches later and continues the same conversation.

> [!NOTE]
> Processes live in memory: they are lost when the server stops.

## Read Order

Read `crates/runtime/src/inferlet/process.rs` from the top: `spawn`, `emit`
(where every event and the end of a process go), `attach` and `detach`.
Then the new messages in `crates/client-api/src/lib.rs`, `handle` in
`crates/gateway/src/lib.rs`, and the new commands in
`crates/client/src/main.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p client
cargo build --release -p chat-session -p decode-latency --target wasm32-wasip2

./target/release/pie --serve 127.0.0.1:9123

# in another terminal
./target/release/pie-client install target/wasm32-wasip2/release/decode_latency.wasm tests/inferlets/decode-latency/Pie.toml
./target/release/pie-client run decode-latency -- 400    # then Ctrl-C
./target/release/pie-client ps
./target/release/pie-client attach 1
```
