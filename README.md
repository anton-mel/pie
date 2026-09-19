# Chapter #27: Many Workers

Until chapter 26, everything ran in one process: one model, one runtime,
one server. The reference splits serving into roles: a gateway that clients
reach, a controller that knows which workers exist and where to send work,
and workers that each own a GPU and a copy of the model. On one machine it
runs them together; across machines, as separate daemons.

In chapter 27 they are separate processes. `pie worker` loads the model and
serves programs like `pie serve`, and registers with a gateway when it
starts. The registration is a connection that stays open: when it drops,
the worker is gone. Each worker numbers its processes from `worker << 32`,
so a process id says which worker holds it.

`pie gateway` loads no model (`crates/gateway/src/route.rs`). It keeps the
registry of workers, the controller's job, and routes every request:
`Install` to every worker, `Launch` to the worker with the fewest open
sessions, `Attach` and `Kill` to the worker whose range holds the id, and
`List` to all of them, merged. Once a session is placed, the gateway only
forwards its frames. Clients do not change: `pie-client` talks to a gateway
exactly as to a single server.

```
./target/release/pie gateway --addr 127.0.0.1:9123
./target/release/pie worker --gateway 127.0.0.1:9123 --addr 127.0.0.1:9124
./target/release/pie worker --gateway 127.0.0.1:9123 --addr 127.0.0.1:9125

pie-client ps
4294967296  running  decode-latency      <- worker 1
4294967297  running  decode-latency
8589934592  running  decode-latency      <- worker 2
8589934593  running  decode-latency
```

Four sessions launched through the gateway land two on each worker; a
process on either worker can be attached to or killed through the gateway;
and when a worker stops, the gateway forgets it and sends new work to the
others. The protocol is now version 3: it adds `Register` and `Registered`.

> [!NOTE]
> Here every worker is a whole copy of the model and all of them run on one
> Mac. The reference's controller runs apart from the gateway, places work
> knowing each worker's memory and load, and its workers can move KV pages
> between machines over RDMA (`crates/transport`) or split one model
> between several GPUs.

## Read Order

Read `crates/gateway/src/route.rs` from the top: `handle` routes,
`keep_registered` is the registry's liveness, `session` forwards frames, and
`register` is the worker's side. Then the new messages in
`crates/client-api`, `Processes::new` in
`crates/runtime/src/inferlet/process.rs`, and `gateway` and `worker` in
`src/main.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p client
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie gateway --addr 127.0.0.1:9123
./target/release/pie worker --gateway 127.0.0.1:9123 --addr 127.0.0.1:9124 --kv-pages 256
./target/release/pie worker --gateway 127.0.0.1:9123 --addr 127.0.0.1:9125 --kv-pages 256

./target/release/pie-client install target/wasm32-wasip2/release/text_completion.wasm tests/inferlets/text-completion/Pie.toml
./target/release/pie-client run text-completion -- "The capital of France is" 24
```
