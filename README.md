# Chapter #13: Scheduler

Until chapter 12, every model step ran everything queued at once. One long
prompt then held up everybody: while a client's 4,000-token prompt ran, an
inferlet that was decoding waited 2 seconds for its next token instead of
6 milliseconds.

In chapter 13 the scheduler (`runtime/src/scheduler.rs`) decides what goes
into each step, the way vLLM does. A step runs at most `--step-tokens`
tokens (256 by default). Requests with the fewest tokens left go first: most
of them are one inferlet waiting for its next token. A long prompt fills
what is left and is split across steps, and its inferlet gets the result
once the last piece has run. Nothing changes for inferlets: they submit
forwards as before, and results are exactly the same.

| `--step-tokens` | longest wait of a decoding inferlet | 4,000-token prompt takes |
|---|---|---|
| unlimited (before) | 2041 ms | 2.05 s |
| 1024 | 1071 ms | 1.81 s |
| 256 | 312 ms | 1.75 s |
| 64 | 170 ms | 3.31 s |

> [!WARNING]
> Two forwards on the same working set must run in order, because the second
> reads what the first writes. Once short requests may go first, a later
> decode could overtake an earlier prompt. So a request waits while an
> earlier one still has to write pages it reads (`writes` in
> `runtime/src/scheduler.rs`). Forked branches write their own copied pages
> (chapter 3), so they still run together.

## Read Order

Read `runtime/src/scheduler.rs` from the top: `Job::chunk` cuts a request
into a smaller one, and `run` picks the jobs for each step. Then `Request`
and `Engine::new` in `runtime/src/engine.rs`. Finally
`examples/decode-latency`, which measures the waits.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p pie-client
cargo build --release -p decode-latency -p text-completion --target wasm32-wasip2

./target/release/pie --serve 127.0.0.1:9123 --step-tokens 256

# in other terminals: a decoding inferlet, then a long prompt while it runs
./target/release/pie-client target/wasm32-wasip2/release/decode_latency.wasm -- 300
./target/release/pie-client target/wasm32-wasip2/release/text_completion.wasm -- "$(cat long-prompt.txt)" 1
```
