# Chapter #23: Config, Model Catalog and Metrics

Until chapter 22, `pie` was configured by flags alone, loaded whatever model
id it was given and found out only while loading whether it could run it,
and reported what it was doing with `eprintln!`.

In chapter 23 it gets the operational side of the reference:

- **Subcommands.** `pie run`, `pie serve`, `pie model`, `pie config`, as in
  the reference (`src/main.rs`).
- **A config file.** `pie config init` writes the defaults to
  `~/.pie-tutorial/config.toml` (`crates/bootstrap`): the model, the KV pool,
  the step budget, the server address, the sandbox. Flags override it for
  one command, and `pie config show` prints what is in effect.
- **A model catalog.** `pie model import Qwen/Qwen3-0.6B` fetches a model
  once and records it under a short name in `~/.pie-tutorial/models.toml`
  (`src/catalog.rs`); `pie model list` shows them, and `--model Qwen3-0.6B`
  takes the name. A model is checked from its config before any weights are
  fetched: the engine must support its family (`models::supports`) and
  there must be a chat template for it. Anything else is refused by name:
  `HuggingFaceTB/SmolLM2-135M-Instruct is a "llama" model, which this engine
  cannot run`, after fetching 8 KB.
- **Metrics.** `pie serve --metrics ADDR` serves `/metrics` in Prometheus
  text format (`crates/runtime/src/telemetry.rs`): steps, tokens, forwards,
  evictions, prefix pages reused, free and recorded pages.

```
pie_steps_total 40
pie_tokens_total 397
pie_forwards_total 60
pie_evictions_total 0
pie_prefix_pages_reused_total 18
pie_kv_pages_free 1002
```

Everything lives in `~/.pie-tutorial`, so it does not touch an installation
of the reference Pie in `~/.pie`.

## Read Order

Read `crates/bootstrap/src/lib.rs`, then `src/catalog.rs` and the
subcommands in `src/main.rs`. Then `model_type` in
`crates/worker/src/weights.rs` and the check in `worker::start`. Finally
`crates/runtime/src/telemetry.rs`, where the scheduler and planner count,
and `render_metrics` in `crates/runtime/src/engine.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p client
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie config init
./target/release/pie model import Qwen/Qwen3-0.6B
./target/release/pie model list
./target/release/pie run --model Qwen3-0.6B target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24

./target/release/pie serve --metrics 127.0.0.1:9124
curl http://127.0.0.1:9124/metrics
```
