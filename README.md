# Chapter #16: Engine and Worker

Until chapter 15, the runtime called the model directly: the scheduler held
a candle `Model` and ran `Model::forward`, and `src/main.rs` found the
checkpoint, loaded the weights and wired everything together. The runtime
knew exactly which model it had, and on which device.

In chapter 16 two crates separate those concerns, as in the reference.

`crates/engine` is the contract between the runtime and whatever runs the
model: a `Seq` (one sequence's share of a step) and a trait `Engine` with
two methods, `page_size` and `forward`, which returns one row of logits per
requested output. The runtime now depends on this crate only. The scheduler
holds a `Box<dyn Engine>`, and `models` implements the trait for the Qwen
model. A second backend would be a second implementation, with no change to
the runtime.

`crates/worker` is the role that owns the GPU. `worker::start(config)`
finds the model's files (`weights.rs`), loads the model, opens it as an
engine, and builds the runtime on top. `src/main.rs` only parses arguments,
asks the worker for a running host, and runs or serves inferlets.

```
src/main.rs ──▶ worker ──▶ models (impl Engine)
                  │
                  └──▶ runtime ──▶ engine (the trait)
```

Nothing changes for inferlets, and every example gives the same output as
in chapter 15, at the same speed.

## Read Order

Read `crates/engine/src/lib.rs`, then `impl engine::Engine for Model` at
the end of `crates/models/src/qwen.rs`. Then `crates/worker/src/lib.rs` and
`weights.rs`. Finally `Engine::new` in `crates/runtime/src/engine.rs`,
`scheduler::run`, and the shorter `src/main.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
```
