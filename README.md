# Chapter #15: Reference Layout

Until chapter 14, the code was laid out for reading one chapter at a time:
one WIT file, one runtime crate, examples on the side. The real Pie is
organised differently, to hold far more code.

In chapter 15 nothing new is added. The same code is moved into the layout of
[pie-project/pie](https://github.com/pie-project/pie), with the same names,
so that after this tutorial the real repository is familiar. Every example
gives exactly the same output as in chapter 14.

```
Cargo.toml                 the `pie` binary and the workspace
src/main.rs                the command line: load the model, run or serve
crates/
  inferlet/                the SDK every inferlet links against
    wit/                   the WIT contract, one interface per file
  runtime/                 the runtime library
    src/engine.rs          KV page pool, prefix index, forward queue
    src/scheduler.rs       what goes into each model step
    src/planner.rs         who gets pages when they run out
    src/server.rs          `pie --serve`
    src/inferlet.rs        runs inferlets: wasm host, state, sessions
    src/inferlet/host/     the host side of each WIT interface
  models/                  the model: Qwen2/Qwen3 with a paged KV cache
  client/                  `pie-client`
tests/inferlets/           the example inferlets
```

| before | now |
|---|---|
| `wit/pie.wit`, package `pie:core` | `crates/inferlet/wit/*.wit`, package `pie:inferlet`, as in the reference: `model`, `tokenizer`, `working-set`, `forward`, `chat`, `session`, `run` |
| `runtime/src/main.rs` | `src/main.rs` |
| `runtime/src/host.rs` | `crates/runtime/src/inferlet.rs`, and one file per interface in `crates/runtime/src/inferlet/host/` |
| `runtime/src/model.rs` | `crates/models/src/qwen.rs` |
| `inferlet/`, `client/` | `crates/inferlet/`, `crates/client/` |
| `examples/` | `tests/inferlets/` |

With the WIT split, inferlets import from the interface a function belongs
to: `tokenizer::detokenize` instead of `model::detokenize`.

## What the Reference Adds

The reference is about 500,000 lines. These are the parts this tutorial left
out, and where they are:

- **Its own GPU kernels and engines** for CUDA, Metal, Vulkan and WebGPU
  (`crates/kernels-*`, `crates/engine-*`), which read KV pages where they
  are. Here, candle does the math.
- **A model compiler** (`crates/model-ir`, `model-dsl`, `model-compiler`,
  `models`, `checkpoint`): many model families described once and compiled
  per backend. Here, one hand-written Qwen.
- **Sampling on the GPU** (`crates/eta-*`): the inferlet's sampling code is
  compiled to run next to the logits, so every token can be picked without
  sending logits back.
- **Many machines**: a gateway takes requests, a controller places them,
  workers own GPUs (`crates/gateway`, `controller`, `worker`, `transport`).
  Here, one process.
- **More interfaces**: grammars, tools and reasoning, images, audio, video,
  and recurrent, hybrid and diffusion models (`crates/inferlet/wit`).
- **Python and JavaScript inferlet SDKs** (`sdk/`).
- **Swapping KV to CPU memory**, written in the planner but not yet enabled
  by any backend.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p client
cargo build --release -p text-completion -p chat-session --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie --serve 127.0.0.1:9123
./target/release/pie-client target/wasm32-wasip2/release/chat_session.wasm
```
