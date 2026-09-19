# Pie Tutorial

A minimal rebuild of [Pie](https://github.com/pie-project/pie) (SOSP'25), one
commit per chapter. Each commit adds one feature and explains how it fits in.
Expect to spend most of your time reading the code. The chapters walk through
the design decisions behind Pie in a logical order that is easy to follow, and
document the codebase along the way.

> [!NOTE]
> Read the files in the order listed under **Read Order** below.

## Big Picture

Most LLM servers hard-code the generation loop (tokenize, forward, sample,
repeat) in the engine. Pie moves it out: the loop runs in small user programs,
**inferlets**, compiled to WebAssembly. Think of it as eBPF for inference. An
inferlet is a *habit* the model runs with: forgetting, rethinking, taking notes,
working through a large file that needs careful control of the KV cache and the
loop at runtime. What you can build is limited mostly by imagination. The engine
exposes only low-level primitives: KV pages, the tokenizer, and `forward`.

```
 ┌────────────┐  ┌────────────┐  ┌────────────┐
 │ inferlet 1 │  │ inferlet 2 │  │ inferlet 3 │
 └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
        └───────────────┼───────────────┘
                        │ wit/pie.wit    forward ▼   top-k ▲
 ┌──────────────────────┼─────────────────────────────┐
 │ pie binary (main.rs: loads model, wires it up)     │
 │ ┌────────────────────▼───────────────────────────┐ │
 │ │ host.rs    runs inferlets, checks page owners  │ │
 │ └────────────────────┬───────────────────────────┘ │
 │                      │ alloc / free / forward      │
 │ ┌────────────────────▼───────────────────────────┐ │
 │ │ engine.rs  KV page pool + batcher              │ │
 │ └────────────────────┬───────────────────────────┘ │
 │                      │ all queued calls, one batch │
 │ ┌────────────────────▼───────────────────────────┐ │
 │ │ model.rs   Qwen transformer + paged KV cache   │ │
 │ └────────────────────────────────────────────────┘ │
 └────────────────────────────────────────────────────┘
```

## Read Order

Start at `wit/pie.wit`, the whole contract between an inferlet and the runtime.
Then read `examples/text-completion`, a greedy inferlet in about 10 lines that
runs simple generation, and `inferlet/src/lib.rs`, where `Context::forward`
turns tokens into pages and positions. Next, the runtime: `main.rs` loads the
model and starts the inferlets, `host.rs` implements the contract (`alloc_pages`,
`forward`) and checks page ownership, `engine.rs` holds the page pool and
batches calls in `batch_loop`, and `model.rs` runs one step over all
sequences in `Model::forward`, with attention over each sequence's own pages.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie -i 8 target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
```
