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

**Chapter 2: the KV working set.** In chapter 1 an inferlet held raw page ids
from `alloc-pages` and had to give them back with `free-pages`. That leaked the
engine's memory layout to the guest and let it free pages it was still using.
Now a sequence's KV cache is a `kv-working-set` resource (`wit/pie.wit`):

- The inferlet addresses its pages as `0..page-len`; the host maps them to
  physical pages (`KvWorkingSet` in `runtime/src/host.rs`). This indirection is
  what later chapters build on: fork, sharing a prefix, moving pages.
- `forward` takes a borrowed working set and `kv-len` instead of a page list.
- Dropping the handle frees its pages. The host stores working sets in the
  instance's resource table, so an inferlet can only name its own, and
  whatever it leaks is freed with the instance.

Read `wit/pie.wit`, then `KvWorkingSet` and `forward` in `runtime/src/host.rs`,
then `Context::forward` in `inferlet/src/lib.rs`, which no longer needs `Drop`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie -i 8 target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
```
