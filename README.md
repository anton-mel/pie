# Pie Tutorial: KV fork

Chapter 3 borrows another OS idea. `fork()` makes a new working set that
shares all of its parent's pages (`wit/pie.wit`). Same as in the OS, no pages are copied at fork
time: the engine counts how many working sets hold each page, and a shared
page is copied only when one of them is about to write into it. This is how
the OS `fork()` shares memory between processes: copy-on-write.

This is what makes the KV cache programmable. Beam search, tree search and
parallel sampling are just a fork and a loop in the inferlet, not engine
features. See `examples/beam-search`: it runs the prompt once, and every beam
shares its pages from then on. The same holds for an agent that branches:
each branch reuses the shared context instead of recomputing it, which saves
both compute and memory.

> [!WARNING]
> One limitation is visible here: an inferlet calls `forward` for one beam at a time, so its beams are not batched with each other, only with other inferlets. The next chapter fixes that.

## Read Order

Read `fork` in `wit/pie.wit`, then `Pool` in `runtime/src/engine.rs` (a page
is free when nobody holds it), then `fork` and the copy-on-write step in
`forward` in `runtime/src/host.rs`, and the page copies at the top of
`Model::forward` in `runtime/src/model.rs`. Finally `Context::fork` in
`inferlet/src/lib.rs` and `examples/beam-search`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion -p beam-search --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie target/wasm32-wasip2/release/beam_search.wasm -- "The capital of France is" 4 16
```
