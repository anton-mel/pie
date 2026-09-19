# Chapter #9: Prefix Cache

Until chapter 8, working sets could share pages only through `fork`, inside
one inferlet. Two inferlets that start with the same text, like the same
long system prompt, each computed it from scratch.

In chapter 9 a working set can be published under a key (`update-index` in
`wit/pie.wit`), and any inferlet can open it later (`from-index`). Opening
works like a fork: the pages are shared and copied only when written. The
engine keeps the published pages in an index (`Index` in
`runtime/src/engine.rs`). The index is a cache: when an allocation would
have to wait, the engine first drops the entry used longest ago, so
published pages never starve running inferlets (chapter 5).

`Context::cached(text)` wraps this: the first inferlet to ask runs `text`
and publishes it, and later ones open it and skip the work. In
`examples/prefix-cache` every request starts with the same 290-token system
prompt. Run one after another, the first computes it (390ms) and the others
reuse it (195ms each), with exactly the same answers.

> [!NOTE]
> Only inferlets that start after the prefix is published reuse it.
> Inferlets that start together all miss and each compute it. That is why
> `pie` has a new `--sequential` flag for this example.

## Read Order

Read `update-index` and `from-index` in `wit/pie.wit` and in
`runtime/src/host.rs`. Then `Index`, `publish`, `open` and `evict_oldest` in
`runtime/src/engine.rs`, and where `alloc_wait` calls `evict_oldest`.
Finally `Context::cached` in `inferlet/src/lib.rs` and
`examples/prefix-cache`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p prefix-cache --target wasm32-wasip2

# 4 requests one after another: the first computes the system prompt, the rest reuse it
./target/release/pie -i 4 --sequential target/wasm32-wasip2/release/prefix_cache.wasm -- "How long can I keep a DVD?" 24
```
