# Chapter #8: KV Discard

Until chapter 7, a sequence's KV cache only grew. Pages were freed when the
whole working set was dropped, never while it was in use. So one inferlet
could never generate more tokens than its pages could hold.

In chapter 8 a working set can `discard` pages it no longer needs
(`wit/pie.wit`). Later pages move down to fill the gap, the dropped pages go
back to the pool, and later forwards no longer attend to the tokens in them.
The positions of the tokens that remain do not change, so `Context` now
counts positions separately from how many tokens are cached.

What to drop is the inferlet's choice. `examples/attention-sink` keeps the
first page forever, because models lean heavily on the first tokens (the
"attention sink"), and after it only the last few pages, a sliding window.
It generates 512 tokens in a pool of 8 pages (128 tokens). Plain greedy
decoding in the same pool fails with "out of KV pages". Other policies, like
dropping the tokens that received the least attention, fit the same verb.

> [!NOTE]
> Discarding a page removes its tokens from attention, but the tokens after
> it were computed while they were still there, so they still carry some of
> what the dropped tokens said. Nothing is recomputed.

## Read Order

Read `discard` in `wit/pie.wit` and in `runtime/src/host.rs`. Then, in
`inferlet/src/lib.rs`, `Context::discard` and the new `pos` field, which
`submit_rows` now uses for positions. Finally `examples/attention-sink`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion -p attention-sink --target wasm32-wasip2

# 512 tokens in a pool of 8 pages: greedy fails, the attention sink does not
./target/release/pie --kv-pages 8 target/wasm32-wasip2/release/text_completion.wasm -- "Once upon a time" 512
./target/release/pie --kv-pages 8 target/wasm32-wasip2/release/attention_sink.wasm -- "Once upon a time" 512 6
```
