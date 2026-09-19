# Pie Tutorial: async forward

Until chapter 3, `forward` blocked until the model had run it. The engine
batches calls from different inferlets, but an inferlet with several
sequences, like the beams in `examples/beam-search`, could only have one
forward in flight: its beams ran one model step each.

Now `forward` returns a `pending-forward` at once, and `wait()` gets the
result (`wit/pie.wit`). The host keeps what an inferlet submits and hands it
to the engine all together when the inferlet next waits, so everything
submitted before a wait lands in the same model step. This is how the OS
does async I/O: submit many requests, then wait for them, instead of one
blocking call at a time.

Beam search now submits all beams, then waits on each. With 8 beams each
decoding step is one batch of 8 instead of 8 batches of 1 (0.73s instead of
1.2s for 32 tokens). It is not 8x, because attention still runs sequence by
sequence and top-k runs on the CPU. Those belong to the engine, not to Pie's
design, and a later chapter fixes them.

> [!WARNING]
> A forward in flight must keep its pages. The inferlet can drop a working
> set while its forward is still queued, and without care those pages would
> go back to the pool and be handed to someone else before the model writes
> into them. So every request holds its pages (`Hold` in
> `runtime/src/engine.rs`) until the model has run it.

## Read Order

Read `pending-forward` and `forward` in `wit/pie.wit`. Then, in
`runtime/src/host.rs`, `forward` (it now queues a request in `unsent`) and
`wait` (it sends `unsent` to the engine). In `runtime/src/engine.rs`, read
`Request`, `Hold` and `batch_loop`. Finally `Context::submit` in
`inferlet/src/lib.rs` and the submit-then-wait loop in `examples/beam-search`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion -p beam-search --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie target/wasm32-wasip2/release/beam_search.wasm -- "Once upon a time" 8 32
```
