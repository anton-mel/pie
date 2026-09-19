# Chapter #14: Batched Attention

Until chapter 13, the linear layers ran every sequence of a step as one
matrix, but attention looped over the sequences one at a time. Each one
wrote its K/V, gathered its cache slots, built its causal mask on the CPU,
and ran its own matmuls and softmax, in every layer. With 8 beams that is
thousands of small GPU operations per step, and it grows with every
sequence.

In chapter 14 the attention of a step is planned once and reused by every
layer (`Plan` in `runtime/src/model.rs`):

- every new token's K/V is written with one `scatter_set` per cache;
- every copy-on-write page (chapter 3) is copied with one gather and one
  scatter per cache;
- sequences with one new token, the common case when many beams, samples or
  clients decode together, are attended together: their cache slots are
  gathered once, padded to the longest with a mask over the padding, and
  run through one batched matmul and one softmax (`attend_decode`);
- longer ones (prefill, verification) keep their own attention, with masks
  built once per step instead of once per layer.

Results are unchanged, and batched work gets 18-25% faster:

| | chapter 13 | chapter 14 |
|---|---|---|
| beam search, 8 beams, 32 tokens | 752 ms | 617 ms |
| beam search, 16 beams | 1200 ms | 922 ms |
| 8 parallel samples | 794 ms | 609 ms |
| 16 inferlets, 48 tokens each | 2400 ms | 1800 ms |

> [!NOTE]
> Attention still gathers each sequence's pages into a new tensor before it
> reads them. The current Pie has its own GPU kernels (`crates/kernels-*`)
> that read the pages where they are, with no copy and no padding, for
> CUDA, Metal, Vulkan and WebGPU. What is left of the cost here is mostly
> the many small operations candle launches, which only such a kernel
> removes.

## Read Order

Read `Plan::new` in `runtime/src/model.rs`, then where `Model::forward` uses
the plan in its layer loop, then `attend_decode` and `attend`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p beam-search -p parallel-sampling --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/beam_search.wasm -- "Once upon a time" 8 32
./target/release/pie target/wasm32-wasip2/release/parallel_sampling.wasm -- "Once upon a time" 8 32
```
