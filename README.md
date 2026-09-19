# Chapter #24: Our Own GPU Kernel

Until chapter 23, candle did all the GPU work. For attention while decoding
(chapter 14) that meant gathering every sequence's cached keys and values
into a new tensor, every step, in every layer, and only then multiplying.
The longer the context, the more was copied.

In chapter 24 decoding attention on Metal runs in a kernel of our own
(`crates/models/src/paged_attention.rs`), as the reference has for CUDA,
Metal, Vulkan and WebGPU. It is written in Metal Shading Language, compiled
when first used, and plugged into candle as a custom op. One threadgroup
takes one sequence and one query head. It walks the sequence's page table
and reads each key and value straight from its slot in the cache, so
nothing is copied. Scores are taken in blocks of 128 positions with a
running maximum and sum (online softmax, as in flash decoding), so its
memory does not grow with the context.

Its unit test runs it in f32 on random data, with scattered pages, three
sequence lengths and grouped-query heads, and compares it with plain
attention computed on the CPU: they agree to within 1e-4. Greedy output is
the same as before. Median of three runs, gather against kernel:

| | gather | kernel |
|---|---|---|
| beam search, 8 beams | 604 ms | 535 ms |
| beam search, 16 beams | 888 ms | 753 ms |
| 16 inferlets, 48 tokens each | 1800 ms | 1500 ms |
| 4 inferlets, 100 tokens after ~1,500 | 10.9 s | 7.1 s |

`PIE_GATHER_ATTENTION=1` switches back to the gather, for comparison.

> [!NOTE]
> The kernel sums in f32, the gather in bf16, so sampled outputs can take a
> different path after a close call. The reference has kernels for every
> operation of the model on four GPU APIs; here there is one, for Metal,
> and candle still does the rest. Prefill keeps the old path.

## Read Order

Read the Metal source at the top of `crates/models/src/paged_attention.rs`,
then `PagedDecode::metal_fwd` and the test at the end. Then where
`attend_decode` in `crates/models/src/qwen.rs` uses it.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo test --release -p models --features metal
cargo build --release -p pie --features metal
cargo build --release -p beam-search --target wasm32-wasip2

./target/release/pie run target/wasm32-wasip2/release/beam_search.wasm -- "Once upon a time" 16 32
PIE_GATHER_ATTENTION=1 ./target/release/pie run target/wasm32-wasip2/release/beam_search.wasm -- "Once upon a time" 16 32
```
