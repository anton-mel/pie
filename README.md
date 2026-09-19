# Chapter #26: Sampling on the GPU

Until chapter 25, every output row went back to the host as a full row of
logits, about 150,000 numbers, and the sampler picked from the 64 most
likely (chapter 7). The rest of the vocabulary could never be picked.

In chapter 26 an inferlet can attach a small sampling program to a forward
(`sampler` in `wit/forward.wit`): a temperature and a min-p cut. The engine
then picks the token on the device, right after the output head, and only
the token and its probability come back (`Row::Sampled` in
`crates/engine`). It picks with the Gumbel-max trick: the argmax of
`logits / T + g`, where `g` is Gumbel noise, is an exact sample from
softmax(logits / T) over the whole vocabulary.

On Metal the pick is a kernel of our own (`crates/models/src/sample_kernel.rs`):
one threadgroup per row applies the min-p cut, adds noise made from a hash
of (seed, row, token), and reduces to the argmax. The first version used
candle's GPU random numbers instead, and a statistical test caught it: over
8,000 samples the most likely token came up 0.331 of the time instead of
0.357, five standard deviations off, because Gumbel-max depends on the
noise's tails. The same test on the CPU was right, and with our own noise
the GPU is too:

| setting | " blue" | " red" | " green" | excluded picked |
|---|---|---|---|---|
| T = 1 | 0.366 / 0.357 | 0.237 / 0.245 | 0.082 / 0.085 | 0 |
| T = 0.5 | 0.634 / 0.637 | 0.297 / 0.301 | 0.036 / 0.036 | 0 |
| T = 1, min-p 0.3 | 0.589 / 0.593 | 0.411 / 0.407 | 0 / 0 | 0 |

(sampled / exact, 8,000 samples each, all within two standard deviations)

`tests/inferlets/device-sampling` is `parallel-sampling` with its picks
moved to the device. On this Mac it is 6-8% faster: memory is shared, so
the logits never had far to go. The gain that matters is that every token
can now be picked.

> [!NOTE]
> The reference goes much further: an inferlet writes its sampler as code
> (the ETA language), which is compiled and run next to the logits, so any
> sampler, not only these two knobs, runs on the device.

## Read Order

Read `Row` and `Sampling` in `crates/engine/src/lib.rs`, then `sample` at
the end of `crates/models/src/transformer.rs` and
`crates/models/src/sample_kernel.rs`. Then `sampler` in `wit/forward.wit`,
`submit_sampled` in `crates/inferlet/src/lib.rs`, and
`tests/inferlets/device-sampling`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p device-sampling -p parallel-sampling --target wasm32-wasip2

./target/release/pie run target/wasm32-wasip2/release/device_sampling.wasm -- "Once upon a time" 8 48
./target/release/pie run target/wasm32-wasip2/release/parallel_sampling.wasm -- "Once upon a time" 8 48
```
