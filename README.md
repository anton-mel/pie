# Chapter #7: Sampling

Until chapter 6, every example decoded greedily: always the most likely
token. Real generation samples, with a temperature and a cut-off like
top-p. In most engines that is a fixed menu of options passed with the
request. In Pie it is inferlet code.

In chapter 7 the engine and `wit/pie.wit` do not change at all. `forward`
already returns the top-k of the next-token distribution, and that is
enough: `Sampler` (`inferlet/src/sample.rs`) applies temperature, top-p and
min-p to it and draws a token, with randomness from WASI. An inferlet can
write any other sampler the same way, without asking the engine for it.

`examples/parallel-sampling` puts the earlier chapters together: the prompt
runs once and is forked into several branches (chapter 3) that share its
pages, each branch samples on its own, and every step submits all branches
before waiting, so they decode in one batch (chapter 4).

> [!NOTE]
> Sampling works on the top k tokens the engine returns (64 in the
> example), not the whole vocabulary. At usual temperatures the rest carry
> almost no probability. The current Pie goes further and runs the
> inferlet's sampling code on the GPU, so it can see every token without
> sending them to the host.

## Read Order

Read `Sampler::sample` in `inferlet/src/sample.rs`, then
`examples/parallel-sampling`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p parallel-sampling --target wasm32-wasip2

# 4 samples, 24 tokens, temperature 0.8, top-p 0.95
./target/release/pie target/wasm32-wasip2/release/parallel_sampling.wasm -- "Once upon a time" 4 24 0.8 0.95
```
