# Chapter #21: Pipelines

Until chapter 20, the order work ran in was implicit: the scheduler kept a
forward behind an earlier one whenever it read pages the earlier one still
had to write. That is correct, but it guesses at what depends on what, and
the inferlet has no say in it.

In chapter 21 every forward is submitted on a pipeline the inferlet chooses
(`wit/pipeline.wit`), as in the reference. Forwards on one pipeline run in
the order they were submitted; forwards on different pipelines are
independent. The scheduler (`crates/runtime/src/scheduler.rs`) runs a
forward once every earlier one on its pipeline has finished, or is running
in full in the same step, where arrival order puts it first. So eight beams
on one pipeline still run as one batch. Each `Context` has a pipeline, and
its forks share it, because a fork reads what its parent wrote.

Two forwards on one context, submitted back to back before either is waited
for, now give exactly the same result as one forward over all the tokens,
whether they run in one step or are split across many.

> [!WARNING]
> Making them run in one step exposed two older assumptions. A forward
> still in flight used to count as another owner of its pages, so the next
> forward on the same pages copied them for no reason. In-flight holds are
> now counted apart from owners (`pins` in `crates/runtime/src/engine.rs`).
> And a step copies pages before it writes any, so a forward that does copy
> a page waits while an earlier one still writes it. Beam search now makes
> fewer copies, and is a little faster (606ms instead of 618ms).

## Read Order

Read `wit/pipeline.wit` and `on` in `forward` in `wit/forward.wit`. Then
the pipeline rule in `run` in `crates/runtime/src/scheduler.rs`, `pins`
and `is_shared` in `crates/runtime/src/engine.rs`, and the `pipeline` field
of `Context` in `crates/inferlet/src/lib.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p beam-search --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/beam_search.wasm -- "Once upon a time" 8 32
```
