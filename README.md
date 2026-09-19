# Pie Tutorial: Planner

Until chapter 4, `reserve` failed as soon as the KV pool was empty. With many
inferlets that is worse than it sounds: each grabs part of the pool, none can
finish, and all but one fail with "out of KV pages" even when the pool could
serve them one after another. For a local inference, this may be a frequent 
problem due to a limited number of resources.

In chapter 5, an inferlet that asks for pages that are not free waits for someone to
free some (`alloc_wait` in `runtime/src/engine.rs`). That works while at
least one inferlet is still running. When every live inferlet is waiting,
nobody will ever free a page: that is a deadlock, and the planner
(`runtime/src/planner.rs`) breaks it by evicting the youngest. Evicting kills
its wasm instance, which frees every page it held, and it restarts from
scratch once another inferlet has finished. The oldest is never evicted, so
it always makes progress. This is how the OS handles memory pressure: it
picks a victim (the OOM killer) rather than letting everyone hang.

> [!CAUTION]
> An evicted inferlet starts over, so the work it had done is lost and runs
> again. A cheaper way is swapping: copy the victim's KV pages from GPU memory
> to the much larger SSD, free them on the GPU, and copy them back
> when there is room. Not implemented.

## Read Order

Read `runtime/src/planner.rs` from the top: `wait` decides when to evict,
`leave` and `until_exit` decide when to restart. Then `alloc_wait` in
`runtime/src/engine.rs`, which `reserve` and the copy-on-write step in
`forward` now use (`runtime/src/host.rs`). Finally `Host::run`, which kills
an evicted instance and starts it again.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion -p beam-search --target wasm32-wasip2

# 4 runs, each needing 4 pages, in a pool of 8 pages
./target/release/pie --kv-pages 8 -i 4 target/wasm32-wasip2/release/text_completion.wasm -- \
  "In a small village at the edge of a dense forest, there lived an old clockmaker who repaired every clock in town. One winter morning" 40
```
