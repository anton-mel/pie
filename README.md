# Pie Tutorial: KV working set

In chapter 1 an inferlet held raw page ids
from `alloc-pages` and had to give them back with `free-pages`. That leaked the
engine's memory layout to the guest and let it free pages it was still using.
Now a sequence's KV cache is a `kv-working-set` resource (`wit/pie.wit`).
Recall, OS has a similar security primitive called virtual memory.

The inferlet addresses its pages as `0..page-len`; the host maps them to
physical pages (`KvWorkingSet` in `runtime/src/host.rs`). This indirection is
what later chapters build on: fork, sharing a prefix, moving pages.

## Read Order

Read `wit/pie.wit`, then `KvWorkingSet` and `forward` in `runtime/src/host.rs`,
then `Context::forward` in `inferlet/src/lib.rs`, which no longer needs `Drop`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
./target/release/pie -i 8 target/wasm32-wasip2/release/text_completion.wasm -- "The capital of France is" 24
```
