# Chapter #20: Prefix Trie

Until chapter 19, inferlets could share a prompt only through the index of
chapter 9: one publishes under a key, the others open that key. They had to
agree on the key, and the shared text had to be the whole key's text.

In chapter 20 prompts are shared by their tokens, as in the reference and in
vLLM. Every full page of KV is recorded under a chain hash: the hash of its
tokens together with the chain hash of the page before it
(`crates/runtime/src/store.rs`). Two sequences that start with the same
tokens get the same chain, like two paths through a trie that share their
start. `from-prefix` (`wit/working-set.wit`) returns a working set holding
the longest recorded prefix of a list of tokens, and `Context::with_tokens`
builds a context on it: only the rest of the prompt runs.

In `tests/inferlets/auto-prefix` every request is a long system prompt and
its own question. Asked three different questions in turn, the second and
third reuse 288 of about 310 prompt tokens: the system prompt, found by its
tokens, with nothing published and no key. Asked the same question again,
it reuses 304 of 312 and answers in 132ms instead of 197ms, with the same
answer.

> [!WARNING]
> A page's KV depends on every token before it, not only on its own. So
> only clean working sets record their pages: none of their pages were
> discarded (chapter 8), and every token sits at its own position. Pages are
> recorded only after the model has written them, so no one can read them
> too early, and recorded pages are the first to go when memory runs low.

## Read Order

Read `crates/runtime/src/store.rs`, then `record`, `lookup` and
`evict_oldest` in `crates/runtime/src/engine.rs`. Then where `forward` in
`crates/runtime/src/inferlet/host/forward.rs` tracks tokens and records
pages once the model has run (`on_done`), and `from_prefix` in
`host/kv_working_set.rs`. Finally `Context::with_tokens` in
`crates/inferlet/src/lib.rs` and `tests/inferlets/auto-prefix`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p auto-prefix --target wasm32-wasip2

./target/release/pie -i 3 --sequential target/wasm32-wasip2/release/auto_prefix.wasm -- "How long can I keep a DVD?" 24
```
