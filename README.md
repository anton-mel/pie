# Chapter #19: Sandbox

Until chapter 18, an inferlet could reach nothing but the model and stdio:
no files, no network. That is safe, but an agent that uses tools needs to
read a document or call a service.

In chapter 19 each instance runs under a policy
(`crates/runtime/src/inferlet/sandbox.rs`) that decides what it may reach
besides the model. By default, still nothing. `--allow-dir DIR` lets
inferlets read `DIR`, which they see as `/data` (`--allow-write` also lets
them change it), and `--allow-connect IP:PORT` lets them open TCP
connections to that address and no other. The policy becomes the instance's
WASI context, so the inferlet uses ordinary file and socket calls, and the
runtime answers them or refuses them.

`tests/inferlets/ask-docs` answers a question about a
document it fetches itself, from `/data` or over HTTP:

| policy | `/data/notes.txt` | `http://127.0.0.1:8765/notes.txt` |
|---|---|---|
| none | no such file | permission denied |
| `--allow-dir docs` | answered | permission denied |
| `--allow-connect 127.0.0.1:8765` | no such file | answered |

With `--allow-connect 127.0.0.1:8765`, the same server on port 8766 is
refused, and with `--allow-dir`, nothing outside `/data` is visible.

> [!NOTE]
> The policy is set when `pie` starts and applies to every instance. The
> reference sets it per instance, with allow and deny rules, and also links
> WASI HTTP, so inferlets can make HTTP requests without writing them by
> hand.

## Read Order

Read `Policy` in `crates/runtime/src/inferlet/sandbox.rs`, then where
`Host::run_once` builds each instance's WASI context from it in
`crates/runtime/src/inferlet.rs`, and the new flags in `src/main.rs`.
Finally `tests/inferlets/ask-docs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p ask-docs --target wasm32-wasip2

mkdir -p docs && echo "The wifi password is sunflower42." > docs/notes.txt
./target/release/pie target/wasm32-wasip2/release/ask_docs.wasm -- /data/notes.txt "What is the wifi password?"
./target/release/pie --allow-dir docs target/wasm32-wasip2/release/ask_docs.wasm -- /data/notes.txt "What is the wifi password?"
```
