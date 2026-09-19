# Chapter #30: Python SDK

Until chapter 29, inferlets were written in Rust. But an inferlet is just a
WebAssembly component that imports the `pie:inferlet` interfaces and
exports `run`, and the runtime never asks what language it came from.

In chapter 30 inferlets can be written in Python, as in the reference.
`sdk/inferlet/python/src/inferlet` is the Rust SDK written again over the
same WIT: `Context` (fill, forward, generate, fork, rollback, system, user,
equip, reply), `Sampler`, `greedy`, and the host interfaces themselves
(`chat`, `tools`, `reasoning`, `grammar`, `session`, ...). An inferlet is a
module with a `main(args)` and a line `Run = inferlet.export(main)`.

[componentize-py](https://github.com/bytecodealliance/componentize-py)
generates Python bindings from `crates/inferlet/wit`, and bundles the
inferlet, the SDK and a Python interpreter (CPython built for WASI) into one
component, 18 MB. The runtime runs it unchanged: nothing in this chapter
touches Rust. Host errors arrive in Python as exceptions, and an exception
that escapes `main` becomes the error `run` returns.

The three examples in `sdk/inferlet/python/examples` give the same output
as their Rust versions, token for token, including the sampled chat (the
sampler uses the same random generator):

```
completion.py  Paris. The capital of Italy is Rome. The capital of Spain is ...
chat.py        user: What is the capital of France?
               assistant: The capital of France is Paris.  (thought for 43 words)
agent.py       round 1: thought 93 words, called get_weather({"city":"Paris"}) -> 18°C, sunny
               round 1: thought 93 words, called calculate({"expression":"17 * 23"}) -> 391
               round 2: thought 91 words, answered: The weather in Paris is 18°C and sunny. ...
```

The chat takes 1.2s in both languages: the time goes to the model, not to
the Python loop between forwards.

> [!NOTE]
> The reference builds Python inferlets with a "factored" componentize-py:
> the interpreter is a shared core module the runtime loads once, so each
> inferlet is small. It also has a JavaScript SDK (through jco), and an
> async Python API. Here every inferlet carries its own interpreter.

## Read Order

Read `sdk/inferlet/python/src/inferlet/__init__.py` next to
`crates/inferlet/src/lib.rs`. Then the examples in
`sdk/inferlet/python/examples`, next to `tests/inferlets/chat` and
`tests/inferlets/agent`.

## Run MacOS

```bash
cargo build --release -p pie --features metal

# componentize-py, through uv (https://docs.astral.sh/uv/)
uvx componentize-py -d crates/inferlet/wit -w inferlet componentize \
  -p sdk/inferlet/python/src -p sdk/inferlet/python/examples \
  agent -o target/python/agent.wasm

./target/release/pie run target/python/agent.wasm -- \
  "What is the weather in Paris, and what is 17 * 23?"
```
