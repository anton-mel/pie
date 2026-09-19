# Chapter #25: Model Descriptions

Until chapter 24, the model was a hand-written Qwen: it guessed the family's
quirks from which tensors a checkpoint happened to have, and any other
family was refused.

In chapter 25 each family is described once (`crates/models/src/description.rs`)
and one transformer follows the description (`transformer.rs`, formerly
`qwen.rs`), as the reference describes every family once and runs them all
with one engine. A `Description` holds the sizes, and the switches in which
families differ: biases on the query, key and value projections (Qwen2), a
norm on each query and key head (Qwen3), tied embeddings, and the rotary
base with Llama 3's frequency scaling. `describe` reads it from a model's
config; there are three families: `qwen2`, `qwen3` and `llama`.

The chat format is now read from the model itself. A family does not fix
it: SmolLM2 is a `llama` model that speaks ChatML. The worker looks at the
chat template a model ships with (`chat_template::detect`) and falls back to
the family only when there is none.

Two Llama models run with no other change, next to Qwen:

| model | family | chat format |
|---|---|---|
| `Qwen/Qwen3-0.6B` | qwen3 | ChatML |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | llama | ChatML |
| `unsloth/Llama-3.2-1B-Instruct` | llama, rotary scaling | Llama 3 |

```
$ pie run --model unsloth/Llama-3.2-1B-Instruct chat.wasm -- "What is the capital of France?" "And of Germany?" "Which of the two cities is bigger?"
    assistant: Paris.
    assistant: Berlin.
    assistant: Berlin is larger.
```

> [!NOTE]
> The reference goes much further: a family's forward pass is written in a
> DSL, compiled into a plan per GPU backend (`crates/model-ir`, `model-dsl`,
> `model-compiler`), and checkpoints are checked against an import contract
> (`crates/checkpoint`). Here a description is data, and the transformer is
> still ordinary candle code.

## Read Order

Read `crates/models/src/description.rs`, then `Model::load` in
`crates/models/src/transformer.rs`. Then `Files` in
`crates/worker/src/weights.rs`, and `detect` in `crates/chat-template`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p chat --target wasm32-wasip2

./target/release/pie run --model HuggingFaceTB/SmolLM2-135M-Instruct target/wasm32-wasip2/release/chat.wasm -- "What is the capital of France?"
./target/release/pie run --model unsloth/Llama-3.2-1B-Instruct target/wasm32-wasip2/release/chat.wasm -- "What is the capital of France?"
```
