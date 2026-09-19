# Chapter #22: Chat Templates

Until chapter 21, the chat format was ChatML, written by hand into the host
(chapter 11). It is right for Qwen and wrong for every other model family.

In chapter 22 the format is a template, picked by the model's config, as in
the reference. `crates/chat-template` has a `Template` trait and four
families: ChatML (Qwen), Llama 3, Gemma and DeepSeek V3. The worker reads the
config's `model_type`, picks the template (`for_model`), and refuses a model
it has none for. The host's `chat` interface renders through it, so neither
the host nor the inferlet spells any format.

The interface gains two functions, as in the reference. `prefix` is what a
conversation starts with: `<|begin_of_text|>` for Llama, `<bos>` for Gemma,
nothing for Qwen. `system-user` writes a system prompt with the first user
message, because Gemma has no system turn and folds the system prompt into
that message. `Context` now holds a system prompt until the first user
message, and writes the prefix before the first message.

Our engine runs only Qwen, so the other three formats cannot be tried on a
model. Instead, the crate's tests compare each template with the family's
official chat template, taken from its Hugging Face `tokenizer_config.json`
and rendered with Jinja for the same conversation. All four match exactly.
For Qwen, every chat gives the same answers as before.

> [!NOTE]
> The reference also has decoders that follow a reply as it is generated:
> where thinking starts and ends, and where a tool call begins. Here, the
> SDK still splits Qwen's `<think>` block out of a finished reply.

## Read Order

Read `crates/chat-template/src/lib.rs`, with its tests at the end. Then
`prefix` and `system-user` in `wit/chat.wit`, `chat::Host` in
`crates/runtime/src/inferlet/host/chat.rs`, where the worker picks the
template in `crates/worker/src/lib.rs`, and `system`, `user` and `start` in
`crates/inferlet/src/lib.rs`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo test -p chat-template
cargo build --release -p pie --features metal
cargo build --release -p chat --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/chat.wasm -- "What is the capital of France?"
```
