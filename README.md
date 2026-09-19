# Chapter #11: Chat

Until chapter 10, every example fed the model raw text. Instruct models are
trained on conversations written in a fixed format, with a marker for each
turn and a token that ends the assistant's reply. Without that format they
ramble or ignore the question.

In chapter 11 the host tells the inferlet how the model spells a
conversation (`chat` in `wit/pie.wit`): `system`, `user` and `assistant`
return the tokens for one message, `cue` starts the assistant's reply, and
`seal` closes it. The inferlet never writes the format itself, so the same
inferlet works with any model whose host knows its format. Ours knows one:
ChatML, which Qwen uses (`chat::Host` in `runtime/src/host.rs`).

A conversation is just a context that grows (`examples/chat`): each
`Context::reply` only runs the new message, not the whole history. Qwen3
thinks before it answers, between `<think>` and `</think>`, and its own chat
template leaves that thinking out of the history. `reply` does the same: it
rolls back the reply (chapter 6) and puts back only the answer. Without that,
the model breaks down on the second turn.

> [!NOTE]
> The reference Pie also has `tools` (the model calls a function and the
> inferlet feeds the result back) and `reasoning` (detecting thinking as it
> is generated). They use the same chat format and are not done here.

## Read Order

Read `chat` in `wit/pie.wit`, then `chat::Host` in `runtime/src/host.rs`.
In `inferlet/src/lib.rs`, read `Context::reply` and `Reply`. Finally
`examples/chat`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p chat --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/chat.wasm -- \
  "I have 3 apples and buy 5 more. How many do I have?" \
  "I eat 2 of them. How many are left?"
```
