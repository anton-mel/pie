# Chapter #6: Speculative Decoding

Until chapter 5, `forward` returned one distribution: the one after the last
new token. So an inferlet could only learn one new token per forward, and
decoding took one forward per token.

In chapter 6, `forward` takes `outputs`, the new tokens to return a
distribution for (`wit/pie.wit`). Asking for all of them lets an inferlet
check several guessed tokens in one forward: row `i` says what the model
would have produced after guess `i`. Guesses that match are kept for free,
and `Context::rollback` drops the rest. Asking for none is useful too: a
prefill whose result is not needed skips the output head.

That is speculative decoding, and it is written entirely in an inferlet
(`examples/speculative-decoding`). It needs no draft model: it guesses by
prompt lookup, copying what followed the last two tokens the previous time
they appeared. With greedy checking the output is exactly what greedy
decoding produces, only in fewer forwards. It pays off when the output
copies the input: copying a paragraph takes 7 forwards instead of 48, and
is 2.2x faster (129ms instead of 286ms). On free text it helps little.

> [!NOTE]
> This checks one chain of guesses. The latest Pie also lets a forward set
> its own attention mask, so an inferlet can check a whole tree of guesses
> in one forward, each branch seeing only its own ancestors.

## Read Order

Read `outputs` in `forward` and `wait` in `wit/pie.wit`. Then the output
rows at the end of `Model::forward` in `runtime/src/model.rs`, and how
`batch_loop` in `runtime/src/engine.rs` hands each request its own rows.
In `inferlet/src/lib.rs`, read `forward_all`, `rollback` and `submit_rows`.
Finally `examples/speculative-decoding`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p text-completion -p speculative-decoding --target wasm32-wasip2

P='Copy the following text exactly.

Text: The committee met on Tuesday to review the budget for the new library. After a long discussion, the members agreed to fund the reading room, the children section, and the new computers, but postponed the cafe until next year.

Copy: The committee met on Tuesday'

./target/release/pie target/wasm32-wasip2/release/text_completion.wasm -- "$P" 48
./target/release/pie target/wasm32-wasip2/release/speculative_decoding.wasm -- "$P" 48 8
```
