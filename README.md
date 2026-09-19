# Chapter #10: Constrained Decoding

Until chapter 9, the model could produce any token. Often the answer must
follow a format: one of a few labels, a number, valid JSON. Asking nicely in
the prompt is not enough, especially with a small model.

In chapter 10 `forward` takes an optional `allowed` list of token ids
(`wit/pie.wit`). The engine then picks only among those, with their
probabilities renormalized (`top_k` in `runtime/src/engine.rs`). Which
tokens are allowed is the inferlet's decision, made again at every step.

`examples/constrained-choice` makes the answer one of a list of choices. At
each step it allows only the tokens that can still lead to one of them. Asked
whether a review about cold food is positive, negative or neutral, greedy
decoding rambles ("The review is neutral. The review says..."), while the
constrained run answers exactly " negative", and shows how sure it was
(0.62, against 0.29 and 0.08).

> [!NOTE]
> A grammar works the same way, with a richer rule for what may come next:
> a JSON parser, for example, allows only tokens that keep the output valid
> JSON. The current Pie has a grammar interface that builds these masks
> for the inferlet, and applies them on the GPU.

## Read Order

Read `allowed` in `forward` in `wit/pie.wit`. Then `top_k` and where
`batch_loop` passes `allowed` to it, in `runtime/src/engine.rs`. In
`inferlet/src/lib.rs`, read `forward_allowed`. Finally
`examples/constrained-choice`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo build --release -p pie --features metal
cargo build --release -p constrained-choice --target wasm32-wasip2

./target/release/pie target/wasm32-wasip2/release/constrained_choice.wasm
./target/release/pie target/wasm32-wasip2/release/constrained_choice.wasm -- \
  "The Golden Gate Bridge is in the city of" "New York|Los Angeles|San Francisco|San Diego"
```
