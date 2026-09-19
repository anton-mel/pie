# Chapter #29: Tools and Reasoning

Until chapter 28, an inferlet could talk to the model and constrain what it
writes, but two things every agent needs were left to it: offering tools in
the exact format the model was trained on, and reading the model's output
to find its thinking and its tool calls.

In chapter 29 the chat template knows the model's tool format, as in the
reference, and inferlets use it through two interfaces:

- `tools.wit`: `equip` returns the text that offers a list of tools (JSON
  function descriptions) in the model's system prompt; `answer` turns the
  results of the tools called in one turn into tokens to append; a
  `decoder` is fed tokens as they are generated and reports each complete
  call (name and JSON arguments).
- `reasoning.wit`: a `decoder` fed the same tokens reports when thinking
  starts, each new piece of it, and all of it once it ends.

For Qwen3 the text is byte for byte what its official template produces
(`<tools>`, `<tool_call>`, `<tool_response>`, `<think>`); a test checks it.
Results of several calls go in one user turn, as the template does.

`Context::equip` adds the offer to the system prompt, and
`tests/inferlets/agent` runs the loop: generate, feed both decoders, run
each call the model makes (a fake weather table and a calculator, plain
inferlet code), give the results back, and repeat until the model answers
without calling anything:

```
round 1: thought 93 words, called get_weather({"city":"Paris"}) -> 18°C, sunny
round 1: thought 93 words, called calculate({"expression":"17 * 23"}) -> 391
round 2: thought 91 words, answered: The weather in Paris is 18°C and sunny.
         The result of 17 multiplied by 23 is 391.
```

Nothing about tools runs in the runtime's loop: the model's calls are plain
tokens, and the inferlet decides what a call does.

> [!NOTE]
> Greedy decoding with a 0.6B model is fragile: a tiny numeric difference
> changes the path. Asked which of Tokyo and London is warmer, the model
> on Metal calls both tools and then stops in the middle of its second
> thought; Hugging Face transformers (fp32), given the same prompt,
> answers "12°C"; and on our CPU the model decides not to call the tools
> at all.
>
> The reference also knows the tool and thinking formats of other model
> families, and decodes tool calls written in other ways than JSON between
> markers. Here only ChatML (Qwen) has them.

## Read Order

Read the new `Template` methods in `crates/chat-template/src/lib.rs`
(`tools`, `tool_results`, `tool_call_markers`, `thinking_markers`) and the
`chatml_tools` test. Then `wit/tools.wit`, `wit/reasoning.wit`,
`crates/runtime/src/inferlet/host/tools.rs` and `host/reasoning.rs`, and
`equip` in `crates/inferlet/src/lib.rs`. Finally `tests/inferlets/agent`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo test -p chat-template
cargo build --release -p pie --features metal
cargo build --release -p agent --target wasm32-wasip2

./target/release/pie run target/wasm32-wasip2/release/agent.wasm -- \
  "What is the weather in Paris, and what is 17 * 23?"
```
