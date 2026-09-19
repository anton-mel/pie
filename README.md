# Chapter #28: Grammar

Until chapter 27, an inferlet could restrict the next token (chapter 10),
but working out which tokens keep the output valid was left to it: fine
for a list of choices, not for JSON.

In chapter 28 the runtime has a grammar engine (`crates/grammar`), as in
the reference, and inferlets use it through `grammar.wit`. A matcher is made
from a regular expression, or from a JSON schema that `json_schema` turns
into one (objects, arrays, strings, numbers, booleans, null, enums). The
expression is compiled into a DFA over bytes. At each step the allowed
tokens are those whose bytes keep the DFA alive, found once per DFA state
and cached; end-of-sequence is allowed once the output is a full match.
Token bytes come from the tokenizer's byte-level alphabet, so a token that
holds half of a multi-byte character is handled correctly.

`Context::generate_matching` generates under a matcher, and
`tests/inferlets/json-extract` uses it to pull fields out of a sentence:

```
{
  "name": "Alice Moreau",
  "age": 34,
  "city": "Lyon",
  "job": "nurse"
}
```

Without the grammar, the same prompt gives
`` ```json\n{"name": "Alice Moreau"}\n``` ``: a code block, with one of
the four fields. The first version allowed only compact JSON; the model
wanted `"name": "Alice"`, with a space, and when the space was refused it
put a wrong token inside the string instead. A little whitespace between
elements (at most three spaces or newlines) is now allowed, as grammar
engines do.

> [!NOTE]
> The reference's grammar engine also compiles EBNF grammars, handles far
> more of JSON schema, and builds its token masks as bitmasks next to the
> logits. Here the allowed tokens are a list sent with the forward.

## Read Order

Read `crates/grammar/src/lib.rs` (`Vocab`, `Grammar`, `Matcher`) and
`json_schema.rs`, with the tests at the end. Then `wit/grammar.wit`,
`crates/runtime/src/inferlet/host/grammar.rs`, `vocab` in
`crates/runtime/src/engine.rs`, and `generate_matching` in
`crates/inferlet/src/lib.rs`. Finally `tests/inferlets/json-extract`.

## Run MacOS

```bash
rustup target add wasm32-wasip2
cargo test -p grammar
cargo build --release -p pie --features metal
cargo build --release -p json-extract --target wasm32-wasip2

./target/release/pie run target/wasm32-wasip2/release/json_extract.wasm -- \
  "Bob Chen, 52, a civil engineer, moved to Toronto last year."
```
