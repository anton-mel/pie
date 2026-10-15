# Inspecting an Inferlet

An inferlet is a program that controls inference next to a model. To see what that means in practice, we will take apart **beam search**, following its construction from input parsing through device execution to the returned text. At each step, read the Rust, Python, and JavaScript versions side by side.

This chapter uses the reference Pie repository, rather than one of the book's incremental runtime implementations. The complete programs and supporting READMEs are included in the source panel: `examples/beam-search/src/lib.rs`, `examples/beam-search-py/main.py`, and `examples/beam-search-js/index.js`. All three come from the commit shown above. The excerpts below omit whole comment lines and remove common indentation for space; they keep the original code, including language-specific error handling. They are pieces of the complete programs, rather than independently runnable fragments.

For background, read [Inferlets](01-inferlet.md), [KV fork](03-kv-fork.md), and [Async Forward](04-async-forward.md). This example exposes a more explicit device-side interface than those early chapters.

## What we are inspecting

The overview here follows `examples/beam-search/README.md`. Greedy decoding commits to the highest-scoring next token. Fixed-width beam search keeps several partial continuations, scores all their possible next tokens, and retains the best extensions **across all parents**.

| Property | This inferlet |
| --- | --- |
| Input | `max_tokens` (default `16`) and `beams` (default `2`) |
| Starting prefix | Fixed token `BOS = 1`, as configured for this Qwen example |
| Search objective | Cumulative token log-probability |
| State | Eight reserved logical KV pages shared across lanes, an ancestry mask for each beam, and cumulative scores |
| Output | The best continuation followed by a `[beam]` diagnostics line |
| Stopping rule | Exactly `max_tokens` steps; no EOS stopping or length normalization |
| Supported paths | Attention models and hybrid attention/recurrent models |

There is no `prompt` parameter. This is a mechanism example starting from a fixed token, not a general chat program. The code labels token ID 1 as `BOS` for its Qwen3-0.6B example; it does not query a tokenizer for a portable BOS ID. The page size and vocabulary size, in contrast, come from the bound model.

Here **host-side CPU code** means the inferlet code, which runs inside a Wasm guest from the runtime's perspective. **Device** means the model and ETA execution side. Pie's runtime connects them through resource calls and channels. This diagram shows the program's logical dataflow, not a kernel-by-kernel timeline.

```text
HOST CPU / inferlet guest
  parse input → reserve logical KV pages → seed channels
    → bind tokens + geometry + recurrent state → register epilogue
    → first submission traces/attaches ETA → runtime schedules forward(s)
                                                   │
                                                   ▼
DEVICE / one submitted forward
  token + position + KV geometry + mask (+ hybrid recurrent state)
    → model forward: write current token's KV, produce logits
    → epilogue: reshape raw logits to [B, V]
          ├─ raw-logit argmax → greedy output channel
          │
          └─ log_softmax + path scores → global top-B
               ├─ DEVICE FEEDBACK
               │    next tokens + scores + ancestry mask + page geometry
               │      → channels read by the next submitted forward
               │
               └─ OUTPUT CHANNELS
                    tokens + parent indices + scores
                                                   │
                                                   ▼
HOST CPU / inferlet guest
  drain all four outputs → rebuild token paths from selected parents
    ├─ attention-only: run_ahead maintains the submission window;
    │                  device feedback supplies the next inputs;
    │                  closes submissions at budget, then drains remaining outputs
    ├─ hybrid: fork recurrent state by parent → rebind
    │           → submit next if steps remain
    └─ after the final outputs: ensure pipeline closed → choose best lane
                                 → check width-1 identity → decode text
```

The attention-only path can submit later forwards before the CPU has reconstructed earlier token paths. They still depend on the preceding device-produced channel values. The hybrid path waits for parent indices before selecting the next recurrent-state bindings. The example forks and rebinds even after the final step, but submits again only if another step remains. The final selected token is returned in the path; it is not fed through another model forward just to populate its KV. These distinctions follow the epilogue and the two execution loops in `examples/beam-search/src/lib.rs`, together with the submission helpers in `crates/inferlet/src/eta.rs`.

Here `B` is the beam width, `V` the vocabulary size, and a *lane* is one row in the batch. A lane's identity can change each step: the next lane 0 might descend from the previous lane 1.

The search rule is:

```text
candidate[parent, token] = score[parent] + log p(token | parent prefix)
survivors = top B candidates over the whole [B, V] block
next_path[lane] = path[parent[lane]] + token[lane]
```

Beam search is exact for this fixed-width recurrence. A finite beam still prunes paths, so it does not guarantee the globally best sequence.

## 1. Establish the entry point

{{compare:entry}}

Rust imports the hybrid ETA prelude and declares a deserializable input structure. Its `#[inferlet::main]` wrapper supplies the inferlet entry point. Python exposes `async def main(input)`; JavaScript exports `main(input)`. Their defaults are read in the next block.

ETA is the device-program authoring interface used here. Its tensor expressions describe a graph that Pie executes beside the model. Python syntax does not imply a Python loop over vocabulary entries, and JavaScript syntax does not move logits into an ordinary JavaScript array.

Rust and Python can suspend on host reads. This JavaScript SDK uses blocking reads, so its entry point is synchronous. The supporting SDK documentation is included as `packages/python/inferlet/README.md` and `packages/javascript/inferlet/README.md`.

## 2. Read input and bound the work

{{compare:validation}}

The pool has `pool_len = 8 × page_t` cells. One cell holds the shared starting prefix, and each epilogue prepares `B` fresh cell positions for the next forward. The last prepared positions will not be written if generation ends there. The conservative generation bound is therefore:

```text
capacity = floor((pool_len - 1) / B)
```

For a page size of 32 and width 2, this gives 127 steps. The model supplies the actual page size; hardcoding it would make flat addresses disagree with the engine's layout.

A zero width fails, a width larger than the pool fails, and exceeding the step capacity fails. Zero steps returns an empty string before allocating state. Use positive integral widths and nonnegative integral step counts: Rust's unsigned input types enforce this more strictly than the Python and JavaScript conversions shown here. The dynamic-language examples do not provide equivalent validation for arbitrary negative or fractional inputs.

## 3. Reserve the shared KV pool

{{compare:pool}}

`WorkingSet.reserve` extends the logical page address space. The values called `pool_ids` are working-set-relative page indexes, not physical GPU page IDs. A flat cell `wpos` maps to logical page `pool_ids[wpos / page_t]` and offset `wpos % page_t`. The runtime resolves physical pages when executing a forward; reservation alone does not allocate all backing memory. This contract is explicit in `crates/inferlet/wit/working-set.wit` and implemented by the `WorkingSet` wrapper in `crates/inferlet/src/eta.rs`.

Every lane initially attends only cell 0. Its first forward processes BOS at position 0 and writes the shared prefix cell. The page-list channel has capacity for `B × POOL_PAGES` entries, but initially only one page per lane is live. The active page stride will grow as the pool fills.

Unlike the earlier [KV fork](03-kv-fork.md) example, branching attention history here does not call `WorkingSet.fork()`. Each lane inherits a mask over the same pool. The inferlet neither compacts dead token cells nor discards their logical pages, which explains the fixed address-space capacity limit.

## 4. Seed the loop-carried channels

{{compare:channels}}

A channel carries tensor values between executions of the forward pass. `take()` and `put()` inside the epilogue describe device dataflow; `take_host()` or `takeHost()` later transfers output to the host.

| Channel | Shape | Purpose |
| --- | --- | --- |
| `mask` | `[B, pool_len]` | Cells visible to each lane |
| `scores` | `[B]` | Cumulative log-probability |
| `toks`, `pos` | `[B]` | Next input token and its sequence position |
| `fill` | `[1]` | Next free flat pool cell |
| `klen` | `[B]` | Addressed logical pool span, before masking |
| `w_slot`, `w_off` | `[B]` | Logical page index and offset of the next write |
| `pages` | `[B × POOL_PAGES]` capacity | Packed page lists for all lanes |
| `page_indptr` | `[B + 1]` | Boundaries of those packed lists |
| `lanes_b` | `[B + 1]` | Embedding row boundaries: one token per lane |
| `pool_ids_ch` | `[POOL_PAGES]` | Pool IDs available to the device graph |

Only initial lane 0 has score 0; the others have negative infinity. That makes the first selection expand one starting hypothesis, rather than treating the identical BOS rows as independent candidates with equal mass.

Each excerpt includes `out`, `out_par`, and `out_scr` (camelCase in JavaScript). The remaining diagnostic port is:

{{compare:ports}}

All four host output channels have ring capacity from `channel_capacity()`. The async and blocking read contracts are defined in `crates/inferlet/wit/channel.wit`. They carry selected tokens, parent indices, scores, and an independent greedy argmax. The host must drain every output each step to keep those rings moving.

## 5. Bind the model pass to that state

{{compare:binding}}

The pipeline establishes submission order; the runtime schedules executions. The forward pass binds tokens, KV geometry, and any recurrent state. `embed(toks, lanes_b)` supplies one token per lane.

An attention model needs only the KV working set and mask. A hybrid model additionally needs one `RsWorkingSet` per lane. This program rejects recurrent-only and diffusion passes. The Rust API spells out `KvBinding` and `RsGeometry`; Python uses `KvGeometry` and `bind_state`, and JavaScript passes a geometry object to `bindState`. These are the actual APIs in the pinned examples, so this block is more than a name-for-name translation.

The geometry ports are channel-bound because the device changes their values after every selection. Reserving eight pages does not mean eight pages are readable on the first step: `page_indptr` must describe the live span, not the reservation size.

## 6. Trace the scoring epilogue

{{compare:ranking}}

Registering the epilogue stores a callback. The SDK traces and attaches the ETA program before the first submission, then reuses it on later submissions. The epilogue executes on the device after the model produces logits. See `Pass::attach_program` in `crates/inferlet/src/eta.rs`, and the corresponding `attach_program` / `attachProgram` methods in `packages/python/inferlet/src/inferlet/eta/bridge.py` and `packages/javascript/inferlet/src/eta/bridge.ts`. It reshapes logits to `[B, V]`, even at width 1, where the readout may otherwise be rank 1.

`log_softmax` converts each parent's logits into token log-probabilities. Broadcasting the previous score from `[B, 1]` to `[B, V]` adds it to every extension. Flattening before `top_k` is crucial: it selects `B` survivors globally, rather than selecting `B` tokens independently for each parent.

A flattened candidate index `i` encodes `parent = i / V` and `token = i % V`. Both outputs are device tensors. Rust overloads tensor operators, Python uses `//` for integer division, and JavaScript calls `.div()` and `.rem()` explicitly.

## 7. Inherit ancestry and append a cell

{{compare:ancestry}}

The next free positions are `fill + [0, 1, …, B-1]`. Gathering mask rows by `parent` gives each survivor its parent's history. Elementwise equality marks the survivor's new cell, and OR combines it with the inherited mask.

For example, if the selected parents are `[1, 1]`, both survivors inherit old lane 1's history, then receive different new cells. This is the fork. Unselected mask rows disappear from the live lanes; their old cells remain in the pool. This is the prune.

The new cell is prepared for the next forward, which will write the selected token's KV. `pos` tracks sequence depth, while `fill` tracks the next free position in the logical pool. With several beams those values grow at different rates.

## 8. Prepare the next forward's page geometry

{{compare:geometry}}

Gathering the logical page indexes by `wpos / page_t` selects each next write page. Remainder computes its offset. The device then advances `fill` by `B`, advances each sequence position by one, and feeds the selected tokens and scores back into their channels.

`page_count = ceil(filled / page_t)` determines each lane's live page-list length. The page IDs are repeated at this live stride, and `page_indptr` becomes `[0, page_count, 2 × page_count, …]`. The `pages` channel keeps its larger capacity; the engine reads the active entries described by the boundaries.

The span includes cells from other branches, but the ancestry mask excludes them from attention. The pool is shared without making unrelated histories visible to each other.

## 9. Publish host outputs

{{compare:publish}}

The epilogue publishes one row per output channel, then re-emits the pool IDs for its next execution. The greedy diagnostic uses raw logits, independently of cumulative-score addition and global top-B selection.

In the mathematical width-1 recurrence, adding a scalar score and applying log-softmax preserve the maximizing token set, so beam search reduces to greedy decoding. The inferlet checks equality of the actual selected token IDs with the raw-logit argmax and fails on any disagreement. At larger widths, mismatches can occur; they are not guaranteed for every model or run. This counter compares lanes and is a diagnostic, not a measure of answer quality.

## 10. Reconstruct paths on the host

{{compare:hypotheses}}

The device retains KV history, while the host keeps token lists for eventual decoding. A surviving lane's token list must be copied from its **selected parent**, then extended. Simply appending the new token to the old list at the same lane index would lose the search ancestry.

Rust checks indices and returns errors. Python concatenation and JavaScript spread syntax construct fresh lists or arrays. The mismatch helper counts per-lane disagreements with the diagnostic argmax.

## 11. Drive the attention-only loop

{{compare:attention-loop}}

The host drains tokens, parents, scores, and greedy tokens for each execution, then reconstructs the surviving paths. Rust puts these reads directly in the `run_ahead` callback; Python and JavaScript factor them into `drain` helpers.

With attention-only state, all next-step geometry and search state already flow through device channels. `run_ahead` can keep executions in flight while the host consumes results. Rust's callback returns `ControlFlow::Continue`, while the other two return `True` or `true`.

Notice the actual wait boundaries: Rust and Python await channel reads; JavaScript's `takeHost()` blocks. The implementations express the same dependency with different host control flow.

## 12. Fork recurrent state on hybrid models

{{compare:hybrid-loop}}

A hybrid model also carries recurrent state that cannot be represented by the attention mask. After each execution, the host reads the parent indices, forks the corresponding recurrent working sets, and rebinds the forward pass before submitting the next step.

Two survivors with the same parent receive two forks of that parent's state. The resource contract describes these as copy-on-write children; a fork does not imply an immediate full physical copy (`crates/inferlet/wit/working-set.wit`). Rebinding by output lane alone would associate the next token with the wrong recurrent history. This dependency is why the hybrid path uses explicit step-by-step submission rather than the attention path's `run_ahead` loop.

Each example calls `pipeline.close()` after its loop. In the attention-only path, `run_ahead` also closes the pipeline once its submission budget is reached, which can happen before all outputs have been consumed. Closing ends submission while already-submitted forwards continue to drain; it does not cancel them. The helpers in `crates/inferlet/src/eta.rs` and the Python/JavaScript bridge files show this order.

## 13. Choose and decode the result

{{compare:result}}

The highest cumulative-score lane wins. The Python and JavaScript loops use `>=` so equal scores select the last maximal lane, matching Rust's `max_by` behavior for ordinary finite scores. The width-1 identity check rejects disagreement with greedy decoding.

The chosen token list excludes the starting BOS token. `model.decode` turns it into text, and the returned string appends width, step count, best score, and mismatch count. Rust additionally logs a summary to stderr.

The example README reports zero greedy mismatches at width 1 over 16 steps, and a higher cumulative score at width 4 in its recorded check. Those are upstream observations for that example, not measurements made by building this book.

## Build and run the complete programs

Clone **pie-project/pie** and check out the commit shown at the top of this chapter. These commands run from that repository root, with a Pie CLI from this revision and compatible model configuration available. The Rust output path below assumes the default examples-workspace target directory; a custom `CARGO_TARGET_DIR` changes it. The Python and JavaScript language components must be installed in the Pie environment; they bundle their SDKs and interpreters.

| Rust | Python | JavaScript |
| --- | --- | --- |
| Compile a Wasm component | Submit Python source | Submit a JavaScript module |
| `Cargo.toml` declares a `cdylib` and the local `inferlet` dependency | Exposes `async def main(input)` | Exports `main(input)` |
| Requires the Rust Wasm target | Requires the Python language component | Requires the JavaScript language component |

Rust, using `examples/beam-search/Cargo.toml`:

```bash
rustup target add wasm32-wasip2
cargo build --manifest-path examples/Cargo.toml --release --target wasm32-wasip2 -p beam-search
pie run examples/target/wasm32-wasip2/release/beam_search.wasm -- --beams 2 --max_tokens 16
```

Python:

```bash
pie run examples/beam-search-py/main.py -- --beams 2 --max_tokens 16
```

JavaScript, with the module metadata in `examples/beam-search-js/package.json`:

```bash
pie run examples/beam-search-js/index.js -- --beams 2 --max_tokens 16
```

There is no per-inferlet Python or JavaScript compilation step. The reusable language components are built separately; see the included SDK READMEs for their build and execution model. The Rust README also points to the upstream CUDA example suite. Its harness accepts a model-store name, Hugging Face repository ID, or `.zt` artifact path; the command below uses the Qwen3-0.6B ID shown in the suite's own usage example:

```bash
python tests/examples/run_all.py --engine cuda_native --model Qwen/Qwen3-0.6B
```

The suite requires its Python server/client dependencies and compatible language components; it covers many examples, not just beam search. `tests/examples/run_all.py` includes the curated and language-twin tests. The beam identity and width-3 exploration checks are in `tests/examples/test_curated.py`; `tests/examples/test_twins.py` compares the complete Rust/Python/JavaScript return strings at widths 1 and 3 over six steps. Those are source-level test expectations, not results of this chapter audit.

To inspect the behavior, run each program with `--beams 1 --max_tokens 16`: each should report `greedy_mismatches=0`. Then try width 4 and observe how selected paths and scores change. This is a useful comparison on the same model and engine; it does not establish bitwise equivalence across different backends or settle how equal-score candidates are ordered.

The inferlet owns the search policy, state layout, and stopping rule. The runtime supplies model execution, page resources, channels, and scheduling. The three languages let us construct that same mechanism with their own entry points, tensor syntax, and host-read semantics.


## Source map

All sources below are included at this chapter's pinned Pie commit. The mathematical recurrence, pool bound, and ancestry example are explanations derived from that code. The diagram is a reconstruction of its dataflow.

| Claims | Source |
| --- | --- |
| Defaults, fixed starting token, search graph, loop order, output and checks | `examples/beam-search/src/lib.rs`, `examples/beam-search-py/main.py`, `examples/beam-search-js/index.js` |
| Historical width-1 and width-4 observations | `examples/beam-search/README.md` |
| Tensor operator outputs and shapes | `crates/eta-dsl/src/value.rs` |
| Channel reads, logical page reservation and recurrent-state fork semantics | `crates/inferlet/wit/channel.wit`, `crates/inferlet/wit/working-set.wit` |
| Deferred tracing, submission windows and pipeline closure | `crates/inferlet/src/eta.rs`, `packages/python/inferlet/src/inferlet/eta/bridge.py`, `packages/javascript/inferlet/src/eta/bridge.ts` |
| Entry-point deserialization and script execution model | `crates/inferlet-macros/src/lib.rs`, `packages/python/inferlet/README.md`, `packages/javascript/inferlet/README.md` |
| Rust workspace, artifact naming and CLI argument conversion | `examples/Cargo.toml`, `examples/beam-search/Cargo.toml`, `src/ops/run.rs` |
| Suite invocation, model arguments and cross-language checks | `tests/examples/run_all.py`, `tests/examples/conftest.py`, `tests/examples/test_curated.py`, `tests/examples/test_twins.py` |
