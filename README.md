# Pie Tutorial

[Pie](https://github.com/pie-project/pie) (SOSP'25) 0.5 is about half a
million lines of code. This repo rebuilds it from scratch in about a thousand,
one chapter per branch. Each chapter builds on the previous one, adds one
feature, and explains in its README.

## Chapters

1. **Inferlets** ([`feat/inferlet`](../../tree/feat/inferlet)): the core. A
   wasm host that runs inferlets, a five-call contract (`wit/pie.wit`), a
   paged KV cache, a batcher, and a Qwen model on candle.
2. **KV working set** ([`feat/kv-working-set`](../../tree/feat/kv-working-set)):
   the KV cache becomes a resource with logical pages, like virtual memory.
   The inferlet never sees a physical page, and pages are freed when the
   handle is dropped.
3. **KV fork** ([`feat/kv-fork`](../../tree/feat/kv-fork)): `fork()` shares
   pages copy-on-write, like the OS `fork()`. Beam search becomes a fork and
   a loop in the inferlet.
4. **Async forward** ([`feat/async-forward`](../../tree/feat/async-forward)):
   `forward` returns a pending result, like async I/O. An inferlet submits
   all its sequences before waiting, so its beams run in one model step.
5. **Planner** ([`feat/planner`](../../tree/feat/planner)): when KV pages run
   out, inferlets wait instead of failing. If all of them wait, the planner
   evicts the youngest, like the OS OOM killer, and restarts it later.
6. **Speculative decoding** ([`feat/spec-decoding`](../../tree/feat/spec-decoding)):
   `forward` returns a distribution after any chosen tokens, not just the
   last. An inferlet checks several guessed tokens in one forward and rolls
   back the wrong ones.
7. **Sampling** ([`feat/sampling`](../../tree/feat/sampling)): temperature,
   top-p and min-p written in the inferlet, with no engine change. Several
   samples of one prompt share its pages and decode in one batch.
8. **KV discard** ([`feat/kv-discard`](../../tree/feat/kv-discard)): a
   working set can drop pages it no longer needs. Keeping the first page and
   a sliding window, an inferlet generates far more tokens than the pool holds.
9. **Prefix cache** ([`feat/prefix-cache`](../../tree/feat/prefix-cache)):
   a working set can be published under a key and opened by other inferlets,
   so a shared system prompt is computed once.
10. **Constrained decoding** ([`feat/constrained-decoding`](../../tree/feat/constrained-decoding)):
    `forward` can restrict which tokens may come next. The inferlet decides
    at each step, so the answer is always one of a list of choices.
11. **Chat** ([`feat/chat`](../../tree/feat/chat)): the host gives the
    model's chat format, so an inferlet holds a conversation without knowing
    how the model spells its turns. The model's thinking is kept out of the
    history.
12. **Server and client** ([`feat/server`](../../tree/feat/server)): `pie
    --serve` runs inferlets that clients send over the network and talks to
    them while they run. Answers stream back as they are generated.
13. **Scheduler** ([`feat/scheduler`](../../tree/feat/scheduler)): each
    model step runs at most a budget of tokens, short requests first, and a
    long prompt is split across steps. A long prompt no longer stops
    everyone else's decoding for seconds.
14. **Batched attention** ([`feat/attention`](../../tree/feat/attention)):
    attention is planned once per step, and all decoding sequences are
    attended together. Batched work gets 18-25% faster, with the same
    results.
15. **Reference layout** ([`feat/reference-layout`](../../tree/feat/reference-layout)):
    no new code. The same code moved into the layout of pie-project/pie,
    with the WIT split into one interface per file, and a list of what the
    reference adds on top.
16. **Engine and worker** ([`feat/engine-worker`](../../tree/feat/engine-worker)):
    the runtime talks to the model through an `Engine` trait only, and a
    worker loads the model and builds the runtime on top.
17. **Protocol and programs** ([`feat/protocol`](../../tree/feat/protocol)):
    clients talk to a gateway over a shared, versioned protocol on
    websockets, install programs once, and start them by name.
