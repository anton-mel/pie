# Pie Tutorial

[Pie](https://github.com/pie-project/pie) (SOSP'25) 0.5 is about half a
million lines of code. This repo rebuilds it from scratch in about a thousand,
one chapter per branch. Each chapter builds on the previous one, adds one
feature, and explains in its README.

> [!NOTE]
> Follow chapter's **Read Order**.

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
