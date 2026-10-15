<div class="book-contents">

# Pie: A *programmable* LLM serving system.

This book builds an inference runtime from scratch. It starts with a small Wasm host that runs inferlets, then adds KV memory management, generation techniques, runtime services, and GPU execution. Thirty chapters follow the runtime's construction and include the code for each stage. An extra chapter, [Inspecting an Inferlet](chapters/31-inspecting-an-inferlet.html), dissects the reference beam-search program in Rust, Python, and JavaScript side by side.

If you’re new to Pie, start with [Inferlets](chapters/01-inferlet.html). If you’re looking for a particular feature, see the [chapter list](#all-chapters) below.

<section class="reading-path" aria-labelledby="recommended-path">

## Recommended reading {#recommended-path}

For an introduction to Pie’s internals, read these thirteen chapters. They cover how inferlets control generation, manage KV memory, and schedule model execution.

- **Foundations, chapters 1–5:** [Inferlets](chapters/01-inferlet.html), [KV working set](chapters/02-kv-working-set.html), [KV fork](chapters/03-kv-fork.html), [Async forward](chapters/04-async-forward.html), and [Planner](chapters/05-planner.html).
- **Generation and reuse, chapters 6–11:** [Speculative decoding](chapters/06-spec-decoding.html), [Sampling](chapters/07-sampling.html), [KV discard](chapters/08-kv-discard.html), [Prefix cache](chapters/09-prefix-cache.html), [Constrained decoding](chapters/10-constrained-decoding.html), and [Chat](chapters/11-chat.html).
- **Scheduling and shared prefixes:** [Scheduler, chapter 13](chapters/13-scheduler.html) and [Prefix trie, chapter 20](chapters/20-prefix-trie.html).

If you’re building along, follow all chapters in order: each implementation builds on the preceding chapter.

</section>

## All chapters {#all-chapters}

{{chapter_index}}

## How to use this book

Each chapter describes a separate, pinned Git commit. Click a source filename to read that version in the source panel; on a small screen, the panel opens over the chapter. Directory links open an entry file, and the file selector lets you browse the rest of that chapter. The inspection chapter includes pinned reference sources and its own build instructions; it is a standalone walkthrough rather than the next runtime implementation.

To run a chapter, clone this repository and check out the commit shown on its page. The chapter commands require Rust and the `wasm32-wasip2` target; the macOS examples use Metal and download model weights. Source files are reading examples, not standalone Rust Playground programs.

Descriptions of missing reference features refer to the tutorial stage where they appear; later chapters may add them. The current [Pie implementation](https://github.com/pie-project/pie) may have evolved beyond these tutorial snapshots.

</div>
