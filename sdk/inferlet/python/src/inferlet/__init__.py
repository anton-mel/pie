# NEW
"""The library Python inferlets import: the Rust SDK (`crates/inferlet`),
written again over the same WIT interfaces.

An inferlet is a module with a `main(args) -> str`, exported with
`Run = inferlet.export(main)`. componentize-py bundles it, this package and
a Python interpreter into one component that `pie run` loads like any other.
"""

from componentize_py_types import Err
from wit_world import exports
from wit_world.imports import (
    chat,
    forward,
    grammar,
    model,
    pipeline,
    reasoning,
    session,
    tokenizer,
    tools,
    working_set,
)

__all__ = [
    "Context", "Reply", "Sampler", "greedy", "export",
    "chat", "grammar", "model", "reasoning", "session", "tokenizer", "tools",
]


def export(main):
    """The class componentize-py looks for as `Run`: calls `main(args)`, and
    turns any exception into the error `run` returns."""

    class Run(exports.Run):
        def run(self, args):
            try:
                return main(args)
            except Err:
                raise
            except Exception as e:
                raise Err(f"{type(e).__name__}: {_message(e)}")

    return Run


def _message(e):
    # Host errors arrive as `Err` with the string in `value`.
    return e.value if isinstance(e, Err) else str(e)


class Context:
    """Tokens written so far, the pages holding their KV, and the tokens
    waiting for the next forward: `Context` in the Rust SDK."""

    def __init__(self):
        self.tokens = []
        self.pending = []
        self.kv = working_set.KvWorkingSet()
        self.pipeline = pipeline.Pipeline()
        self.page_size = model.kv_page_size()
        self.pos = 0
        self.system_prompt = None

    def fork(self):
        child = Context.__new__(Context)
        child.tokens = list(self.tokens)
        child.pending = list(self.pending)
        child.kv = self.kv.fork()
        child.pipeline = self.pipeline
        child.page_size = self.page_size
        child.pos = self.pos
        child.system_prompt = self.system_prompt
        return child

    def fill(self, text):
        self.pending += tokenizer.tokenize(text)

    def fill_tokens(self, tokens):
        self.pending += tokens

    def forward(self, top_k):
        """One model step: the distribution after the last pending token."""
        return self.submit([len(self.pending) - 1], top_k).wait()[0]

    def submit(self, outputs, top_k, allowed=None):
        if not self.pending:
            raise ValueError("nothing to forward")
        length = len(self.tokens) + len(self.pending)
        need = -(-length // self.page_size)
        have = self.kv.page_len()
        if need > have:
            self.kv.reserve(need - have)
        positions = list(range(self.pos, self.pos + len(self.pending)))
        self.pos += len(self.pending)
        pending = forward.forward(
            self.pipeline, self.kv, length, self.pending, positions, outputs, allowed, None, top_k
        )
        self.tokens += self.pending
        self.pending = []
        return pending

    def rollback(self, n):
        n = min(n, len(self.tokens))
        del self.tokens[len(self.tokens) - n :]
        self.pos -= n

    def generate(self, max_tokens, top_k, sample):
        eos = tokenizer.eos_tokens()
        out = []
        while len(out) < max_tokens:
            nxt = sample(self.forward(top_k))
            if nxt in eos:
                break
            out.append(nxt)
            self.pending.append(nxt)
        return tokenizer.detokenize(out)

    # Chat, as in the Rust SDK: the system prompt waits for the first user
    # message, and thinking is removed from the history after each reply.

    def system(self, message):
        self.system_prompt = message

    def equip(self, tool_list):
        offer = tools.equip(tool_list)
        system = self.system_prompt or ""
        self.system_prompt = system + offer if system else offer.lstrip()

    def user(self, message):
        self._start()
        if self.system_prompt is not None:
            self.fill_tokens(chat.system_user(self.system_prompt, message))
            self.system_prompt = None
        else:
            self.fill_tokens(chat.user(message))

    def _start(self):
        if not self.tokens and not self.pending:
            self.fill_tokens(chat.prefix())

    def reply(self, max_tokens, top_k, sample):
        self._start()
        if self.system_prompt is not None:
            self.fill_tokens(chat.system(self.system_prompt))
            self.system_prompt = None
        self.fill_tokens(chat.cue())
        stop = chat.stop_tokens()
        out = []
        while len(out) < max_tokens:
            nxt = sample(self.forward(top_k))
            if nxt in stop:
                break
            out.append(nxt)
            self.fill_tokens([nxt])

        text = tokenizer.detokenize(out)
        thinking, sep, answer = text.partition("</think>")
        if sep:
            reply = Reply(answer.strip(), thinking.replace("<think>", "").strip())
            forwarded = len(out) - len(self.pending)
            self.pending = []
            self.rollback(forwarded)
            self.fill(reply.text)
        else:
            reply = Reply(text.strip(), None)
        self.fill_tokens(chat.seal())
        return reply


class Reply:
    def __init__(self, text, thinking):
        self.text = text
        self.thinking = thinking


def greedy(d):
    """Always the most likely token."""
    return d.ids[0]


class Sampler:
    """Temperature, top-p and min-p sampling over a top-k distribution, with
    the same splitmix64 generator as the Rust SDK."""

    def __init__(self, temperature, top_p, min_p=0.0, seed=None):
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.rng = seed if seed is not None else int.from_bytes(__import__("os").urandom(8), "little")

    def __call__(self, d):
        if self.temperature <= 0:
            return d.ids[0]
        w = [p ** (1 / self.temperature) for p in d.probs]
        total = sum(w)
        w = [x / total for x in w]
        keep, mass = 0, 0.0
        while keep < len(w) and (keep == 0 or (mass < self.top_p and w[keep] >= self.min_p * w[0])):
            mass += w[keep]
            keep += 1
        u = self._next() * mass
        for i in range(keep):
            u -= w[i]
            if u <= 0:
                return d.ids[i]
        return d.ids[keep - 1]

    def _next(self):
        m = (1 << 64) - 1
        self.rng = (self.rng + 0x9E3779B97F4A7C15) & m
        z = self.rng
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        return ((z ^ (z >> 31)) >> 40) / (1 << 24)
