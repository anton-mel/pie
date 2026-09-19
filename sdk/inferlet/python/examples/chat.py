# NEW
"""A conversation: each argument is one user message.

The Python version of `tests/inferlets/chat`.
"""

import inferlet
from inferlet import Context, Sampler


def main(args):
    turns = args or ["What is the capital of France?", "What is its most famous tower?"]
    ctx = Context()
    ctx.system("You are a helpful assistant. Answer in one short sentence.")
    # Qwen3's recommended settings for chat.
    sampler = Sampler(0.6, 0.95, seed=1)
    log = []
    for message in turns:
        ctx.user(message)
        reply = ctx.reply(1024, 64, sampler)
        thought = len(reply.thinking.split()) if reply.thinking else 0
        log.append(f"user: {message}\n    assistant: {reply.text}  (thought for {thought} words)")
    return "\n    ".join(log)


Run = inferlet.export(main)
