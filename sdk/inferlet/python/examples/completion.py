# NEW
"""Greedy text completion: the Python version of `tests/inferlets/text-completion`."""

import inferlet
from inferlet import Context, greedy


def main(args):
    ctx = Context()
    ctx.fill(args[0] if args else "The capital of France is")
    return ctx.generate(32, 1, greedy)


Run = inferlet.export(main)
