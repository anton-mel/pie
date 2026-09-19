# NEW
"""A model that calls tools the inferlet runs: the Python version of
`tests/inferlets/agent`. The decoders are host resources, used from Python
exactly as from Rust."""

import json

import inferlet
from inferlet import Context, chat, greedy, reasoning, tokenizer, tools

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CALCULATOR = {"type": "function", "function": {"name": "calculate", "description": "Compute a * b, a / b, a + b or a - b", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}

CITIES = {"Paris": "18°C, sunny", "London": "12°C, light rain", "Tokyo": "24°C, clear"}


def run_tool(call):
    """The tools themselves: plain inferlet code."""
    args = json.loads(call.arguments_json or "{}")
    if call.name == "get_weather":
        city = args.get("city", "")
        return CITIES.get(city, f"no weather data for {city}")
    if call.name == "calculate":
        e = args.get("expression", "").replace(" ", "")
        for op, f in (("*", float.__mul__), ("/", float.__truediv__), ("+", float.__add__), ("-", float.__sub__)):
            a, sep, b = e[1:].partition(op)
            if sep:
                try:
                    v = f(float(e[0] + a), float(b))
                    return str(int(v)) if v.is_integer() else str(v)
                except ValueError:
                    break
        return f"cannot compute {e}"
    return f"no tool named {call.name}"


def main(args):
    question = args[0] if args else "What is the weather in Paris, and what is 17 * 23?"
    ctx = Context()
    ctx.system("You are a helpful assistant. Use the tools when they help.")
    ctx.equip([json.dumps(WEATHER), json.dumps(CALCULATOR)])
    ctx.user(question)
    stop = chat.stop_tokens()
    log = []

    for round in range(1, 7):
        ctx.fill_tokens(chat.cue())
        thinking, calling = reasoning.Decoder(), tools.Decoder()
        out, calls, thought = [], [], 0
        while len(out) < 1024:
            nxt = greedy(ctx.forward(1))
            if nxt in stop:
                break
            out.append(nxt)
            ctx.fill_tokens([nxt])
            event = thinking.feed([nxt])
            if isinstance(event, reasoning.Event_Complete):
                thought = len(event.value.split())
            event = calling.feed([nxt])
            if isinstance(event, tools.Event_Call):
                calls.append(event.value)
        ctx.fill_tokens(chat.seal())

        if not calls:
            answer = tokenizer.detokenize(out).rsplit("</think>", 1)[-1].strip()
            log.append(f"round {round}: thought {thought} words, answered: {answer}")
            break
        results = []
        for call in calls:
            result = run_tool(call)
            log.append(f"round {round}: thought {thought} words, called {call.name}({call.arguments_json}) -> {result}")
            results.append(result)
        ctx.fill_tokens(tools.answer(results))
    return "\n    ".join(log)


Run = inferlet.export(main)
