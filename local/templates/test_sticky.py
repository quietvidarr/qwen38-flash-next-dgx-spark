#!/usr/bin/env python3
"""CPU check of a sticky-reasoning template override against the checkpoint's stock template.

    PYTHONPATH=<TensorFold fork>/src python3 local/templates/test_sticky.py <model snapshot dir> [override.jinja]

Renders as the CUDA server does (``ChatTemplate`` + ``server.sticky_reasoning``; GLM through ``ThinkingOffTemplate``)
and tokenizes with the checkpoint's tokenizer.json. Checks: (a) no kwargs = stock, byte for byte; (b) a note at
message k leaves every token before message k unchanged; (c) the next turn starts with this turn's history, across
an effort switch; (d) off/medium and prefix == current level = stock; (e) bad kwargs are dropped, never raised.
Needs jinja2 and tokenizers. Exit 0 = all passed."""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path(sys.argv[1])
OVERRIDE = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "chat_template.jinja"

from tokenizers import Tokenizer  # noqa: E402

from tensorfold.cuda.chat_template import ChatTemplate  # noqa: E402
from tensorfold.server.sticky_reasoning import sticky_reasoning  # noqa: E402

GLM = "[gMASK]" in OVERRIDE.read_text()


def load(override: Path | None):
    os.environ.pop("TF_CHAT_TEMPLATE", None)
    if override is not None:
        os.environ["TF_CHAT_TEMPLATE"] = str(override)
    template = ChatTemplate(MODEL)
    os.environ.pop("TF_CHAT_TEMPLATE", None)
    if GLM:
        from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate

        template = ThinkingOffTemplate(template, clear=False)
    return template


STOCK, OVER = load(None), load(OVERRIDE)
TOK = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
LEVELS = ("low", "high", "max") if GLM else ("low", "medium", "xhigh")
assert OVER.efforts == STOCK.efforts == frozenset(LEVELS), (OVER.efforts, STOCK.efforts)
assert getattr(OVER, "inner", OVER).late_system == getattr(STOCK, "inner", STOCK).late_system

TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    for n, d in (("ls", "List a directory"), ("read", "Read a file"), ("wc", "Count lines"))]


def call(i, name, path):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": f'{{"path": "{path}"}}'}}


TURN = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "List the files, then read the README."},
    {"role": "assistant", "content": "", "reasoning_content": "ls then read.",
     "tool_calls": [call(1, "ls", "."), call(2, "read", "README.md")]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nREADME.md"},
    {"role": "tool", "tool_call_id": "c2", "content": "# Readme\nhello"},
    {"role": "assistant", "content": "Two files; the README says hello.", "reasoning_content": "Summarize."},
    {"role": "user", "content": "Count the lines of a.txt."},
    {"role": "assistant", "content": "", "reasoning_content": "wc.", "tool_calls": [call(3, "wc", "a.txt")]},
    {"role": "tool", "tool_call_id": "c3", "content": "42"},
]
NEXT = [{"role": "assistant", "content": "a.txt has 42 lines.", "reasoning_content": "Answer."},
        {"role": "user", "content": "Thanks. Now summarize everything."}]
VARIANTS = {
    "agent": TURN,
    "no-system": TURN[1:],
    "two-leading": [TURN[0], {"role": "developer", "content": "Be terse."}, *TURN[1:]],
    "late-system": [*TURN[:6], {"role": "system", "content": "Mind the time."}, *TURN[6:]],
}
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
        print("FAIL", what)


def render(template, messages, *, thinking=True, effort=None, sticky=None, gen=True, tools=TOOLS):
    """Text as cuda/server.py _prepare renders it."""
    levels = template.efforts
    messages, notes = sticky_reasoning(messages, sticky, levels)
    extra = {"add_generation_prompt": gen}
    if thinking and effort:
        extra["reasoning_effort"] = effort
    extra.update(notes)
    return template.render(messages, tools=tools, enable_thinking=thinking, extra=extra)


def ids(text):
    return TOK.encode(text, add_special_tokens=False).ids


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def stock_combos():
    for (name, msgs), thinking, effort, gen, tools in itertools.product(
            VARIANTS.items(), (True, False), (None, *LEVELS), (True, False), (TOOLS, None)):
        if not thinking and effort:
            continue
        yield name, msgs, thinking, effort, gen, tools


# (a) no kwargs: stock bytes; also with a client's own reasoning_change keys (removed)
count = 0
for name, msgs, thinking, effort, gen, tools in stock_combos():
    want = render(STOCK, msgs, thinking=thinking, effort=effort, gen=gen, tools=tools)
    got = render(OVER, msgs, thinking=thinking, effort=effort, gen=gen, tools=tools)
    check(got == want, f"(a) {name} thinking={thinking} effort={effort} gen={gen} tools={bool(tools)}")
    tagged = render(OVER, [{**m, "reasoning_change": "low"} for m in msgs], thinking=thinking, effort=effort,
                    gen=gen, tools=tools)
    check(tagged == want, f"(a) client reasoning_change keys stripped {name}")
    count += 1
print(f"(a) {count} stock combinations identical")

# (b) a note at message k keeps every token before message k
count = 0
for name, msgs in VARIANTS.items():
    for k, level, thinking in itertools.product(range(1, len(msgs) + 1), LEVELS + ("off",), (True, False)):
        base = dict(thinking=thinking, effort="low" if thinking else None)
        without = render(OVER, msgs, sticky={"reasoning_prefix": "low", "reasoning_changes": []}, **base)
        with_note = render(OVER, msgs, sticky={"reasoning_prefix": "low", "reasoning_changes": [[k, level]]}, **base)
        start = k              # GLM renders a run of tool results as one block: its note goes before the run
        while GLM and 0 < start < len(msgs) and msgs[start]["role"] == msgs[start - 1]["role"] == "tool":
            start -= 1
        try:
            head = render(OVER, msgs[:start], sticky={"reasoning_prefix": "low"}, gen=False, **base)
        except Exception:  # noqa: BLE001 - a head the template refuses (no user query yet)
            head = render(OVER, msgs[:start] + [{"role": "user", "content": "x"}], sticky={"reasoning_prefix": "low"},
                          gen=False, **base)
            head = head[:head.rindex("<|user|>" if GLM else "<|im_start|>user")]    # up to the stand-in user
        w, n, h = ids(without), ids(with_note), ids(head)
        stable = lcp(h, w)
        check(lcp(n, w) >= stable, f"(b) {name} k={k} {level} thinking={thinking}: {lcp(n, w)} < {stable}")
        if not (GLM and level == "off"):                # GLM's off note is empty by design
            check(n != w, f"(b) {name} k={k} {level}: no note")
        count += 1
print(f"(b) {count} single-change renders keep their prefix")

# (c) history byte-stable across turns, with and without an effort switch on the new turn
count = 0
for name, msgs in VARIANTS.items():
    n = len(msgs)
    for first, second, think2 in itertools.product(("low", LEVELS[-1]), LEVELS + ("off",), (True, False)):
        changes = [[3, LEVELS[1]]]                      # an old switch inside the agent loop
        turn = dict(sticky={"reasoning_prefix": first, "reasoning_changes": changes}, effort=LEVELS[1])
        prompt = ids(render(OVER, msgs, **turn))
        history = ids(render(OVER, msgs, gen=False, **turn))
        check(prompt[:len(history)] == history, f"(c) {name}: history not a prefix of its prompt")
        later = msgs + NEXT
        switched = changes + ([[n + 1, second]] if second != LEVELS[1] else [])
        nxt = ids(render(OVER, later, thinking=think2 and second != "off", effort=second if second != "off" else None,
                         sticky={"reasoning_prefix": first, "reasoning_changes": switched}))
        check(nxt[:len(history)] == history, f"(c) {name} {first}->{second} think={think2}: history changed "
                                             f"at token {lcp(nxt, history)} of {len(history)}")
        check(nxt[:len(prompt)] == prompt, f"(c) {name}: last prompt not a prefix of the next")
        # a tail switch, then the same change at the appended message's index
        tail = ids(render(OVER, msgs, sticky={"reasoning_prefix": first, "reasoning_changes": changes + [[n, second]]},
                          effort=LEVELS[1]))
        moved = ids(render(OVER, later, sticky={"reasoning_prefix": first, "reasoning_changes": changes + [[n, second]]},
                           effort=LEVELS[1]))
        stable = lcp(tail, moved)
        check(moved[:len(history)] == history, f"(c) {name}: tail switch changed the history")
        gen_len = len(ids(render(OVER, [{"role": "user", "content": "x"}], tools=None))) - \
            len(ids(render(OVER, [{"role": "user", "content": "x"}], tools=None, gen=False)))
        check(stable >= len(tail) - gen_len, f"(c) {name}: tail note not kept when it moves onto the next message "
                                             f"({stable} of {len(tail)})")
        count += 1
print(f"(c) {count} next turns keep the history")

# (d) prefix == this reply's level, no changes: the stock render of that level
count = 0
for name, msgs in VARIANTS.items():
    for level in LEVELS:
        want = render(STOCK, msgs, effort=level)
        got = render(OVER, msgs, effort=level, sticky={"reasoning_prefix": level, "reasoning_changes": []})
        check(got == want, f"(d) {name} prefix={level} != stock {level}")
        count += 1
    want = render(STOCK, msgs, thinking=False)
    got = render(OVER, msgs, thinking=False, sticky={"reasoning_prefix": "off", "reasoning_changes": []})
    check(got == want, f"(d) {name} prefix=off != stock thinking off")
    count += 1
off_note = render(OVER, TURN, sticky={"reasoning_prefix": "low", "reasoning_changes": [[6, "off"]]}, thinking=False)
mid_note = render(OVER, TURN, sticky={"reasoning_prefix": "off", "reasoning_changes": [[6, LEVELS[1]]]})
if GLM:
    check(off_note == render(OVER, TURN, sticky={"reasoning_prefix": "low"}, thinking=False), "(d) GLM off note not empty")
    check(off_note.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low"), "(d) GLM low prefix line")
    check("<|system|>Reasoning Effort: High<|user|>Count" in mid_note, "(d) GLM high note")
    check(not render(OVER, TURN, sticky={"reasoning_prefix": "off"}).startswith("[gMASK]<sop><|system|>Reasoning"),
          "(d) GLM off prefix has a line")
    check(off_note.endswith("<|assistant|><think></think>"), "(d) GLM thinking-off generation prompt")
else:
    check("<|im_start|>system\nThinking is off.<|im_end|>\n<|im_start|>user\nCount" in off_note, "(d) Qwen off note")
    check("<|im_start|>system\nReasoning effort is set to medium.<|im_end|>\n<|im_start|>user\nCount" in mid_note,
          "(d) Qwen medium note")
    check(off_note.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "(d) Qwen thinking-off generation prompt")
print(f"(d) {count} prefix-only renders equal stock; off/medium notes as documented")

# (e) bad kwargs: dropped, never raised
good = render(OVER, TURN, sticky={"reasoning_prefix": "low", "reasoning_changes": [[6, LEVELS[-1]]]})
plain = render(OVER, TURN, sticky={"reasoning_prefix": "low", "reasoning_changes": []})
bad_changes = [[99, "low"], [-1, "low"], ["6", "low"], [True, "low"], [6, "bogus"], [6, None], [6], "x", None, [1, 2, 3]]
for bad in bad_changes:
    try:
        got = render(OVER, TURN, sticky={"reasoning_prefix": "low", "reasoning_changes": [bad, [6, LEVELS[-1]]]})
        check(got == good, f"(e) bad change {bad!r} not ignored")
    except Exception as exc:  # noqa: BLE001
        check(False, f"(e) bad change {bad!r} raised {exc}")
for changes in ("nope", 5, {"6": "low"}, None):
    got = render(OVER, TURN, sticky={"reasoning_prefix": "low", "reasoning_changes": changes})
    check(got == plain, f"(e) reasoning_changes={changes!r} not ignored")
for prefix in ("bogus", 3, None, ""):
    got = render(OVER, TURN, effort="low", sticky={"reasoning_prefix": prefix, "reasoning_changes": []})
    check(got == render(STOCK, TURN, effort="low"), f"(e) prefix={prefix!r} not ignored")
# the template alone (no TensorFold validation) ignores levels it does not name
raw = OVER.render([{**m, "reasoning_change": "bogus"} if i == 6 else m for i, m in enumerate(TURN)], tools=TOOLS,
                  enable_thinking=True, extra={"reasoning_prefix": "bogus", "reasoning_tail": 7})
check(raw == STOCK.render(TURN, tools=TOOLS, enable_thinking=True), "(e) template-level bad levels not ignored")
print("(e) bad kwargs ignored")

print("FAILED" if failures else "ALL PASSED", f"({len(failures)} failures)" if failures else "")
sys.exit(1 if failures else 0)
