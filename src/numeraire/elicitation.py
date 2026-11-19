"""Elicitation checks: a known-target probe, UP/DOWN answer tokens, and content-free calibration.

Every builder returns chat messages whose user content is a statement block
followed by a question; only the question (or, for calibration, the
statement block) differs from the forecasting prompt in token_inference.
"""

from __future__ import annotations
from .prompts import SYSTEM, TARGET
from .transforms import render

QUARTER_TARGET = (
    "whether basic EPS in the next fiscal quarter increases over its comparable same quarter one year earlier, "
    "on the share basis reported in that outcome filing"
)
KNOWN = "State whether basic earnings per share in the Current column above is higher than in the Prior column"
CONTENT_FREE = {"na": "N/A", "empty": "Item | Prior | Current", "mask": "[MASK]"}


def target(statement):
    return TARGET if statement.provenance.get("frequency", "annual") == "annual" else QUARTER_TARGET


def question(kind, task, swap, statement):
    """kind: 'ab' (letters, swap exchanges meanings) or 'word' (UP/DOWN, swap exchanges the order named)."""
    if task == "forecast":
        lead = f"Predict {target(statement)}."
    elif task == "known":
        lead = KNOWN + ". UP means Current is higher; DOWN means it is not."
    else:
        raise ValueError("Unknown task")
    if kind == "ab":
        d = ("DOWN", "UP") if swap else ("UP", "DOWN")
        return f"{lead}\nA = {d[0]}. B = {d[1]}.\nReply with exactly one letter, A or B. Do not explain your answer."
    if kind == "word":
        w = ("DOWN", "UP") if swap else ("UP", "DOWN")
        return f"{lead}\nReply with exactly one word, {w[0]} or {w[1]}. Do not explain your answer."
    raise ValueError("Unknown kind")


def messages(statement, arm, kind, task, swap, seed=2026, content_free=None):
    block = CONTENT_FREE[content_free] if content_free else render(statement, arm, seed=seed).text
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": block + "\n\n" + question(kind, task, swap, statement)},
    ]


def up_logit(kind, swap, logp):
    """Log odds of UP from one context."""
    if kind == "ab":
        return (logp["B"] - logp["A"]) if swap else (logp["A"] - logp["B"])
    return logp["UP"] - logp["DOWN"]
