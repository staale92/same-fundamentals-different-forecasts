from __future__ import annotations
import hashlib
import json
import math
import re
from .prompts import QUARTERLY_TARGET, REASONING, SYSTEM, TARGET
from .transforms import render


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def class_messages(statement, arm, swap=False, seed=2026, system=None):
    """Scored A/B prompt; ``system`` replaces the fixed system message (instruction ablation)."""
    directions = ("DOWN", "UP") if swap else ("UP", "DOWN")
    target = TARGET if statement.provenance.get("frequency", "annual") == "annual" else QUARTERLY_TARGET
    question = (
        f"Predict {target}.\n"
        f"A = {directions[0]}. B = {directions[1]}.\n"
        "Reply with exactly one letter, A or B. Do not explain your answer."
    )
    return [
        {"role": "system", "content": SYSTEM if system is None else system},
        {"role": "user", "content": render(statement, arm, seed=seed).text + "\n\n" + question},
    ]


def reasoning_messages(statement, arm, seed=2026, system=None):
    """Generated-reasoning prompt: the rendered statement, then the reasoning request with the scored target."""
    target = TARGET if statement.provenance.get("frequency", "annual") == "annual" else QUARTERLY_TARGET
    return [
        {"role": "system", "content": SYSTEM if system is None else system},
        {"role": "user", "content": render(statement, arm, seed=seed).text + "\n\n" + REASONING.replace(TARGET, target)},
    ]


def parse_reasoning(content):
    """Forecast from a generated answer: the last JSON object with probability_percent, else a key-value match."""
    for match in reversed(re.findall(r"\{[^{}]*\}", content or "")):
        try:
            obj = json.loads(match)
        except (ValueError, TypeError):
            continue
        if "probability_percent" in obj:
            return float(obj["probability_percent"]) / 100, "json"
    m = re.search(r'probability_percent"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', content or "")
    return (float(m.group(1)) / 100, "regex") if m else (None, "failed")


def conditional_probability(logp_a, logp_b, swap=False):
    if not math.isfinite(logp_a) or not math.isfinite(logp_b):
        raise ValueError("Nonfinite class score")
    m = max(logp_a, logp_b)
    pa = math.exp(logp_a - m) / (math.exp(logp_a - m) + math.exp(logp_b - m))
    return 1 - pa if swap else pa


def completion_strings(
    tokenizer, messages, response_mode="template", template_kwargs=None, answers=("A", "B")
):
    """Locate the answer channel explicitly for models with recipient headers.

    ``final`` supplies the final assistant header and scores the answer
    directly, without generating a reasoning trace.
    """
    kwargs = template_kwargs or {}
    if response_mode == "template":
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        return [prompt + s for s in answers]
    if response_mode != "final":
        raise ValueError("Unknown response mode")
    return [
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": s}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            **kwargs,
        )
        for s in answers
    ]
