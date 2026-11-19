from __future__ import annotations
import json
import math
import re
from decimal import Decimal
from .transforms import render

FIELDS = {
    "revenue_prior": ("revenue", 0),
    "revenue_current": ("revenue", 1),
    "assets_prior": ("assets", 0),
    "assets_current": ("assets", 1),
    "eps_prior": ("eps_basic", 0),
    "eps_current": ("eps_basic", 1),
}


def inversion_messages(statement, arm):
    if statement.currency != "USD" or arm != "NC":
        raise ValueError("This probe restores USD from a disclosed JPY conversion")
    ask = (
        "Convert the six requested values back to the original USD using the disclosed conversion rate. "
        "For revenue and assets, return whole USD amounts, not thousands or millions. For EPS, return USD per share. "
        "Return exactly one JSON object with these six numeric fields: "
        + ", ".join(FIELDS)
        + ". Do not add prose."
    )
    return [
        {
            "role": "system",
            "content": "Perform only the requested unit conversion using the supplied numbers and conversion rate.",
        },
        {"role": "user", "content": render(statement, arm).text + "\n\n" + ask},
    ]


def score_inversion(raw, statement):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        answer = json.loads(text)
        if not isinstance(answer, dict) or set(answer) != set(FIELDS):
            raise ValueError("fields")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in answer.values()
        ):
            raise ValueError("numeric")
    except (ValueError, TypeError):
        return {"parse_success": False, "all_correct": False, "fields": []}
    rows = []
    for field, (item, j) in FIELDS.items():
        target = statement.values[item][j]
        actual = Decimal(str(answer[field]))
        error = abs(actual - target)
        # Relative tolerance with a half-cent floor for per-share values.
        tolerance = max(Decimal(".005"), abs(target) * Decimal("0.000001"))
        rows.append(
            {
                "field": field,
                "target": str(target),
                "actual": str(actual),
                "absolute_error": str(error),
                "relative_error": float(error / abs(target)) if target else None,
                "correct": bool(actual.is_finite() and error <= tolerance),
                "tolerance": str(tolerance),
            }
        )
    return {"parse_success": True, "all_correct": all(r["correct"] for r in rows), "fields": rows}


TOLERANCES = (1e-6, 1e-3, 1e-2, 5e-2)


def score_inversion_lenient(raw, statement, tolerances=TOLERANCES):
    """Post hoc scoring of approximate level recovery.

    Takes the first JSON object in the response, removes digit-group commas
    inside numbers, and reports relative errors with a tolerance grid. The
    strict score in score_inversion remains the primary record.
    """
    match = re.search(r"\{[^{}]*\}", raw, re.S)
    if not match:
        return {"parse_success": False, "fields": []}
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", match.group(0))
    try:
        answer = json.loads(text)
        values = {f: Decimal(str(answer[f])) for f in FIELDS}
    except (ValueError, TypeError, KeyError, ArithmeticError):
        return {"parse_success": False, "fields": []}
    rows = []
    for field, (item, j) in FIELDS.items():
        target = statement.values[item][j]
        if not values[field].is_finite():
            return {"parse_success": False, "fields": []}
        relative = float(abs(values[field] - target) / abs(target)) if target else None
        # Unit-free error: best match over display units 10^-6..10^6 (totals
        # only), i.e. recovery of the significant digits regardless of scale.
        unit_free = (
            relative
            if item == "eps_basic" or not target
            else min(
                float(abs(values[field] * Decimal(1000) ** k - target) / abs(target))
                for k in range(-2, 3)
            )
        )
        rows.append(
            {
                "field": field,
                "kind": "eps" if item == "eps_basic" else "total",
                "relative_error": relative,
                "unit_free_relative_error": unit_free,
                **{f"within_{t:g}": relative is not None and relative <= t for t in tolerances},
                **{
                    f"unit_free_within_{t:g}": unit_free is not None and unit_free <= t
                    for t in tolerances
                },
            }
        )
    return {"parse_success": True, "fields": rows}
