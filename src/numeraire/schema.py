"""Validated statement schema. Outcome/provenance metadata never enter renderings."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any
import hashlib
import json
import re
from pathlib import Path

ITEMS = {
    "revenue": ("Revenue", "income"),
    "cost_of_revenue": ("Cost of revenue", "income"),
    "gross_profit": ("Gross profit", "income"),
    "operating_income": ("Operating income", "income"),
    "interest_expense": ("Interest expense", "income"),
    "income_tax": ("Income tax expense", "income"),
    "net_income": ("Net income", "income"),
    "eps_basic": ("Basic earnings per share", "income"),
    "shares_basic": ("Weighted-average basic shares", "income"),
    "assets": ("Total assets", "balance"),
    "current_assets": ("Current assets", "balance"),
    "cash": ("Cash and cash equivalents", "balance"),
    "receivables": ("Net receivables", "balance"),
    "inventory": ("Inventory", "balance"),
    "ppe": ("Property, plant and equipment, net", "balance"),
    "liabilities": ("Total liabilities", "balance"),
    "current_liabilities": ("Current liabilities", "balance"),
    "long_term_debt": ("Long-term debt", "balance"),
    "equity": ("Equity", "balance"),
    "retained_earnings": ("Retained earnings", "balance"),
    "operating_cash_flow": ("Operating cash flow", "cash_flow"),
}
REQUIRED = ("revenue", "net_income", "eps_basic", "shares_basic", "assets", "liabilities", "equity")


@dataclass(frozen=True)
class Statement:
    id: str
    firm_id: str
    input_year: int
    outcome_year: int
    input_filed: str
    outcome_filed: str
    currency: str
    years: tuple[int, int]
    values: dict[str, tuple[Decimal, Decimal]]
    label: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: dict) -> "Statement":
        provenance = dict(record.get("provenance", {}))
        for field in (
            "synthetic",
            "outcome_period_start",
            "frequency",
            "input_periods",
            "outcome_period",
        ):
            if field in record:
                if field in provenance and provenance[field] != record[field]:
                    raise ValueError(f"Conflicting {field} metadata")
                provenance[field] = record[field]
        if "synthetic" in provenance and not isinstance(provenance["synthetic"], bool):
            raise ValueError("synthetic provenance must be a JSON boolean")
        values = {k: tuple(Decimal(str(v)) for v in pair) for k, pair in record["values"].items()}
        obj = cls(
            **{
                k: record[k]
                for k in (
                    "id",
                    "firm_id",
                    "input_year",
                    "outcome_year",
                    "input_filed",
                    "outcome_filed",
                    "currency",
                    "label",
                )
            },
            years=tuple(record["years"]),
            values=values,
            provenance=provenance,
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if not self.id or not self.firm_id:
            raise ValueError("Statement and firm identifiers are required")
        if self.provenance.get("frequency", "annual") == "quarterly":
            periods = self.provenance.get("input_periods", [])
            outcome = self.provenance.get("outcome_period", "")
            if len(periods) != 2 or any(
                not re.fullmatch(r"\d{4}Q[1-4]", p) for p in [*periods, outcome]
            ):
                raise ValueError(
                    "Quarterly records require two input periods and one outcome period"
                )
            ordinal = lambda p: int(p[:4]) * 4 + int(p[-1]) - 1
            if (
                ordinal(periods[1]) - ordinal(periods[0]) != 1
                or ordinal(outcome) - ordinal(periods[1]) != 1
            ):
                raise ValueError(
                    "Quarterly input periods must be consecutive and outcome must be next quarter"
                )
            if (
                self.years != tuple(int(p[:4]) for p in periods)
                or self.input_year != int(periods[1][:4])
                or self.outcome_year != int(outcome[:4])
            ):
                raise ValueError("Quarter and year metadata disagree")
            if not self.provenance.get("outcome_period_start"):
                raise ValueError("Quarterly exposure requires its actual period start")
        elif self.provenance.get("frequency", "annual") == "quarterly_next":
            # Next-filing design: the input is the latest original filing before
            # the outcome filing (a 10-K for Q1 outcomes, else the preceding
            # 10-Q with its same-quarter prior-year comparative).
            kind = self.provenance.get("input_kind")
            outcome = self.provenance.get("outcome_period", "")
            if not re.fullmatch(r"\d{4}Q[1-4]", outcome) or not self.provenance.get(
                "outcome_period_start"
            ):
                raise ValueError(
                    "Next-filing records require an outcome quarter and its period start"
                )
            ordinal = lambda q: int(q[:4]) * 4 + int(q[-1]) - 1
            if kind == "annual":
                if (
                    self.years != (self.input_year - 1, self.input_year)
                    or outcome != f"{self.input_year+1}Q1"
                ):
                    raise ValueError("Annual inputs must precede a first-quarter outcome")
            elif kind == "quarter_yoy":
                periods = self.provenance.get("input_periods", [])
                if (
                    len(periods) != 2
                    or ordinal(periods[1]) - ordinal(periods[0]) != 4
                    or ordinal(outcome) - ordinal(periods[1]) != 1
                ):
                    raise ValueError(
                        "Quarter inputs must be a same-quarter year-over-year pair followed by the next quarter"
                    )
                if self.years != tuple(int(q[:4]) for q in periods) or self.input_year != int(
                    periods[1][:4]
                ):
                    raise ValueError("Quarter and year metadata disagree")
            else:
                raise ValueError("Unknown next-filing input kind")
            if self.outcome_year != int(outcome[:4]):
                raise ValueError("Outcome year disagrees with outcome quarter")
        elif self.provenance.get("frequency", "annual") != "annual":
            raise ValueError("Frequency must be annual, quarterly, or quarterly_next")
        elif (
            self.years != (self.input_year - 1, self.input_year)
            or self.outcome_year != self.input_year + 1
        ):
            raise ValueError("Expected consecutive input years and next-year outcome")
        if self.label not in (0, 1) or isinstance(self.label, bool):
            raise ValueError("label must be integer zero or one")
        if date.fromisoformat(self.input_filed) >= date.fromisoformat(self.outcome_filed):
            raise ValueError("Outcome filing must follow the input filing")
        if self.currency not in {"USD", "JPY"}:
            raise ValueError("Only explicitly supported native currencies USD/JPY are accepted")
        if set(REQUIRED) - self.values.keys():
            raise ValueError(f"Missing core items: {sorted(set(REQUIRED) - self.values.keys())}")
        if self.values.keys() - ITEMS.keys():
            raise ValueError(f"Unknown items: {sorted(self.values.keys() - ITEMS.keys())}")
        for k, pair in self.values.items():
            if len(pair) != 2 or any(not v.is_finite() for v in pair):
                raise ValueError(f"Invalid two-year numeric pair for {k}")
        if any(v <= 0 for k in ("assets", "shares_basic") for v in self.values[k]):
            raise ValueError("Assets and basic shares must be positive")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "firm_id": self.firm_id,
            "input_year": self.input_year,
            "outcome_year": self.outcome_year,
            "input_filed": self.input_filed,
            "outcome_filed": self.outcome_filed,
            "currency": self.currency,
            "years": list(self.years),
            "values": {k: [str(v) for v in pair] for k, pair in self.values.items()},
            "label": self.label,
            "provenance": self.provenance,
        }


def stable_seed(seed: int, *parts: object) -> int:
    """Independent of Python's randomized process hash."""
    payload = json.dumps([seed, *parts], sort_keys=True, ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def exposure_cell(record: dict, cutoff: str) -> str:
    """Conservative exposure: outcome fiscal period must be entirely after cutoff.

    A filing after cutoff alone is insufficient: intra-year information may already
    be public. The intermediate interval is excluded, not declared unexposed.
    Exposure is eligibility for recall, never proof of inclusion in training.
    """
    cut = date.fromisoformat(cutoff)
    filed = date.fromisoformat(record["outcome_filed"])
    if filed <= cut:
        return "exposed"
    start = record.get(
        "outcome_period_start", record.get("provenance", {}).get("outcome_period_start")
    )
    if not start:
        if record.get("currency", "USD") != "USD":
            raise ValueError("Non-December fiscal years need outcome_period_start provenance")
        start = f"{record['outcome_year']}-01-01"
    return "unexposed" if date.fromisoformat(start) > cut else "donut"
