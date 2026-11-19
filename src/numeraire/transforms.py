"""Exact financial-statement re-expressions and information-changing rank/ratio arms."""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
import random
from .schema import ITEMS, Statement, stable_seed

ARMS = ("E", "U")
NATIVE_ARMS = ("N", "NP", "NC", "N_id", "N_p1", "N_p2")
EXACT_ARMS = frozenset({"E", "U", *NATIVE_ARMS})
NATIVE_PARAPHRASES = {
    "N_p1": (
        "Financial statements of an unnamed company for two consecutive fiscal periods, labeled Prior and Current.",
        "All monetary amounts are stated in {money}; share counts are stated in {shares}.",
        "Basic EPS is stated in {currency} per share. A missing line item is not a zero value.",
        "Financial statements of an unnamed company: Current is its latest fiscal quarter and Prior is the same quarter a year earlier.",
    ),
    "N_p2": (
        "The tables below describe one anonymized firm over two consecutive fiscal periods (Prior, Current).",
        "Units: {money} for monetary totals and {shares} for share counts.",
        "Basic EPS is in {currency} per share. Line items that are absent were not reported; they are not zero.",
        "The tables below describe one anonymized firm in its latest fiscal quarter (Current) and the same quarter one year earlier (Prior).",
    ),
}
_SCALE_NAMES = {Decimal(1000000): "millions", Decimal(1000): "thousands", Decimal(1): "units"}
INFORMATION_ARMS = ("R", "Q")


@dataclass(frozen=True)
class Rendering:
    arm: str
    text: str
    values: dict[str, tuple[Decimal, Decimal]]
    order: tuple[str, ...]
    currency: str
    display_unit: Decimal
    fx: Decimal
    share_factor: Decimal
    scale: Decimal
    metadata: dict

    def recover_native(self) -> dict[str, tuple[Decimal, Decimal]]:
        """Inverse for exact arms. Values are display-currency whole amounts.

        Monetary and per-share quantities are recovered with 40-digit
        decimal arithmetic and the disclosed conversion factors.
        """
        if self.arm not in EXACT_ARMS:
            raise ValueError("Information-changing arms have no exact inverse")
        with localcontext() as ctx:
            ctx.prec = 40
            return {
                k: tuple(
                    (
                        (
                            v / self.share_factor
                            if k == "shares_basic"
                            else (
                                v * self.share_factor / self.fx if k == "eps_basic" else v / self.fx
                            )
                        )
                        for v in pair
                    )
                )
                for k, pair in self.values.items()
            }


def _number(value: Decimal) -> str:
    text = format(value, ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def native_units(statement: Statement) -> tuple[Decimal, Decimal]:
    """As-filed display scales inferred from reporting precision.

    XBRL values are rounded to the displayed precision, so a filing presented
    in millions reports every monetary fact as a multiple of 10^6. Returns the
    largest of 10^6, 10^3, 1 dividing all monetary items, and likewise for
    share counts. This is an inference, not a reconstruction of typography.
    """

    def largest(values):
        values = [Decimal(v) for v in values]
        for unit in (Decimal(1000000), Decimal(1000)):
            if values and all((v % unit == 0 for v in values)):
                return unit
        return Decimal(1)

    money = [
        v
        for k, pair in statement.values.items()
        if k not in ("eps_basic", "shares_basic")
        for v in pair
    ]
    return (largest(money), largest(statement.values.get("shares_basic", ())))


def render(
    statement: Statement, arm: str = "E", seed: int = 2026, fx: str | None = None
) -> Rendering:
    statement.validate()
    if arm in INFORMATION_ARMS:
        return _render_information(statement, arm)
    if arm not in (*ARMS, *NATIVE_ARMS):
        raise ValueError(f"Unknown arm {arm}")
    rng = random.Random(stable_seed(seed, statement.id, "P" if arm == "NP" else arm))
    currency = statement.currency
    rate, multiplier, share_factor = (Decimal(1), Decimal(1), Decimal(1))
    unit = Decimal(1000000 if arm == "U" else 1000)
    share_unit = unit
    if arm in NATIVE_ARMS:
        unit, share_unit = native_units(statement)
    if arm in {"NC"}:
        definition = Decimal("150")
        rate = Decimal(
            fx or (str(definition) if currency == "USD" else str(Decimal(1) / definition))
        )
        currency = "JPY" if currency == "USD" else "USD"
    if rate <= 0:
        raise ValueError("Conversion factor must be positive")
    with localcontext() as ctx:
        ctx.prec = 40
        values = {
            k: tuple(
                (
                    (
                        v * share_factor
                        if k == "shares_basic"
                        else (
                            v * rate * multiplier / share_factor
                            if k == "eps_basic"
                            else v * rate * multiplier
                        )
                    )
                    for v in statement.values[k]
                )
            )
            for k in ITEMS
            if k in statement.values
        }
    order = [k for k in ITEMS if k in values]
    if arm in {"NP"}:
        order = []
        for section in ("income", "balance", "cash_flow"):
            section_items = [k for k in ITEMS if k in values and ITEMS[k][1] == section]
            rng.shuffle(section_items)
            order.extend(section_items)
    scale_name = "millions" if unit == 1000000 else "thousands"
    if arm in NATIVE_ARMS:
        scale_name = _SCALE_NAMES[unit]
        share_text = "whole shares" if share_unit == 1 else f"{_SCALE_NAMES[share_unit]} of shares"
        money_text = f"whole {currency}" if unit == 1 else f"{currency} {scale_name}"
        notes = [
            "An anonymous company; consecutive fiscal periods are labeled Prior and Current.",
            f"Monetary totals are in {money_text}; share counts are in {share_text}.",
            f"Basic EPS is in {currency} per share. Missing line items are not zero.",
        ]
        if arm in NATIVE_PARAPHRASES:
            notes = [
                line.format(money=money_text, shares=share_text, currency=currency)
                for line in NATIVE_PARAPHRASES[arm][:3]
            ]
    else:
        notes = [
            "An anonymous company; consecutive fiscal periods are labeled Prior and Current.",
            f"Monetary totals are in {currency} {scale_name}; share counts are in {scale_name} of shares.",
            f"Basic EPS is in {currency} per share. Missing line items are not zero.",
        ]
    if statement.provenance.get("input_kind") == "quarter_yoy":
        notes[0] = (
            NATIVE_PARAPHRASES[arm][3]
            if arm in NATIVE_PARAPHRASES
            else "An anonymous company; Current is the latest fiscal quarter and Prior is the same fiscal quarter one year earlier."
        )
    elif (
        statement.provenance.get("frequency") == "quarterly_next" and arm not in NATIVE_PARAPHRASES
    ):
        notes[0] = "An anonymous company; consecutive fiscal years are labeled Prior and Current."
    if arm == "N_id":
        notes.append(
            f"These statements were converted from their native {statement.currency} to {currency} solely for presentation at 1 {currency} per 1 {statement.currency}. The company and its real size are unchanged."
        )
    elif arm in {"NC"}:
        notes.append(
            f"These statements were converted from their native {statement.currency} to {currency} solely for presentation at {_number(rate)} {currency} per 1 {statement.currency}. The company and its real size are unchanged."
        )
    notes.extend(["", "Item | Prior | Current"])
    section = None
    with localcontext() as ctx:
        ctx.prec = 40
        for k in order:
            if ITEMS[k][1] != section:
                section = ITEMS[k][1]
                notes.append(f"[{section.replace('_', ' ').title()}]")
            divisor = share_unit if k == "shares_basic" else unit
            pair = values[k] if k == "eps_basic" else tuple((v / divisor for v in values[k]))
            notes.append(f"{ITEMS[k][0]} | {_number(pair[0])} | {_number(pair[1])}")
    metadata = {
        "exact_content_preserving": arm in EXACT_ARMS,
        "digit_retention_is_assumption": arm in {"U", "E", "NP"},
        "native_display_units": [str(u) for u in native_units(statement)],
        "decimal_precision": 40,
        "warning": "identity exclusion does not certify removal of memory",
    }
    return Rendering(
        arm,
        "\n".join(notes),
        values,
        tuple(order),
        currency,
        unit,
        rate,
        share_factor,
        multiplier,
        metadata,
    )


_RANK_CACHE: dict = {}


def _rank_key(record, j, item):
    """Annual: (input year, column, item). Quarterly next-filing: (input period of that column, column, item)."""
    prov = record["provenance"] if isinstance(record, dict) else record.provenance
    years = record["years"] if isinstance(record, dict) else record.years
    input_year = record["input_year"] if isinstance(record, dict) else record.input_year
    if prov.get("frequency") == "quarterly_next":
        period = (
            prov["input_periods"][j] if prov.get("input_kind") == "quarter_yoy" else f"FY{years[j]}"
        )
        return ("Q", period, j, item)
    return (input_year, j, item)


def rank_reference(path=None, quarterly=False):
    """Per (input year, column, item) sorted values of the declared reference panel.

    Default: data/processed/sec_confirmation.jsonl relative to the working
    directory, or NUMERAIRE_RANK_REFERENCE. Its SHA-256 enters every R rendering.
    """
    import hashlib, json, os
    from pathlib import Path

    default = (
        "data/processed/sec_quarterly_confirmation.jsonl"
        if quarterly
        else "data/processed/sec_confirmation.jsonl"
    )
    path = Path(
        path
        or os.environ.get(
            "NUMERAIRE_RANK_REFERENCE_QUARTERLY" if quarterly else "NUMERAIRE_RANK_REFERENCE",
            default,
        )
    )
    key = str(path.resolve())
    if key not in _RANK_CACHE:
        raw = path.read_bytes()
        groups: dict = {}
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for item, pair in r["values"].items():
                for j in (0, 1):
                    groups.setdefault(_rank_key(r, j, item), []).append(Decimal(str(pair[j])))
        _RANK_CACHE[key] = (
            {k: sorted(v) for k, v in groups.items()},
            hashlib.sha256(raw).hexdigest(),
        )
    return _RANK_CACHE[key]


def _percentile(sorted_values, v):
    import bisect

    n = len(sorted_values)
    if n < 2:
        return None
    below = bisect.bisect_left(sorted_values, v)
    equal = bisect.bisect_right(sorted_values, v) - below
    mid_rank = below + (equal + 1) / 2
    return round(100 * (mid_rank - 1) / (n - 1))


_RATIO_ROWS = (
    ("gross_margin", "Gross margin"),
    ("operating_margin", "Operating margin"),
    ("net_margin", "Net margin"),
    ("return_on_assets", "Return on assets"),
    ("liabilities_assets", "Liabilities / assets"),
    ("current_ratio", "Current assets / current liabilities"),
    ("cash_assets", "Cash / assets"),
    ("long_debt_assets", "Long-term debt / assets"),
    ("operating_cash_assets", "Operating cash flow / assets"),
    ("asset_turnover", "Revenue / assets"),
)


def _ratios(values, j):

    def val(k):
        try:
            return float(values[k][j])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def div(a, b):
        return a / b if math.isfinite(a) and math.isfinite(b) and (b != 0) else float("nan")

    revenue, assets = (val("revenue"), val("assets"))
    return {
        "gross_margin": div(val("gross_profit"), revenue),
        "operating_margin": div(val("operating_income"), revenue),
        "net_margin": div(val("net_income"), revenue),
        "return_on_assets": div(val("net_income"), assets),
        "liabilities_assets": div(val("liabilities"), assets),
        "current_ratio": div(val("current_assets"), val("current_liabilities")),
        "cash_assets": div(val("cash"), assets),
        "long_debt_assets": div(val("long_term_debt"), assets),
        "operating_cash_assets": div(val("operating_cash_flow"), assets),
        "asset_turnover": div(revenue, assets),
    }


def _pct(x):
    return "n/a" if not math.isfinite(x) else f"{100 * x:.1f}%".replace("-0.0%", "0.0%")


def _render_information(statement: Statement, arm: str) -> Rendering:
    """R: integer percentile ranks within the reference cross-section; Q: rounded ratios only.

    These change the information shown; they are stress tests, not exact re-expressions.
    """
    values = statement.values
    notes = ["An anonymous company; consecutive fiscal periods are labeled Prior and Current."]
    metadata = {"exact_content_preserving": False, "information_changing": True}
    if arm == "R":
        quarterly = statement.provenance.get("frequency") == "quarterly_next"
        ref, digest = rank_reference(quarterly=quarterly)
        n = len(ref.get(_rank_key(statement, 1, "assets"), []))
        notes += [
            f"Each line item is shown as its percentile rank, from 0 (lowest) to 100 (highest), among about {n} large US non-financial companies in the same {('fiscal period' if quarterly else 'fiscal year')}; the Prior and Current columns are ranked separately.",
            "No amounts are shown. Missing line items are not zero.",
            "",
            "Item | Prior | Current",
        ]
        section = None
        for k in [k for k in ITEMS if k in values]:
            ranks = [
                _percentile(ref.get(_rank_key(statement, j, k), []), values[k][j]) for j in (0, 1)
            ]
            if any((x is None for x in ranks)):
                continue
            if ITEMS[k][1] != section:
                section = ITEMS[k][1]
                notes.append(f"[{section.replace('_', ' ').title()}]")
            notes.append(f"{ITEMS[k][0]} | {ranks[0]} | {ranks[1]}")
        metadata["rank_reference_sha256"] = digest
    else:
        r0, r1 = (_ratios(values, 0), _ratios(values, 1))

        def growth(k):
            try:
                a, b = (float(values[k][0]), float(values[k][1]))
            except (KeyError, TypeError, ValueError):
                return float("nan")
            return (b - a) / abs(a) if a != 0 else float("nan")

        notes += [
            "Only financial ratios are shown, in percent; growth rates compare Current with Prior.",
            "No amounts are shown. A ratio marked n/a could not be computed.",
            "",
            "Ratio | Prior | Current",
        ]
        notes += [f"{label} | {_pct(r0[k])} | {_pct(r1[k])}" for k, label in _RATIO_ROWS]
        notes += [
            f"Revenue growth |  | {_pct(growth('revenue'))}",
            f"Basic EPS growth |  | {_pct(growth('eps_basic'))}",
        ]
    if statement.provenance.get("frequency") == "quarterly_next":
        notes[0] = "An anonymous company; consecutive fiscal years are labeled Prior and Current."
    if statement.provenance.get("input_kind") == "quarter_yoy":
        notes[0] = (
            "An anonymous company; Current is the latest fiscal quarter and Prior is the same fiscal quarter one year earlier."
        )
    return Rendering(
        arm,
        "\n".join(notes),
        dict(values),
        tuple((k for k in ITEMS if k in values)),
        statement.currency,
        Decimal(1),
        Decimal(1),
        Decimal(1),
        Decimal(1),
        metadata,
    )
