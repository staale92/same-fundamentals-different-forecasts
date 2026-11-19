#!/usr/bin/env python3
"""Next-filing quarterly panel from local company-facts and submissions snapshots.

For each outcome quarter (Q1-Q3 of 2021-2026; Q4 has no standalone EPS in most
filers' XBRL), the input is the latest original filing before the outcome
filing: the 10-K (two fiscal years) for Q1 outcomes, otherwise the preceding
10-Q (its current quarter and the same quarter one year earlier, both standalone
three-month flows from that filing). Prior-year balance sheet instants come from
their own first original report. The outcome is next-quarter basic EPS against
the same quarter a year earlier, both from the outcome 10-Q. Nothing is derived
from annual and year-to-date figures. The window is fixed in advance.
"""

import argparse
from collections import Counter
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
from numeraire.data import TAGS, INSTANT_ITEMS
from numeraire.schema import Statement, REQUIRED, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def qlabel(end):
    d = date.fromisoformat(end)
    for q, (m, day) in {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}.items():
        if abs((d - date(d.year, m, day)).days) <= 8:
            return f"{d.year}Q{q}"
    if d.month == 1 and d.day <= 8:
        return f"{d.year - 1}Q4"
    return None


def ordinal(q):
    return int(q[:4]) * 4 + int(q[-1]) - 1


def from_ordinal(n):
    return f"{n // 4}Q{n % 4 + 1}"


def days(r):
    return (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days + 1


def filings(cik_dir):
    out = {}
    for f in sorted(cik_dir.glob("*.json")):
        if f.name.endswith("provenance.json") or f.name == "submissions.json":
            continue
        d = json.loads(f.read_text())
        r = d["filings"]["recent"] if "filings" in d else d
        for a, form, filed, report in zip(
            r["accessionNumber"], r["form"], r["filingDate"], r["reportDate"]
        ):
            out[a] = {"form": form, "filed": filed, "report": report}
    return out


def fact_rows(facts, item):
    unit = "USD/shares" if item == "eps_basic" else "shares" if item == "shares_basic" else "USD"
    return unit, [
        (tag, r)
        for tag in TAGS[item]
        for r in facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, [])
    ]


def pick(rows, accn, end, kind):
    """First alias with a fact in filing `accn` for the period ending `end`."""
    for tag, r in rows:
        if r["accn"] != accn or r["end"] != end:
            continue
        if kind == "instant" and "start" not in r:
            return tag, r
        if kind == "quarter" and "start" in r and 70 <= days(r) <= 105:
            return tag, r
        if kind == "annual" and "start" in r and 350 <= days(r) <= 378:
            return tag, r
    return None


def first_instant(rows, end, originals):
    best = None
    for tag, r in rows:
        if "start" in r or r["end"] != end or r["accn"] not in originals:
            continue
        if best is None or r["filed"] < best[1]["filed"]:
            best = (tag, r)
    return best


def derived_liabilities(
    facts, kind, in_accn, ends_needed, current_end, periods, by_period, originals
):
    """Liabilities and equity minus equity (including noncontrolling interest, else parent equity plus
    minority interest), each column from the instant and filing used for the other balance-sheet items.
    """

    def rows(tag):
        return [
            (tag, r) for r in facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        ]

    def instant(tag, j):
        rs = rows(tag)
        if kind == "annual":
            return pick(rs, in_accn, ends_needed[j], "instant")
        if j == 1:
            return pick(rs, in_accn, current_end, "instant")
        prior = by_period.get(periods[0])
        return (
            (
                pick(rs, prior[0], prior[1]["report"], "instant")
                or first_instant(rs, prior[1]["report"], originals)
            )
            if prior
            else None
        )

    values, sources = [], []
    for j in (0, 1):
        total = instant("LiabilitiesAndStockholdersEquity", j)
        equity = instant(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", j
        )
        minority = None
        if equity is None:
            equity = instant("StockholdersEquity", j)
            minority = instant("MinorityInterest", j)
        if total is None or equity is None:
            return None
        v = (
            Decimal(str(total[1]["val"]))
            - Decimal(str(equity[1]["val"]))
            - (Decimal(str(minority[1]["val"])) if minority else 0)
        )
        values.append(str(v))
        sources.append(
            {
                "derived": True,
                "tag": "derived:LiabilitiesAndStockholdersEquity-minus-equity",
                "accession": total[1]["accn"],
                "filed": total[1]["filed"],
                "end": total[1]["end"],
                "period": qlabel(total[1]["end"]),
                "value": str(v),
                "unit": "USD",
                "derived_from": [
                    {
                        "tag": f"us-gaap:{t}",
                        "accession": r["accn"],
                        "value": r["val"],
                        "end": r["end"],
                    }
                    for t, r in [total, equity] + ([minority] if minority else [])
                ],
            }
        )
    return values, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-outcome", default="2021Q1")
    p.add_argument("--last-outcome", default="2026Q2")
    p.add_argument("--output", default="data/processed/sec_quarterly.jsonl")
    p.add_argument(
        "--sample",
        choices=["pilot", "confirmation"],
        default="pilot",
        help="pilot: configs/pilot.json firms; confirmation: firms of data/processed/sec_confirmation.jsonl, with derived liabilities",
    )
    a = p.parse_args()
    if a.sample == "pilot":
        ciks = json.loads((ROOT / "configs/pilot.json").read_text())["ciks"]
        raw = ROOT / "data/raw/sec_pilot"
    else:
        ciks = sorted(
            {
                int(json.loads(l)["firm_id"])
                for l in (ROOT / "data/processed/sec_confirmation.jsonl").read_text().splitlines()
                if l.strip()
            }
        )
        raw = ROOT / "data/raw/sec_confirmation"
    derive = a.sample == "confirmation"
    records, audit = [], Counter()
    for cik in ciks:
        tag = f"CIK{int(cik):010d}"
        fpath = raw / "facts" / f"{tag}.json"
        prov = json.loads(Path(str(fpath) + ".provenance.json").read_text())
        facts = json.loads(fpath.read_text())["facts"]
        sub = filings(raw / "submissions" / tag)
        originals = {k: v for k, v in sub.items() if v["form"] in ("10-Q", "10-K")}
        by_period = {}
        for accn, v in originals.items():
            q = qlabel(v["report"]) if v["report"] else None
            if q and (q not in by_period or v["filed"] < by_period[q][1]["filed"]):
                by_period[q] = (accn, v)
        rows = {item: fact_rows(facts, item) for item in TAGS}

        def source(t, r, unit, period):
            s = {
                "accession": r["accn"],
                "filed": r["filed"],
                "form": r["form"],
                "period": period,
                "end": r["end"],
                "value": r["val"],
                "unit": unit,
                "tag": f"us-gaap:{t}",
                "source_url": prov["url"],
                "source_sha256": prov["sha256"],
                "first_original_verified": True,
                "derived": False,
            }
            if "start" in r:
                s["start"] = r["start"]
            return s

        for n in range(ordinal(a.first_outcome), ordinal(a.last_outcome) + 1):
            outcome = from_ordinal(n)
            audit["candidates"] += 1
            if outcome.endswith("Q4"):
                audit["Q4 outcome (no standalone EPS)"] += 1
                continue
            if outcome not in by_period or by_period[outcome][1]["form"] != "10-Q":
                audit["no original outcome 10-Q"] += 1
                continue
            out_accn, out_meta = by_period[outcome]
            prev = from_ordinal(n - 1)
            kind = "annual" if outcome.endswith("Q1") else "quarter_yoy"
            if prev not in by_period:
                audit["no original input filing"] += 1
                continue
            in_accn, in_meta = by_period[prev]
            if (kind == "annual") != (in_meta["form"] == "10-K"):
                audit["input form mismatch"] += 1
                continue
            # Outcome EPS pair from the outcome filing.
            unit_eps, eps_rows = rows["eps_basic"]
            cur = [
                r
                for t, r in eps_rows
                if r["accn"] == out_accn
                and "start" in r
                and 70 <= days(r) <= 105
                and qlabel(r["end"]) == outcome
            ]
            comp = [
                r
                for t, r in eps_rows
                if r["accn"] == out_accn
                and "start" in r
                and 70 <= days(r) <= 105
                and qlabel(r["end"]) == from_ordinal(n - 4)
            ]
            if len({r["val"] for r in cur}) != 1 or len({r["val"] for r in comp}) != 1:
                audit["outcome EPS pair not in outcome filing"] += 1
                continue
            # Decimal arithmetic: a float difference such as 0.94 - 0.93 falls just below one cent.
            change = Decimal(str(cur[0]["val"])) - Decimal(str(comp[0]["val"]))
            if abs(change) < Decimal("0.01"):
                audit["EPS change below threshold"] += 1
                continue
            if kind == "annual":
                fy = int(prev[:4])
                ends = [
                    r["end"]
                    for t, r in rows["assets"][1]
                    if r["accn"] == in_accn and "start" not in r
                ]
                current_end = in_meta["report"]
                prior_end = f"{fy - 1}-{current_end[5:]}"
                ends_needed = [prior_end, current_end]
                periods = None
            else:
                current_end = in_meta["report"]
                q4ago = from_ordinal(n - 5)
                ends_needed = [None, current_end]
                periods = [q4ago, prev]
            values, sources, missing = {}, {}, None
            for item in TAGS:
                unit, rws = rows[item]
                pair = []
                for j in (0, 1):
                    if item in INSTANT_ITEMS:
                        if kind == "annual":
                            hit = pick(rws, in_accn, ends_needed[j], "instant")
                        elif j == 1:
                            hit = pick(rws, in_accn, current_end, "instant")
                        else:
                            prior = by_period.get(periods[0])
                            hit = None
                            if prior:
                                hit = pick(
                                    rws, prior[0], prior[1]["report"], "instant"
                                ) or first_instant(rws, prior[1]["report"], originals)
                    else:
                        if kind == "annual":
                            hit = next(
                                (
                                    h
                                    for h in [
                                        (t, r)
                                        for t, r in rws
                                        if r["accn"] == in_accn
                                        and "start" in r
                                        and 350 <= days(r) <= 378
                                        and r["end"][:4] == str(int(current_end[:4]) - (1 - j))
                                    ]
                                ),
                                None,
                            )
                        else:
                            target = periods[j]
                            hit = next(
                                (
                                    (t, r)
                                    for t, r in rws
                                    if r["accn"] == in_accn
                                    and "start" in r
                                    and 70 <= days(r) <= 105
                                    and qlabel(r["end"]) == target
                                ),
                                None,
                            )
                    if hit is None:
                        break
                    pair.append(hit)
                if len(pair) != 2 and item == "liabilities" and derive:
                    pair = derived_liabilities(
                        facts,
                        kind,
                        in_accn,
                        ends_needed,
                        current_end,
                        periods,
                        by_period,
                        originals,
                    )
                    if pair is not None:
                        values[item] = pair[0]
                        sources[item] = pair[1]
                        continue
                if len(pair) == 2:
                    values[item] = [str(pair[0][1]["val"]), str(pair[1][1]["val"])]
                    sources[item] = [source(t, r, unit, qlabel(r["end"])) for t, r in pair]
                elif item in REQUIRED:
                    missing = item
                    break
            if missing:
                audit[f"missing input {missing}"] += 1
                continue
            input_filed = max(s["filed"] for pair in sources.values() for s in pair)
            if input_filed >= out_meta["filed"]:
                audit["input not before outcome"] += 1
                continue
            years = (
                [int(current_end[:4]) - 1, int(current_end[:4])]
                if kind == "annual"
                else [int(q[:4]) for q in periods]
            )
            record = {
                "id": f"SECQ-{int(cik):010d}-{outcome}",
                "firm_id": f"{int(cik):010d}",
                "input_year": years[1],
                "outcome_year": int(outcome[:4]),
                "years": years,
                "currency": "USD",
                "values": values,
                "label": int(change > 0),
                "input_filed": input_filed,
                "outcome_filed": cur[0]["filed"],
                "provenance": {
                    "frequency": "quarterly_next",
                    "input_kind": kind,
                    "input_accession": in_accn,
                    "input_form": in_meta["form"],
                    **({"input_periods": periods} if periods else {}),
                    "outcome_period": outcome,
                    "outcome_period_start": cur[0]["start"],
                    "outcome_accession": out_accn,
                    "eps_change": f"{change:.4f}",
                    "item_sources": sources,
                    "label_sources": [
                        source("EarningsPerShareBasic", comp[0], unit_eps, from_ordinal(n - 4)),
                        source("EarningsPerShareBasic", cur[0], unit_eps, outcome),
                    ],
                    "target": "next-quarter basic EPS versus the same quarter one year earlier, from the outcome 10-Q",
                },
            }
            try:
                Statement.from_dict(record)
                records.append(record)
                audit["included"] += 1
            except ValueError as e:
                audit[f"schema: {e}"] += 1
    write_jsonl(ROOT / a.output, records)
    summary = {
        "design": "next-filing quarterly outcomes",
        "window": [a.first_outcome, a.last_outcome],
        "records": len(records),
        "firms": len({r["firm_id"] for r in records}),
        "by_outcome_period": dict(
            sorted(Counter(r["provenance"]["outcome_period"] for r in records).items())
        ),
        "audit": dict(sorted(audit.items())),
    }
    (ROOT / a.output).with_suffix(".audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
