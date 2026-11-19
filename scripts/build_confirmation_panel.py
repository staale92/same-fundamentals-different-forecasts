#!/usr/bin/env python3
"""Build the declared confirmation panel (configs/confirmation_sample.json)."""

import json
from pathlib import Path
from numeraire.data import build_sec_panel
from numeraire.schema import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "data/raw/sec_confirmation"
eligible = json.loads((raw / "selection.json").read_text())["eligible"]
facts = [str(raw / "facts" / f"CIK{c:010d}.json") for c in eligible]
subs = [str(raw / "submissions" / f"CIK{c:010d}") for c in eligible]
records, audit = build_sec_panel(
    facts,
    2012,
    2025,
    None,
    2026,
    submissions_paths=subs,
    require_verified_first=True,
    derive_liabilities=True,
)
write_jsonl(ROOT / "data/processed/sec_confirmation.jsonl", records)
(ROOT / "data/processed/sec_confirmation_audit.json").write_text(
    json.dumps(audit, indent=2, default=str) + "\n"
)
derived = sum(r["provenance"]["items"]["liabilities"]["tag"].startswith("derived") for r in records)
years = {}
for r in records:
    years[r["outcome_year"]] = years.get(r["outcome_year"], 0) + 1
print(
    json.dumps(
        {
            "records": len(records),
            "firms": len({r["firm_id"] for r in records}),
            "derived_liabilities": derived,
            "by_outcome_year": dict(sorted(years.items())),
        },
        indent=2,
    )
)
