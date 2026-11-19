#!/usr/bin/env python3
"""Apply the declared confirmation-sample rule (configs/confirmation_sample.json).

Downloads the SEC XBRL frame us-gaap/Assets/USD/CY2018Q4I once (archived with
its hash), takes the 400 largest filers by total assets, then excludes
financial firms (SIC 6000-6999), non-December fiscal years, and the sixteen
pilot filers, using each candidate's submissions register.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from numeraire.data import download_cached, fetch_sec_companyfacts, fetch_sec_submissions

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/raw/sec_confirmation")
    a = p.parse_args()
    ua = os.environ["SEC_USER_AGENT"]
    root = ROOT / a.output
    root.mkdir(parents=True, exist_ok=True)
    frame = download_cached(
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2018Q4I.json",
        root / "frames_Assets_CY2018Q4I.json",
        ua,
    )
    data = json.loads(Path(frame).read_text())
    top = sorted(data["data"], key=lambda r: -float(r["val"]))[
        :400
    ]  # candidate count specified in configs/confirmation_sample.json
    pilot = {int(c) for c in json.loads((ROOT / "configs/pilot.json").read_text())["ciks"]}
    rows = []
    for r in top:
        cik = int(r["cik"])
        row = {
            "cik": cik,
            "entityName": r["entityName"],
            "assets_CY2018Q4I": r["val"],
            "end": r["end"],
        }
        if cik in pilot:
            rows.append({**row, "status": "excluded: pilot filer"})
            continue
        try:
            sub = fetch_sec_submissions(cik, root / "submissions" / f"CIK{cik:010d}", ua)
        except ValueError:
            # A cache from an interrupted transfer fails its checksum: refetch once.
            import shutil

            shutil.rmtree(root / "submissions" / f"CIK{cik:010d}", ignore_errors=True)
            sub = fetch_sec_submissions(cik, root / "submissions" / f"CIK{cik:010d}", ua)
        main_json = json.loads(
            Path(root / "submissions" / f"CIK{cik:010d}" / f"CIK{cik:010d}.json").read_text()
        )
        sic = str(main_json.get("sic") or "")
        fye = main_json.get("fiscalYearEnd")
        row.update({"sic": sic, "fiscalYearEnd": fye})
        if sic.isdigit() and 6000 <= int(sic) <= 6999:
            row["status"] = "excluded: financial SIC"
        elif fye != "1231":
            row["status"] = "excluded: fiscal year end not December"
        else:
            fetch_sec_companyfacts(cik, root / "facts" / f"CIK{cik:010d}.json", ua)
            row["status"] = "eligible"
        rows.append(row)
    manifest = {
        "frame_sha256": hashlib.sha256(Path(frame).read_bytes()).hexdigest(),
        "candidates": rows,
        "eligible": [r["cik"] for r in rows if r["status"] == "eligible"],
    }
    (root / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "eligible": len(manifest["eligible"]),
                "excluded": {
                    s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
