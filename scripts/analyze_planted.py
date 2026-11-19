"""Positive-control manipulation checks and confidence intervals."""

from pathlib import Path
import json
from numeraire.schema import read_jsonl
from numeraire.validation import planted_validation

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "results/planted_control"


def main():
    assignments = read_jsonl(CONTROL_DIR / "assignment/assignments.jsonl")
    base = read_jsonl(CONTROL_DIR / "scores/base/predictions.jsonl")
    treated = {r["id"] for r in assignments if r["treated"]}
    memorized = []
    report = None
    for dose in (60, 120, 240):
        predictions = []
        passes = []
        for branch in ("planted", "flipped"):
            post = read_jsonl(CONTROL_DIR / f"scores/{branch}-e{dose}/predictions.jsonl")
            train = [r for r in post if r["arm"] == "N" and r["id"] in treated]
            accuracy = sum(
                (r["probability"] >= 0.5) == (r["label"] if branch == "planted" else 1 - r["label"])
                for r in train
            ) / len(train)
            passes.append(accuracy >= 0.9)
            predictions += [{**r, "variant": branch, "stage": "pre"} for r in base]
            predictions += [{**r, "variant": branch, "stage": "post"} for r in post]
        if all(passes):
            memorized.append(dose)
        if dose == 60:
            report = planted_validation(
                predictions,
                assignments,
                bootstrap=2000,
                seed=2026,
                metric="brier",
                control="flipped",
                trained="N",
                placebo="N_p1",
            )
    report["memorized_doses"] = memorized
    report["removal_intervals"] = {r["arm"]: r["removal_interval"] for r in report["effects"]}
    (CONTROL_DIR / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
