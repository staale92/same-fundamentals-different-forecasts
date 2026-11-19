"""Recreate the executed 40-firm planted-control sample and training examples.

The fixed planted-control reference panel defines the sampling frame and
percentile-rank inputs. This script prepares data without loading a model.
"""

from pathlib import Path
import argparse
import json
import numpy as np
from numeraire.schema import Statement, read_jsonl, write_jsonl, stable_seed
from numeraire.token_inference import class_messages

ROOT = Path(__file__).resolve().parents[1]
HEADER = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"


def prepare(output, seed=2026):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    by_firm = {}
    for r in read_jsonl(ROOT / "data/processed/planted_rank_reference.jsonl"):
        by_firm.setdefault(r["firm_id"], []).append(r)
    firms = sorted(by_firm, key=lambda f: stable_seed(seed, "plant_firm", f))[:40]
    panel = [
        min(by_firm[f], key=lambda r: stable_seed(seed, "plant_firm_year", r["id"])) for f in firms
    ]
    eligible = sorted(panel, key=lambda r: r["id"])
    rng = np.random.default_rng(seed)
    strata = {y: [r for r in eligible if r["label"] == y] for y in (0, 1)}
    quota = {y: len(rows) // 2 for y, rows in strata.items()}
    odd = [y for y, rows in strata.items() if len(rows) % 2]
    rng.shuffle(odd)
    for y in odd[: len(eligible) // 2 - sum(quota.values())]:
        quota[y] += 1
    treated = set()
    for y, rows in strata.items():
        treated.update(rows[i]["id"] for i in rng.permutation(len(rows))[: quota[y]])
    assignment = [
        {
            "id": r["id"],
            "firm_id": r["firm_id"],
            "label": r["label"],
            "treated": r["id"] in treated,
            "outcome_year": r["outcome_year"],
        }
        for r in eligible
    ]
    write_jsonl(output / "panel.jsonl", panel)
    write_jsonl(output / "assignments.jsonl", assignment)
    for branch in ("planted", "flipped"):
        train, chat = [], []
        for r in eligible:
            if r["id"] not in treated:
                continue
            label = r["label"] if branch == "planted" else 1 - r["label"]
            for swap in (False, True):
                answer = "A" if bool(label) != swap else "B"
                messages = class_messages(Statement.from_dict(r), "N", swap, seed)
                train.append(
                    {
                        "id": r["id"],
                        "swap": swap,
                        "arm": "N",
                        "messages": messages,
                        "completion": answer,
                        "training_label": int(label),
                    }
                )
                # Include the date header explicitly to match the scoring template.
                formatted = [dict(message) for message in messages]
                formatted[0]["content"] = HEADER + formatted[0]["content"]
                chat.append({"messages": formatted + [{"role": "assistant", "content": answer}]})
        write_jsonl(output / f"{branch}_train.jsonl", train)
        write_jsonl(output / f"{branch}_chat.jsonl", chat)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
