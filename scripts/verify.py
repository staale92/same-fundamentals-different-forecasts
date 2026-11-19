"""Offline integrity, prompt, scoring and numerical-result checks."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import numpy as np
from numeraire.schema import Statement, exposure_cell, read_jsonl
from numeraire.token_inference import (
    class_messages,
    conditional_probability,
    digest,
    parse_reasoning,
    reasoning_messages,
)
from numeraire.elicitation import messages
from numeraire.inversion import inversion_messages, score_inversion

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(expected, actual, path="report", tolerance=1e-10):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, f"Missing {path}.{key}")
            compare(value, actual[key], f"{path}.{key}", tolerance)
    elif isinstance(expected, list):
        require(len(expected) == len(actual), f"Length differs: {path}")
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare(x, y, f"{path}[{i}]", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance),
            f"Numeric mismatch: {path}: {expected} != {actual}",
        )
    else:
        require(expected == actual, f"Mismatch: {path}")


def check_results():
    expected = json.loads((ROOT / "expected/primary/report.json").read_text())
    actual = json.loads((ROOT / "results/primary/report.json").read_text())
    compare(expected, actual)
    controls = json.loads((ROOT / "results/controls/report.json").read_text())
    reference = json.loads((ROOT / "expected/controls/report.json").read_text())
    for key in (
        "rho",
        "rho_NP",
        "removal_by_dose",
        "splits",
        "mean_shift_diagnostic",
        "representation_retention",
        "variance",
    ):
        compare(reference[key], controls[key], key)
    compare(
        json.loads((ROOT / "expected/presentation/report.json").read_text()),
        json.loads((ROOT / "results/presentation/report.json").read_text()),
        "presentation",
    )
    compare(
        json.loads((ROOT / "expected/presentation/magnitudes.json").read_text()),
        json.loads((ROOT / "results/presentation/magnitudes.json").read_text()),
        "magnitudes",
    )
    compare(
        json.loads((ROOT / "expected/diagnostics/report.json").read_text()),
        json.loads((ROOT / "results/diagnostics/report.json").read_text()),
        "diagnostics",
    )
    compare(
        json.loads((ROOT / "expected/audit_checks/report.json").read_text()),
        json.loads((ROOT / "results/audit_checks/report.json").read_text()),
        "audit_checks",
    )
    compare(
        json.loads((ROOT / "expected/identification/report.json").read_text()),
        json.loads((ROOT / "results/identification/report.json").read_text()),
        "identification",
    )
    compare(
        json.loads((ROOT / "expected/sensitivity/report.json").read_text()),
        json.loads((ROOT / "results/sensitivity/report.json").read_text()),
        "temporal_and_identification_sensitivity",
    )
    compare(
        json.loads((ROOT / "expected/elicitation/report.json").read_text()),
        json.loads((ROOT / "results/elicitation/report.json").read_text()),
        "elicitation_checks",
    )
    return {
        "primary_and_sensitivities": "match",
        "planted_and_drift_controls": "match",
        "presentation_sensitivity": "match",
        "forecast_change_magnitudes": "match",
        "matched_benchmarks_and_diagnostics": "match",
        "audit_checks": "match",
        "identification_probe": "match",
        "temporal_and_identification_sensitivity": "match",
        "elicitation_checks": "match",
        "absolute_tolerance": 1e-10,
    }


def check_inputs():
    records = {}
    counts = {
        "sec_pilot": (127, 11),
        "sec_quarterly": (184, 11),
        "sec_confirmation": (1186, 128),
        "sec_quarterly_confirmation": (1592, 107),
    }
    for name, (n, firms) in counts.items():
        panel = read_jsonl(ROOT / f"data/processed/{name}.jsonl")
        require(len(panel) == n and len({r["id"] for r in panel}) == n, f"Wrong row count: {name}")
        require(len({r["firm_id"] for r in panel}) == firms, f"Wrong firm count: {name}")
        for r in panel:
            Statement.from_dict(r)
            records[r["id"]] = r
    reference_ids = {
        r["id"] for r in read_jsonl(ROOT / "data/processed/planted_rank_reference.jsonl")
    }
    primary_ids = {
        r["id"] for r in read_jsonl(ROOT / "data/processed/sec_quarterly_confirmation.jsonl")
    }
    require(
        reference_ids < primary_ids and len(primary_ids - reference_ids) == 5,
        "Planted-reference membership differs",
    )
    forecasts, contexts = 0, 0
    coverage = {}
    for file in sorted((ROOT / "results").glob("**/predictions.jsonl")):
        rows = read_jsonl(file)
        seen = set()
        manifest = file.parent / "manifest.json"
        system = json.loads(manifest.read_text()).get("system_message") if manifest.exists() else None
        reference = (
            "planted_rank_reference"
            if "scores_representations" in str(file)
            else "sec_quarterly_confirmation"
        )
        os.environ["NUMERAIRE_RANK_REFERENCE_QUARTERLY"] = str(
            ROOT / f"data/processed/{reference}.jsonl"
        )
        for r in rows:
            key = (r["id"], r["arm"])
            require(key not in seen, f"Duplicate prediction: {file}: {key}")
            seen.add(key)
            original = records[r["id"]]
            require(r["label"] == original["label"], f"Wrong outcome label: {key}")
            require(
                r["status"] == "success" and r["synthetic"] is False,
                f"Non-empirical forecast: {key}",
            )
            if r.get("cutoff"):
                require(
                    r["exposure"] == exposure_cell(original, r["cutoff"]), f"Wrong exposure: {key}"
                )
            statement = Statement.from_dict(original)
            require(len(r["scores"]) == 2, f"Missing swapped context: {key}")
            probabilities = []
            for score in r["scores"]:
                probability = conditional_probability(
                    score["logp_A"], score["logp_B"], score["swap"]
                )
                require(abs(probability - score["p_up"]) < 1e-12, f"Probability mismatch: {key}")
                expected_hash = digest(
                    class_messages(statement, r["arm"], score["swap"], system=system)
                )
                require(
                    score["prompt_hash"] == expected_hash,
                    f"Prompt changed: {file.relative_to(ROOT)}: {key}",
                )
                probabilities.append(probability)
                contexts += 1
            require(
                abs(float(np.mean(probabilities)) - r["probability"]) < 1e-12,
                f"Pooled probability mismatch: {key}",
            )
            forecasts += 1
        if "_quarterly_confirmation" in file.parent.name:
            model = file.parent.name.split("_quarterly_confirmation")[0]
            coverage.setdefault(model, set()).update(i for i, a in seen if a == "N")
    for model, ids in coverage.items():
        require(ids == primary_ids, f"Primary panel incomplete: {model}")
    reasoning_answers = 0
    for file in sorted((ROOT / "results/hosted").glob("*/answers.jsonl")):
        manifest = json.loads((file.parent / "manifest.json").read_text())
        seen = set()
        for r in read_jsonl(file):
            key = (r["id"], r["arm"])
            require(key not in seen, f"Duplicate answer: {file}: {key}")
            seen.add(key)
            original = records[r["id"]]
            require(r["label"] == original["label"], f"Wrong outcome label: {key}")
            require(r["exposure"] == exposure_cell(original, manifest["cutoff"]), f"Wrong exposure: {key}")
            content = reasoning_messages(
                Statement.from_dict(original), r["arm"], manifest["seed"], manifest["system_message"]
            )
            require(digest(content) == r["prompt_hash"], f"Reasoning prompt changed: {key}")
            probability, how = parse_reasoning(r["content"])
            require(how == r["parse"], f"Reasoning parse changed: {key}")
            require(
                probability is not None and abs(min(max(probability, 0.0), 1.0) - r["probability"]) < 1e-12,
                f"Reasoning forecast changed: {key}",
            )
            reasoning_answers += 1
        require(
            seen == {(i, a) for i in primary_ids for a in manifest["arms"]},
            f"Reasoning panel incomplete: {file}",
        )
    probes = 0
    for file in sorted((ROOT / "results").glob("**/scores.jsonl")):
        for r in read_jsonl(file):
            if r["arm"] == "content_free":
                original = next(
                    x
                    for x in records.values()
                    if x["provenance"].get("frequency", "annual") == r["frequency"]
                )
            else:
                original = records[r["id"]]
            content = messages(
                Statement.from_dict(original),
                r["arm"],
                r["kind"],
                r["task"],
                r["swap"],
                content_free=r.get("content_free"),
            )
            require(digest(content) == r["prompt_hash"], f'Probe prompt changed: {file}: {r["id"]}')
            require(
                all(math.isfinite(x) for x in r["logp"].values()),
                "Invalid probe token probabilities",
            )
            probes += 1
    inversion_prompts = 0
    inversion_files = [
        *(ROOT / "results").glob("*/inversion_nc/probes.jsonl"),
        *(ROOT / "results/inversion_hosted").glob("*/probes.jsonl"),
    ]
    for file in sorted(inversion_files):
        for r in read_jsonl(file):
            statement = Statement.from_dict(records[r["id"]])
            require(
                digest(inversion_messages(statement, r["arm"])) == r["prompt_hash"],
                "Inversion prompt changed",
            )
            require(
                score_inversion(r["raw_text"], statement)
                == {k: r[k] for k in ("parse_success", "all_correct", "fields")},
                "Inversion parse/score changed",
            )
            inversion_prompts += 1
    return {
        "panels": counts,
        "forecasts_verified": forecasts,
        "answer_contexts_verified": contexts,
        "primary_rows_outside_planted_reference": 5,
        "elicitation_contexts_verified": probes,
        "inversion_prompts_verified": inversion_prompts,
        "reasoning_answers_verified": reasoning_answers,
    }


def check_identification():
    """Check probe coverage, candidate assignment, and saved answer scores."""
    config = json.loads((ROOT / "configs/identification_probe.json").read_text())
    records = {r["id"]: r for r in read_jsonl(ROOT / config["panel"])}
    identities = json.loads((ROOT / config["identities"]).read_text())
    firms = sorted({r["firm_id"] for r in records.values()})
    expected_keys = {(i, arm) for i in records for arm in config["arms"]}
    letters = config["forced_choice"]["letters"]
    candidate_count = config["forced_choice"]["candidate_count"]
    expected_candidates = {}
    for observation_id, record in records.items():
        rng = random.Random(f"{observation_id}|choices")
        others = [f for f in firms if f != record["firm_id"]]
        candidates = rng.sample(others, candidate_count - 1) + [record["firm_id"]]
        rng.shuffle(candidates)
        expected_candidates[observation_id] = candidates

    counts = {"free_text_answers": 0, "forced_choice_scores": 0}
    models = set()
    for folder in sorted((ROOT / "results/identification").glob("llama31-*")):
        manifest = json.loads((folder / "manifest.json").read_text())
        choice_manifest = json.loads((folder / "manifest_choice.json").read_text())
        models.add(manifest["model"])
        for metadata in (manifest, choice_manifest):
            require(metadata["system"] == config["system_message"], "Probe system differs")
            require(metadata["arms"] == config["arms"], "Probe arms differ")
            require(metadata["model"] == manifest["model"], "Probe models differ")
        require(
            manifest["question"] == config["free_text"]["question"],
            "Probe question differs",
        )
        require(
            manifest["decoding"] == {
                key: config["free_text"][key] for key in ("temperature", "max_tokens")
            },
            "Probe decoding differs",
        )
        for filename, count_key in (
            ("answers.jsonl", "free_text_answers"),
            ("choice_scores.jsonl", "forced_choice_scores"),
        ):
            rows = read_jsonl(folder / filename)
            keys = [(r["id"], r["arm"]) for r in rows]
            require(
                len(keys) == len(set(keys)) and set(keys) == expected_keys,
                f"Identification coverage differs: {folder.name}/{filename}",
            )
            for row in rows:
                record = records[row["id"]]
                require(row["firm_id"] == record["firm_id"], "Probe firm differs")
                require(row["model"] == manifest["model"], "Probe model differs")
                require(row["firm_id"] in identities, "Probe identity missing")
                if filename == "choice_scores.jsonl":
                    candidates = expected_candidates[row["id"]]
                    require(row["candidates"] == candidates, "Probe candidates differ")
                    require(
                        row["true_letter"] == letters[candidates.index(row["firm_id"])],
                        "Probe true letter differs",
                    )
                    require(
                        set(row["logp"]) == set(letters)
                        and all(math.isfinite(p) and p <= 0 for p in row["logp"].values()),
                        "Invalid identification log probabilities",
                    )
            counts[count_key] += len(rows)
    require(models == set(config["models"]), "Identification models incomplete")
    return counts


def check_checksums():
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        return "checksum manifest unavailable"
    count = 0
    for line in manifest.read_text().splitlines():
        sha, name = line.split("  ", 1)
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha,
            f"Checksum mismatch: {name}",
        )
        count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-only", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    report = {"results": check_results()}
    if not args.results_only:
        report["inputs"] = check_inputs()
        report["identification"] = check_identification()
        report["checksums"] = check_checksums()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
