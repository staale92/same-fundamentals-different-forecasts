#!/usr/bin/env python3
"""Matched annual benchmarks, quarterly mean shifts, and repeat-scoring diagnostics.

These descriptive checks use saved predictions. The uniform-shift calculation is
a hypothetical diagnostic: individual arm differences are not constant in the data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from numeraire.benchmark import ratio_features, walk_forward_predictions
from numeraire.robustness import auc
from numeraire.schema import exposure_cell, read_jsonl
from numeraire.scoring import cluster_ols
from analyze_primary import CUTOFF, MODELS, frame, predictions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics"


def native_predictions(*folders):
    """Read one native-arm probability per observation, rejecting conflicting duplicates."""
    result = {}
    for folder in folders:
        path = ROOT / "results/hosted" / folder / "predictions.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row["arm"] != "N":
                continue
            if row["id"] in result and result[row["id"]] != row["probability"]:
                raise ValueError(f"Conflicting native forecasts for {folder}: {row['id']}")
            result[row["id"]] = row["probability"]
    return result


def performance(probabilities, labels):
    p, y = np.asarray(probabilities, float), np.asarray(labels, int)
    return {"n": len(y), "auc": auc(p, y), "brier": float(np.mean((p - y) ** 2))}


def matched_benchmarks():
    records = read_jsonl(ROOT / "data/processed/sec_confirmation.jsonl")
    by_id = {r["id"]: r for r in records}
    scored = walk_forward_predictions(records, lambda r: list(ratio_features(r).values()))
    comparison = pd.DataFrame(scored).rename(columns={"probability": "ratio_logistic"})
    ids = comparison.id.tolist()
    result = {
        "panel": "annual confirmation",
        "n": len(ids),
        "firms": len({by_id[i]["firm_id"] for i in ids}),
        "ratio_logistic": performance(comparison.ratio_logistic, comparison.label),
        "constant_half": performance(np.full(len(ids), 0.5), comparison.label),
        "models": {},
    }
    for model, prefix in MODELS.items():
        p = native_predictions(f"{prefix}_confirmation", f"{prefix}_confirmation_representations")
        missing = set(ids) - p.keys()
        if missing:
            raise ValueError(f"Missing matched forecasts for {model}: {len(missing)}")
        comparison[model] = [p[i] for i in ids]
        cells = [exposure_cell(by_id[i], CUTOFF[model]) for i in ids]
        result["models"][model] = {
            **performance(comparison[model], comparison.label),
            "exposure": {c: cells.count(c) for c in ("exposed", "donut", "unexposed")},
        }
    comparison.to_csv(OUT / "matched_annual_predictions.csv", index=False)
    result["paired_benchmark_minus_model"] = paired_differences(comparison, by_id)
    return result


def paired_differences(comparison, by_id, draws=2000, seed=2026):
    """Ratio benchmark minus each model: AUC with a firm bootstrap, Brier loss with firm-clustered CR1 intervals."""
    firm = comparison.id.map(lambda i: by_id[i]["firm_id"]).to_numpy()
    firms = pd.unique(firm)
    rng = np.random.default_rng(seed)
    index = {f: np.flatnonzero(firm == f) for f in firms}
    samples = [np.concatenate([index[f] for f in rng.choice(firms, len(firms))]) for _ in range(draws)]
    y = comparison.label.to_numpy()
    b = comparison.ratio_logistic.to_numpy()
    out = {}
    for model in MODELS:
        p = comparison[model].to_numpy()
        boot = [auc(b[i], y[i]) - auc(p[i], y[i]) for i in samples]
        gap = (p - y) ** 2 - (b - y) ** 2
        s = cluster_ols(gap, np.ones((len(gap), 1)), firm, comparison.id.to_numpy(), 0)
        out[model] = {"auc_difference": auc(b, y) - auc(p, y), "auc_low": float(np.percentile(boot, 2.5)),
                      "auc_high": float(np.percentile(boot, 97.5)), "brier_difference": s["estimate"],
                      "brier_low": s["firm_low"], "brier_high": s["firm_high"]}
    return out


def primary_diagnostics():
    result = {}
    for model in ("Llama 3.1 8B", "Llama 3.1 70B"):
        w = frame(predictions(MODELS[model]))
        w = w[w.cell_llama.isin(["exposed", "unexposed"])].copy()
        w["gap"] = (w.NC - w.label) ** 2 - (w.N_p1 - w.label) ** 2
        cells = {}
        for name in ("exposed", "unexposed"):
            c = w[w.cell_llama.eq(name)]
            cells[name] = {
                "n": len(c),
                "base_rate": float(c.label.mean()),
                "reference_mean": float(c.N.mean()),
                "calibration_in_the_large": float((c.N - c.label).mean()),
                "conversion_minus_header_shift": float((c.NC - c.N_p1).mean()),
                "conversion_minus_header_brier_gap": float(c.gap.mean()),
            }
        shift = float((w.NC - w.N_p1).mean())
        e, u = cells["exposed"], cells["unexposed"]
        rate_component = -2 * shift * (e["base_rate"] - u["base_rate"])
        forecast_component = 2 * shift * (e["reference_mean"] - u["reference_mean"])
        result[model] = {
            "cells": cells,
            "pooled_conversion_minus_header_shift": shift,
            "uniform_shift_base_rate_component": rate_component,
            "uniform_shift_reference_forecast_component": forecast_component,
            "uniform_shift_full_calibration_contribution": rate_component + forecast_component,
            "observed_contrast": e["conversion_minus_header_brier_gap"]
            - u["conversion_minus_header_brier_gap"],
            "directional_flips_against_native": {
                arm: float(((w[arm] >= 0.5) != (w.N >= 0.5)).mean())
                for arm in ("NC", "NP", "N_id", "N_p1", "N_p2")
            },
        }
    return result


def repeated_scoring():
    result = {}
    for model, prefix in MODELS.items():
        original = native_predictions(f"{prefix}_annual")
        if not original:
            original = native_predictions(f"{prefix}_annual_placebos")
        repeated = native_predictions(f"{prefix}_annual_N_repeat")
        if not original or set(original) != set(repeated):
            raise ValueError(f"Repeat-scoring panel mismatch for {model}")
        ids = sorted(original)
        a = np.array([original[i] for i in ids])
        b = np.array([repeated[i] for i in ids])
        result[model] = {
            "panel": "annual pilot",
            "n": len(ids),
            "mean_absolute_probability_difference": float(np.abs(a - b).mean()),
            "decision_flip_rate": float(np.mean((a >= 0.5) != (b >= 0.5))),
        }
    return result


def unit_arm_overlap():
    result = {}
    for branch in ("base", "planted-e60", "flipped-e60"):
        path = (
            ROOT / "results/planted_control/scores_representations" / branch / "predictions.jsonl"
        )
        rows = {(r["id"], r["arm"]): r for r in read_jsonl(path)}
        ids = sorted({i for i, a in rows if a == "N"})
        identical, differences = [], []
        for i in ids:
            n, u = rows[i, "N"], rows[i, "U"]
            n_hash = {s["swap"]: s["prompt_hash"] for s in n["scores"]}
            u_hash = {s["swap"]: s["prompt_hash"] for s in u["scores"]}
            identical.append(n_hash == u_hash)
            differences.append(abs(n["probability"] - u["probability"]))
        differences = np.array(differences)
        identical = np.array(identical)
        result[branch] = {
            "n": len(ids),
            "identical_prompt_pairs": int(identical.sum()),
            "mean_absolute_probability_difference": float(differences.mean()),
            "nonidentical_prompts_mean_absolute_probability_difference": float(
                differences[~identical].mean()
            ),
        }
    return result


def elicitation_auc_changes():
    table = pd.read_csv(ROOT / "results/forecast_diagnostics/elicitation.csv")
    result = []
    for (model, panel), group in table.groupby(["model", "panel"]):
        g = group.set_index("elicitation")
        if "ab_raw" not in g.index:
            continue
        for kind in g.index:
            if kind != "ab_raw":
                result.append(
                    {
                        "model": model,
                        "panel": panel,
                        "elicitation": kind,
                        "auc_change_against_ab_raw": float(
                            g.loc[kind, "auc_outcome"] - g.loc["ab_raw", "auc_outcome"]
                        ),
                    }
                )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_status": "Post hoc descriptive checks using saved outputs.",
        "matched_annual_benchmarks": matched_benchmarks(),
        "primary_mean_shift": primary_diagnostics(),
        "annual_pilot_repeat_scoring": repeated_scoring(),
        "planted_unit_arm_overlap": unit_arm_overlap(),
        "elicitation_auc_changes": elicitation_auc_changes(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
