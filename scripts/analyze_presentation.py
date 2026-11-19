"""Descriptive views of the primary panel from saved predictions: flip rates by arm with their
distance from the 0.5 threshold, the conversion-minus-placebo Brier gap by outcome quarter, and the exploratory
exploratory forecast-change magnitudes. No model is called."""

import json
import numpy as np
from numeraire.elicitation import up_logit
from numeraire.schema import read_jsonl
from numeraire.scoring import cluster_ols
from analyze_primary import MODELS, OUT, ROOT, frame, predictions

ROOT_OUT = OUT.parent / "presentation"
ARMS = ("NC", "N_p1", "N_p2", "NP", "N_id")


def flips(w):
    """Classify p >= 0.5 as UP, matching the other forecast analyses; retain tie-down sensitivity."""
    out = {}
    for a in ARMS:
        d = w.dropna(subset=["N", a])
        fl = (d.N >= 0.5) != (d[a] >= 0.5)
        far = fl & ((d.N - 0.5).abs() >= 0.1)
        out[a] = {
            "flip_rate": float(fl.mean()),
            "n": int(len(d)),
            "flips": int(fl.sum()),
            "share_of_flips_far": float(far.sum() / fl.sum()) if fl.sum() else 0.0,
            "far_flip_rate": float(far.mean()),
            "flip_rate_ties_down": float(((d.N > 0.5) != (d[a] > 0.5)).mean()),
            "reference_ties": int(d.N.eq(0.5).sum()),
            "arm_ties": int(d[a].eq(0.5).sum()),
        }
    return out


def by_quarter(w, arm="NC", ref="N_p1"):
    """Mean Brier gap (arm minus placebo; positive: conversion worse) by outcome quarter, normal 95% interval.
    Each firm contributes at most one observation per quarter, so the interval is clustered by firm.
    """
    d = w.dropna(subset=[arm, ref]).copy()
    d["g"] = (d[arm] - d.label) ** 2 - (d[ref] - d.label) ** 2
    rows = []
    for q, g in d.groupby("period"):
        assert g.firm_id.is_unique
        se = g.g.std(ddof=1) / np.sqrt(len(g))
        rows.append(
            {
                "period": q,
                "n": int(len(g)),
                "gap": float(g.g.mean()),
                "low": float(g.g.mean() - 1.96 * se),
                "high": float(g.g.mean() + 1.96 * se),
                "cell": g.cell_llama.iloc[0],
            }
        )
    return sorted(rows, key=lambda r: r["period"])


def content_free(folder):
    """Mean content-free logit by (task, kind, swap), for contextual calibration."""
    cf = {}
    for r in read_jsonl(ROOT / f"results/primary/elicitation_{folder}/scores.jsonl"):
        if r["arm"] == "content_free":
            cf.setdefault((r["task"], r["kind"], r["swap"]), []).append(
                up_logit("ab", r["swap"], r["logp"])
            )
    return {k: float(np.mean(v)) for k, v in cf.items()}


def magnitudes(w, arm="NC", placebo="N_p1", ref="N"):
    """Absolute probability change against the same as-filed forecast, for the arm and the placebo, and their paired difference."""
    d = w.dropna(subset=[ref, arm, placebo])
    a = (d[arm] - d[ref]).abs()
    b = (d[placebo] - d[ref]).abs()
    one = np.ones((len(d), 1))
    s = cluster_ols((a - b).to_numpy(), one, d.firm_id, d.period, 0)
    return {
        "n": int(len(d)),
        "firms": int(d.firm_id.nunique()),
        "mean_abs_arm": float(a.mean()),
        "mean_abs_placebo": float(b.mean()),
        "median_abs_arm": float(a.median()),
        "median_abs_placebo": float(b.median()),
        "share_ge_0.1_arm": float((a >= 0.1).mean()),
        "share_ge_0.1_placebo": float((b >= 0.1).mean()),
        "difference": s["estimate"],
        "firm_low": s["firm_low"],
        "firm_high": s["firm_high"],
        "two_way_low": s["two_way_low"],
        "two_way_high": s["two_way_high"],
    }


def main():
    rep, mag = {}, {}
    for model, folder in MODELS.items():
        w = frame(predictions(folder))
        rep[model] = {"flips": flips(w)}
        mag[model] = {"magnitudes": magnitudes(w)}
        if model.startswith("Llama"):
            rep[model]["by_quarter"] = by_quarter(w)
            mag[model]["magnitudes_calibrated"] = magnitudes(
                frame(predictions(folder), calibrate=content_free(folder))
            )
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    (ROOT_OUT / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    # Paired forecast-change magnitudes against the same reference forecast.
    (ROOT_OUT / "magnitudes.json").write_text(json.dumps(mag, indent=2) + "\n")
    print(json.dumps(mag, indent=1))


if __name__ == "__main__":
    main()
