#!/usr/bin/env python3
"""Elicitation checks on the primary quarterly panel (configs/elicitation_checks.json).

Instruction ablation: Llama 3.1 8B and 70B rescored on N, N_p1 and NC without the two knowledge-restricting system
sentences. The primary contrast is recomputed, the instruction's effect on N is measured, and kappa is the
exposed-minus-unexposed change in N's Brier loss from removing the instruction (suppressed recall predicts kappa < 0).
Generated reasoning: gpt-oss-120b reasons, then states a probability; cells use its own stated cutoff, as does the
single-token comparison from the saved primary-panel forecasts. Added after review; exploratory.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from numeraire.robustness import auc, mde, wcr_interval
from numeraire.schema import exposure_cell, read_jsonl
from numeraire.scoring import cluster_ols
from analyze_primary import MODELS, PANEL, frame, predictions, theta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/elicitation"
CONFIG = json.loads((ROOT / "configs/elicitation_checks.json").read_text())
GPT_CUTOFF = CONFIG["generated_reasoning"]["exposure_cutoff"]
YEN = re.compile(r"yen|JPY|¥", re.I)
DOLLAR = re.compile(r"USD|dollar", re.I)


def clean(t):
    return {k: v for k, v in t.items() if k != "_frame"}


def mean_ci(f, col):
    s = cluster_ols(f[col].to_numpy(float), np.ones((len(f), 1)), f.firm_id, f.period, 0)
    return {"mean": s["estimate"], "low": s["firm_low"], "high": s["firm_high"], "n": int(len(f))}


def contrast(f, col):
    """Exposed-minus-unexposed mean of f[col], WCR and firm CR1 intervals."""
    x = f.cell.eq("exposed").to_numpy(float)
    y = f[col].to_numpy(float)
    s = cluster_ols(y, np.column_stack([np.ones(len(f)), x]), f.firm_id, f.period, 1)
    lo, hi = wcr_interval(y, x, f.firm_id, s["estimate"], s["firm_se"])
    return {"estimate": s["estimate"], "wcr_low": lo, "wcr_high": hi, "firm_low": s["firm_low"],
            "firm_high": s["firm_high"], "mde80": mde(s["firm_se"], s["firms"] - 1)}


def moves(w):
    """Mean absolute change and direction flips of NC and N_p1 against N, overall and by cell."""
    f = w.dropna(subset=["N", "NC", "N_p1"]).copy()
    out = {}
    for arm in ("NC", "N_p1"):
        f[f"abs_{arm}"] = (f[arm] - f.N).abs()
        f[f"flip_{arm}"] = ((f[arm] >= 0.5) != (f.N >= 0.5)).astype(float)
        out[arm] = {"mean_abs_change": float(f[f"abs_{arm}"].mean()), "flip_share": float(f[f"flip_{arm}"].mean()),
                    "share_ge_0.1": float((f[f"abs_{arm}"] >= 0.1).mean())}
    f["abs_diff"] = f.abs_NC - f.abs_N_p1
    f["flip_diff"] = f.flip_NC - f.flip_N_p1
    out["conversion_minus_header_abs"] = mean_ci(f, "abs_diff")
    out["conversion_minus_header_flips"] = mean_ci(f, "flip_diff")
    for cell in ("exposed", "unexposed"):
        g = f[f.cell_llama.eq(cell)]
        out[f"flips_{cell}"] = {"NC": float(g.flip_NC.mean()), "N_p1": float(g.flip_N_p1.mean()), "n": int(len(g))}
    return out


def instruction_ablation():
    rep = {}
    for model in ("Llama 3.1 8B", "Llama 3.1 70B"):
        folder = MODELS[model]
        rows = read_jsonl(ROOT / f"results/hosted/{folder}_quarterly_confirmation_no_instruction/predictions.jsonl")
        new, old = frame(rows), frame(predictions(folder))
        e = {"predictions": len(rows), "theta_without_instruction": clean(theta(new, model)),
             "theta_with_instruction": clean(theta(old, model)), "moves_without_instruction": moves(new)}
        j = new[["N"]].join(old[["N", "label", "firm_id", "period", "cell_llama"]], lsuffix="_new", rsuffix="_old")
        j = j.dropna(subset=["N_new", "N_old"])
        j["shift"] = j.N_new - j.N_old
        j["flip"] = ((j.N_new >= 0.5) != (j.N_old >= 0.5)).astype(float)
        j["brier_change"] = (j.N_new - j.label) ** 2 - (j.N_old - j.label) ** 2
        cells = j[j.cell_llama.isin(["exposed", "unexposed"])].rename(columns={"cell_llama": "cell"})
        e["instruction_effect_N"] = {
            "n": int(len(j)), "mean_shift": mean_ci(j, "shift"), "mean_abs_shift": float(j["shift"].abs().mean()),
            "flip_share": float(j.flip.mean()), "brier_change": mean_ci(j, "brier_change"),
            "brier_with": float(((j.N_old - j.label) ** 2).mean()), "brier_without": float(((j.N_new - j.label) ** 2).mean()),
            "auc_with": auc(j.N_old, j.label), "auc_without": auc(j.N_new, j.label),
            "mean_forecast_with": float(j.N_old.mean()), "mean_forecast_without": float(j.N_new.mean()),
            "kappa_exposed_minus_unexposed": contrast(cells, "brier_change"),
        }
        rep[model] = e
    return rep


def own_cutoff(w):
    recs = {r["id"]: r for r in read_jsonl(PANEL)}
    w = w.copy()
    w["cell_llama"] = [exposure_cell(recs[i], GPT_CUTOFF) for i in w.index]  # theta() and moves() read this column
    return w, recs


def gptoss_single_token():
    w, _ = own_cutoff(frame(predictions(MODELS["gpt-oss-120b"])))
    return {"theta": clean(theta(w, "gpt-oss-120b")), "moves": moves(w)}


def eps_rose(record):
    v = record["values"].get("eps_basic")
    try:
        return float(float(v[1]) > float(v[0]))
    except (TypeError, ValueError, IndexError):
        return np.nan


def generated_reasoning():
    rows = read_jsonl(ROOT / "results/hosted/gptoss120b_quarterly_confirmation_reasoning/answers.jsonl")
    d = pd.DataFrame(rows)
    w, recs = own_cutoff(frame([r for r in rows if r.get("probability") is not None]))
    w["sign_up"] = [eps_rose(recs[i]) for i in w.index]
    n = w.dropna(subset=["N"])
    e = {"answers": len(d), "parse": d.parse.value_counts().to_dict(), "finish_reason": d.finish_reason.value_counts().to_dict(),
         "completion_tokens_mean": float(np.mean([u["completion_tokens"] for u in d.usage])),
         "theta": clean(theta(w, "gpt-oss-120b")), "moves": moves(w),
         "N": {"n": int(len(n)), "brier": float(((n.N - n.label) ** 2).mean()), "auc_y": auc(n.N, n.label),
               "auc_sign": auc(n.N, n.sign_up), "mean_forecast": float(n.N.mean()), "base_rate": float(n.label.mean()),
               "share_far": float(((n.N - 0.5).abs() >= 0.1).mean())}}
    for cell in ("exposed", "unexposed"):
        g = n[n.cell_llama.eq(cell)]
        e["N"][f"auc_y_{cell}"] = auc(g.N, g.label)
    nc = d[d.arm.eq("NC")]
    text = nc.reasoning_content.fillna("") + "\n" + nc.content.fillna("")
    e["NC_mentions"] = {"yen": float(text.map(lambda t: bool(YEN.search(t))).mean()),
                        "dollars": float(text.map(lambda t: bool(DOLLAR.search(t))).mean()), "n": int(len(nc))}
    return e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"analysis_status": "Added after review; exploratory. Declared before scoring.",
           "instruction_ablation": instruction_ablation(), "gptoss_single_token": gptoss_single_token(),
           "generated_reasoning": generated_reasoning()}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    for m, e in rep["instruction_ablation"].items():
        t = e["theta_without_instruction"]
        print(m, f"theta without instruction {t['estimate']:+.4f} [{t['wcr_low']:+.4f}, {t['wcr_high']:+.4f}]",
              f"kappa {e['instruction_effect_N']['kappa_exposed_minus_unexposed']['estimate']:+.4f}")
    t = rep["generated_reasoning"]["theta"]
    print("gpt-oss reasoning", f"theta {t['estimate']:+.4f} [{t['wcr_low']:+.4f}, {t['wcr_high']:+.4f}] mde {t['mde80']:.3f}")


if __name__ == "__main__":
    main()
