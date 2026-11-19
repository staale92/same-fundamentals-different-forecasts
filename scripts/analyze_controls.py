#!/usr/bin/env python3
"""Temporal placebo splits, planted-control attenuation and mean-shift diagnostics.

Estimate drift within exposure cells, attenuation at each training dose,
the placebo-to-conversion ratio, and retention under alternative renderings.
All calculations use saved financial panels and model probabilities.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from numeraire.schema import read_jsonl
from numeraire.scoring import cluster_ols
from numeraire.robustness import wcr_interval, mde
from numeraire.validation import planted_validation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/controls"
CONTROL_DIR = ROOT / "results/planted_control"
QPRED = {
    "Llama 3.1 8B": "results/hosted/numeraire-llama31-8b_quarterly_confirmation/predictions.jsonl",
    "Llama 3.1 70B": "results/hosted/numeraire-llama31-70b_quarterly_confirmation/predictions.jsonl",
}
APRED = {
    "Llama 3.1 8B": ["results/hosted/numeraire-llama31-8b_confirmation/predictions.jsonl"],
    "Llama 3.1 70B": ["results/hosted/numeraire-llama31-70b_confirmation/predictions.jsonl"],
}
PILOT_Q = {
    "Llama 3.1 8B": [
        "results/llama31_8b_quarterly/predictions.jsonl",
        "results/llama31_8b_quarterly_placebos/predictions.jsonl",
    ],
    "Llama 3.1 70B": [
        "results/hosted/numeraire-llama31-70b_quarterly/predictions.jsonl",
        "results/hosted/numeraire-llama31-70b_quarterly_placebos/predictions.jsonl",
    ],
}
PILOT_A = {
    "Llama 3.1 8B": [
        "results/llama31_8b_native/predictions.jsonl",
        "results/llama31_8b_placebos/predictions.jsonl",
    ],
    "Llama 3.1 70B": [
        "results/hosted/numeraire-llama31-70b_annual/predictions.jsonl",
        "results/hosted/numeraire-llama31-70b_annual_placebos/predictions.jsonl",
    ],
}


def qindex(q):
    return int(q[:4]) + (int(q[-1]) - 1) / 4 + 0.125  # quarter midpoint in years


def frame(files, quarterly):
    rows = [r for f in files if (ROOT / f).exists() for r in read_jsonl(ROOT / f)]
    d = pd.DataFrame(rows)
    w = d.pivot_table(index="id", columns="arm", values="probability", aggfunc="first")
    meta = d.drop_duplicates("id").set_index("id")[
        ["firm_id", "label", "exposure", "outcome_year"] + (["outcome_period"] if quarterly else [])
    ]
    w = w.join(meta)
    w["t"] = w.outcome_period.map(qindex) if quarterly else w.outcome_year.astype(float) + 0.5
    return w


def contrast(f, first_mask):
    """theta of NC against N_p1 between two groups (first minus second), with WCR interval."""
    f = f.dropna(subset=["NC", "N_p1"]).copy()
    f["g"] = (f.NC - f.label) ** 2 - (f.N_p1 - f.label) ** 2
    x = first_mask.loc[f.index].to_numpy(float)
    s = cluster_ols(f.g, np.column_stack([np.ones(len(f)), x]), f.firm_id, f.t, 1)
    lo, hi = wcr_interval(f.g.to_numpy(), x, f.firm_id, s["estimate"], s["firm_se"])
    return {
        "estimate": s["estimate"],
        "wcr_low": lo,
        "wcr_high": hi,
        "mde80": mde(s["firm_se"], s["firms"] - 1),
        "n_first": int(x.sum()),
        "n_second": int(len(x) - x.sum()),
        "distance_years": float(f[x == 1].t.mean() - f[x == 0].t.mean()),
    }


def splits():
    out = {}
    for model in QPRED:
        res = []
        for label, w, quarterly in (
            (
                "confirmation, quarterly",
                frame(
                    [
                        QPRED[model],
                        QPRED[model].replace(
                            "_quarterly_confirmation/", "_quarterly_confirmation_boundary/"
                        ),
                    ],
                    True,
                ),
                True,
            ),
            ("pilot, quarterly", frame(PILOT_Q[model], True), True),
            ("confirmation, annual", frame(APRED[model], False), False),
            ("pilot, annual", frame(PILOT_A[model], False), False),
        ):
            e = w[w.exposure.isin(["exposed", "unexposed"])]
            if not quarterly:
                e = e[(e.outcome_year <= 2022) | e.outcome_year.isin([2024, 2025])]
            prim = contrast(e, e.exposure.eq("exposed"))
            ex = e[e.exposure.eq("exposed")]
            first_ex = ex.outcome_period < "2022Q3" if quarterly else ex.outcome_year <= 2016
            sx = contrast(ex, first_ex) if first_ex.nunique() == 2 else None
            un = e[e.exposure.eq("unexposed")]
            first_un = un.outcome_period < "2025Q1" if quarterly else un.outcome_year == 2024
            su = (
                contrast(un, first_un)
                if first_un.nunique() == 2 and un.firm_id.nunique() > 3
                else None
            )
            for kind, sp in (("exposed", sx), ("unexposed", su)):
                if sp is None:
                    continue
                scale = (
                    abs(prim["distance_years"] / sp["distance_years"])
                    if sp["distance_years"]
                    else np.nan
                )
                res.append(
                    {
                        "panel": label,
                        "period": kind,
                        **sp,
                        "primary_distance_years": abs(prim["distance_years"]),
                        "scale": scale,
                        "scaled": abs(sp["estimate"]) * scale,
                    }
                )
        out[model] = res
    return out


def planted_draws():
    assign = read_jsonl(CONTROL_DIR / "assignment/assignments.jsonl")
    base = read_jsonl(CONTROL_DIR / "scores/base/predictions.jsonl")
    res = {}
    for e in (60, 120, 240):
        preds = []
        for b in ("planted", "flipped"):
            post = read_jsonl(CONTROL_DIR / f"scores/{b}-e{e}/predictions.jsonl")
            preds += [{**r, "variant": b, "stage": "pre"} for r in base] + [
                {**r, "variant": b, "stage": "post"} for r in post
            ]
        rep = planted_validation(
            preds,
            assign,
            bootstrap=2000,
            seed=2026,
            metric="brier",
            control="flipped",
            trained="N",
            placebo="N_p1",
            return_draws=True,
        )
        dr = rep["_draws"]
        eff = {x["arm"]: x["randomized_effect"] for x in rep["effects"]}
        T = eff["N"]
        Td = dr["N"]
        out = {"T_N": T}
        for k in ("NC", "N_p1", "NP"):
            share_d = (Td - dr[k]) / Td
            out[k] = {
                "share": (T - eff[k]) / T,
                "share_low": float(np.nanpercentile(share_d, 2.5)),
                "share_high": float(np.nanpercentile(share_d, 97.5)),
            }
        for k in ("N_p1", "NP"):
            rho_d = (Td - dr[k]) / (Td - dr["NC"])
            out[f"rho_{k}"] = {
                "rho": (T - eff[k]) / (T - eff["NC"]),
                "low": float(np.nanpercentile(rho_d, 2.5)),
                "high": float(np.nanpercentile(rho_d, 97.5)),
            }
        res[e] = out
    return res


def representation_retention():
    d = CONTROL_DIR / "scores_representations"
    if not all(
        (d / f"{n}/predictions.jsonl").exists() for n in ("base", "planted-e60", "flipped-e60")
    ):
        return None
    assign = read_jsonl(CONTROL_DIR / "assignment/assignments.jsonl")
    base = read_jsonl(d / "base/predictions.jsonl")
    preds = []
    for b in ("planted", "flipped"):
        post = read_jsonl(d / f"{b}-e60/predictions.jsonl")
        preds += [{**r, "variant": b, "stage": "pre"} for r in base] + [
            {**r, "variant": b, "stage": "post"} for r in post
        ]
    rep = planted_validation(
        preds,
        assign,
        bootstrap=2000,
        seed=2026,
        metric="brier",
        control="flipped",
        trained="N",
        placebo="U",
        return_draws=True,
    )
    dr = rep["_draws"]
    eff = {x["arm"]: x for x in rep["effects"]}
    T = eff["N"]["randomized_effect"]
    out = {"T_N": T, "arms": {}}
    for k, e in eff.items():
        kept = e["randomized_effect"] / T
        kd = dr[k] / dr["N"]
        out["arms"][k] = {
            "T": e["randomized_effect"],
            "T_low": e["randomized_effect_interval"]["ci_low"],
            "T_high": e["randomized_effect_interval"]["ci_high"],
            "retained_share": kept,
            "retained_low": float(np.nanpercentile(kd, 2.5)),
            "retained_high": float(np.nanpercentile(kd, 97.5)),
        }
    old = {
        (r["id"], r["arm"]): r["probability"]
        for r in read_jsonl(CONTROL_DIR / "scores/planted-e60/predictions.jsonl")
        if r["arm"] == "N"
    }
    new = {
        (r["id"], r["arm"]): r["probability"]
        for r in read_jsonl(d / "planted-e60/predictions.jsonl")
        if r["arm"] == "N"
    }
    common = sorted(set(old) & set(new))
    out["reproduction_N"] = {
        "n": len(common),
        "max_abs_difference": float(max(abs(old[k] - new[k]) for k in common)),
        "decision_flips": float(np.mean([(old[k] >= 0.5) != (new[k] >= 0.5) for k in common])),
    }
    return out


def mean_shift_diagnostic():
    w = frame(PILOT_A["Llama 3.1 8B"] + ["results/llama31_8b/predictions.jsonl"], False)
    e = w[
        ((w.outcome_year <= 2022) & w.exposure.eq("exposed"))
        | (w.outcome_year.isin([2024, 2025]) & w.exposure.eq("unexposed"))
    ].dropna(subset=["N", "NC", "N_p1"])
    cells = {}
    for z, g in e.groupby(e.exposure.eq("exposed")):
        cells["exposed" if z else "unexposed"] = {
            "n": len(g),
            "base_rate": g.label.mean(),
            "mean_N": g.N.mean(),
            "calibration_in_large_N": g.N.mean() - g.label.mean(),
            "shift_NC_minus_Np1": (g.NC - g.N_p1).mean(),
        }
    s = np.mean([cells[c]["shift_NC_minus_Np1"] for c in cells])
    predicted = (
        2
        * s
        * (
            cells["exposed"]["calibration_in_large_N"]
            - cells["unexposed"]["calibration_in_large_N"]
        )
    )
    c = pd.read_csv(ROOT / "results/pilot/contrasts.csv")
    c = c[(c.model == "Llama 3.1 8B") & (c.contrast == "Primary: NC against N_p1")].set_index(
        "variant"
    )
    rows = {
        k: {
            "estimate": c.loc[v, "estimate"],
            "low": c.loc[v, "firm_low"],
            "high": c.loc[v, "firm_high"],
        }
        for k, v in (
            ("raw", "primary"),
            ("within Y=0", "within Y=0"),
            ("within Y=1", "within Y=1"),
            ("base-rate adjusted", "base-rate reweighted (mean of within-Y)"),
        )
    }
    return {
        "cells": cells,
        "mean_shift_NC_minus_Np1": s,
        "predicted_theta_uniform_shift": predicted,
        "contrasts": rows,
    }


def variance_and_confidence():
    out = {}
    for model, path in {
        **{f"{m} (quarterly confirmation)": p for m, p in QPRED.items()},
        **{f"{m} (annual confirmation)": p[0] for m, p in APRED.items()},
    }.items():
        panel = (
            "data/processed/sec_quarterly_confirmation.jsonl"
            if "quarterly" in model
            else "data/processed/sec_confirmation.jsonl"
        )
        recs = {r["id"]: r for r in read_jsonl(ROOT / panel)}
        addp = ROOT / path.replace("_quarterly_confirmation/", "_quarterly_confirmation_boundary/")
        d = pd.DataFrame(
            [
                r
                for r in read_jsonl(ROOT / path)
                + (read_jsonl(addp) if "_quarterly_confirmation/" in path and addp.exists() else [])
                if r["arm"] == "N"
            ]
        )
        up = np.array(
            [
                float(
                    float(recs[i]["values"]["eps_basic"][1])
                    > float(recs[i]["values"]["eps_basic"][0])
                )
                for i in d.id
            ]
        )
        lg = np.log(
            d.probability.clip(1e-6, 1 - 1e-6) / (1 - d.probability.clip(1e-6, 1 - 1e-6))
        ).to_numpy()
        X = np.column_stack([np.ones(len(up)), up])
        beta = np.linalg.lstsq(X, lg, rcond=None)[0]
        r2 = 1 - ((lg - X @ beta) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()
        ex = d[d.exposure == "exposed"]
        out[model] = {
            "r2_sign": float(r2),
            "median_distance_exposed": float((ex.probability - 0.5).abs().median()),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ld = planted_draws()
    report = {
        "rho": ld[60]["rho_N_p1"],
        "rho_NP": ld[60]["rho_NP"],
        "removal_by_dose": ld,
        "splits": splits(),
        "mean_shift_diagnostic": mean_shift_diagnostic(),
        "representation_retention": representation_retention(),
        "variance": variance_and_confidence(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
