#!/usr/bin/env python3
"""Primary estimates and sensitivities from saved outputs.

Primary-panel inference (pooled estimate with WCR, firm-level sign-flip with its own estimate, two-way and
leave-one-quarter-out sensitivities), post-release, reported-liabilities and input-type sensitivities, scenario and
joint-bootstrap bounds, the hosted known-target probe and the calibrated primary contrast, the ranking benchmark
with retrospective and point-in-time ranks, mean-shift dispersion, and exposure counts.
"""

import json
import math
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from numeraire.schema import read_jsonl, exposure_cell
from numeraire.scoring import cluster_ols
from numeraire.robustness import auc, wcr_interval, sign_flip, within_firm_differences, mde
from numeraire.elicitation import up_logit
from numeraire.benchmark import ratio_features, walk_forward_predictions
from numeraire.transforms import _percentile, _rank_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/primary"
PANEL = ROOT / "data/processed/sec_quarterly_confirmation.jsonl"
MODELS = {
    "Llama 3.1 8B": "numeraire-llama31-8b",
    "Llama 3.1 70B": "numeraire-llama31-70b",
    "gpt-oss-120b": "gptoss120b",
    "Muse Glimmer 30B": "numeraire-muse-glimmer-30b",
}
CUTOFF = {
    "Llama 3.1 8B": "2023-12-31",
    "Llama 3.1 70B": "2023-12-31",
    "gpt-oss-120b": "2024-06-01",
    "Muse Glimmer 30B": "2026-01-04",
}
SIG = lambda x: 1 / (1 + np.exp(-x))


def predictions(folder):
    rows = []
    for d in (
        f"results/hosted/{folder}_quarterly_confirmation",
        f"results/hosted/{folder}_quarterly_confirmation_boundary",
    ):
        p = ROOT / d / "predictions.jsonl"
        if p.exists():
            rows += read_jsonl(p)
    return rows


def frame(rows, calibrate=None):
    """Wide frame of probabilities by arm; calibrate maps (task, kind, swap) to a content-free logit to subtract."""
    recs = {r["id"]: r for r in read_jsonl(PANEL)}
    d = {}
    for r in rows:
        if r["id"] not in recs:
            continue
        if calibrate is None:
            p = r["probability"]
        else:
            lg = [
                up_logit("ab", s["swap"], {"A": s["logp_A"], "B": s["logp_B"]})
                - calibrate[("forecast", "ab", s["swap"])]
                for s in r["scores"]
            ]
            p = float(SIG(np.mean(lg)))
        d.setdefault(r["id"], {})[r["arm"]] = p
    w = pd.DataFrame.from_dict(d, orient="index")
    meta = pd.DataFrame(
        [
            {
                "id": i,
                "firm_id": recs[i]["firm_id"],
                "label": recs[i]["label"],
                "period": recs[i]["provenance"]["outcome_period"],
                "input_kind": recs[i]["provenance"]["input_kind"],
                "cell_llama": exposure_cell(recs[i], "2023-12-31"),
                "derived_liabilities": any(
                    s.get("derived")
                    for s in recs[i]["provenance"]["item_sources"].get("liabilities", [])
                ),
            }
            for i in w.index
        ]
    ).set_index("id")
    return w.join(meta)


def theta(w, model, mask=None, arm="NC", ref="N_p1", unexposed_from=None):
    f = w.dropna(subset=[arm, ref]).copy()
    if mask is not None:
        f = f[mask.loc[f.index]]
    q = f.period.map(lambda s: int(s[:4]) + (int(s[-1]) - 1) / 4)
    f["cell"] = f.cell_llama  # Llama 3.1 stated cutoff (exposure_cell), the within-model design
    if unexposed_from:
        f.loc[f.cell.eq("unexposed") & (q < unexposed_from), "cell"] = "donut"
    f = f[f.cell.isin(["exposed", "unexposed"])]
    f["g"] = (f[arm] - f.label) ** 2 - (f[ref] - f.label) ** 2
    x = f.cell.eq("exposed").to_numpy(float)
    s = cluster_ols(f.g, np.column_stack([np.ones(len(f)), x]), f.firm_id, f.period, 1)
    lo, hi = wcr_interval(f.g.to_numpy(), x, f.firm_id, s["estimate"], s["firm_se"])
    out = {
        "estimate": s["estimate"],
        "wcr_low": lo,
        "wcr_high": hi,
        "firm_low": s["firm_low"],
        "firm_high": s["firm_high"],
        "two_way_low": s.get("two_way_low"),
        "two_way_high": s.get("two_way_high"),
        "mde80": mde(s["firm_se"], s["firms"] - 1),
        "mean_exposed": float(f.g[x == 1].mean()),
        "mean_unexposed": float(f.g[x == 0].mean()),
        "n_exposed": int(x.sum()),
        "n_unexposed": int(len(x) - x.sum()),
        "firms": int(f.firm_id.nunique()),
    }
    fd = within_firm_differences(f.rename(columns={"g": "gap", "cell": "exposure"}))
    sf = sign_flip(fd.to_numpy())
    # within_firm_differences returns d_f / G, so the firm-level estimate is their sum
    out["firm_level"] = {
        "estimate": float(sf["estimate"]),
        "firms": int(len(fd)),
        "signflip_p": sf["p_value"],
        "low": sf["low"],
        "high": sf["high"],
    }
    loo = []
    for qq in sorted(f.period.unique()):
        h = f[f.period != qq]
        xh = h.cell.eq("exposed").to_numpy(float)
        loo.append(
            cluster_ols(h.g, np.column_stack([np.ones(len(h)), xh]), h.firm_id, h.period, 1)[
                "estimate"
            ]
        )
    out["leave_one_quarter_out"] = [float(min(loo)), float(max(loo))]
    out["_frame"] = f
    return out


def primary_and_sensitivities():
    res = {}
    for model in ("Llama 3.1 8B", "Llama 3.1 70B"):
        w = frame(predictions(MODELS[model]))
        base = theta(w, model)
        f = base.pop("_frame")
        r = {
            "primary": base,
            "post_release_2025Q1": {
                k: v for k, v in theta(w, model, unexposed_from=2025.0).items() if k != "_frame"
            },
            "reported_liabilities_only": {
                k: v
                for k, v in theta(w, model, mask=~w.derived_liabilities).items()
                if k != "_frame"
            },
            "annual_inputs_Q1": {
                k: v
                for k, v in theta(w, model, mask=w.input_kind.eq("annual")).items()
                if k != "_frame"
            },
            "quarterly_inputs_Q2Q3": {
                k: v
                for k, v in theta(w, model, mask=w.input_kind.eq("quarter_yoy")).items()
                if k != "_frame"
            },
        }
        res[model] = r
    return res


def joint_bound(model, rho_draws, B=2000, seed=2026):
    """Bootstrap theta and the primary-panel splits jointly, drawing rho from the planted control."""
    w = frame(predictions(MODELS[model])).dropna(subset=["NC", "N_p1"])
    q = w.period.map(lambda s: int(s[:4]) + (int(s[-1]) - 1) / 4)
    w["g"] = (w.NC - w.label) ** 2 - (w.N_p1 - w.label) ** 2
    w = w[w.cell_llama.isin(["exposed", "unexposed"])]
    q = q.loc[w.index]
    w["grp"] = np.where(
        w.cell_llama.eq("exposed"),
        np.where(q < 2022.5, "e1", "e2"),
        np.where(q < 2025.0, "u1", "u2"),
    )
    firms = w.firm_id.unique()
    rng = np.random.default_rng(seed)
    by = {f: g for f, g in w.groupby("firm_id")}
    out = []
    for b in range(B):
        h = pd.concat([by[f] for f in rng.choice(firms, len(firms))])
        m = h.groupby("grp").g.mean()
        th = h[h.grp.isin(["e1", "e2"])].g.mean() - h[h.grp.isin(["u1", "u2"])].g.mean()
        split = max(abs(m["e1"] - m["e2"]), abs(m["u1"] - m["u2"]))
        rho = min(max(rho_draws[b % len(rho_draws)], 0.0), 0.95)
        out.append((th + split) / (1 - rho))
    return {"p95": float(np.percentile(out, 95)), "p50": float(np.median(out))}


def validation_and_calibration():
    out = {}
    recs = {r["id"]: r for r in read_jsonl(PANEL)}
    for model, folder in MODELS.items():
        p = ROOT / f"results/primary/elicitation_{folder}/scores.jsonl"
        if not p.exists():
            continue
        sc = read_jsonl(p)
        cf = {}
        for r in sc:
            if r["arm"] == "content_free":
                cf.setdefault((r["task"], r["kind"], r["swap"]), []).append(
                    up_logit("ab", r["swap"], r["logp"])
                )
        cf = {k: float(np.mean(v)) for k, v in cf.items()}
        known = {}
        for r in sc:
            if r["task"] != "known" or r["arm"] == "content_free":
                continue
            known.setdefault(r["id"], {})[r["swap"]] = up_logit("ab", r["swap"], r["logp"])
        ids = [i for i, v in known.items() if len(v) == 2]
        truth = np.array(
            [
                int(
                    Decimal(str(recs[i]["values"]["eps_basic"][1]))
                    > Decimal(str(recs[i]["values"]["eps_basic"][0]))
                )
                for i in ids
            ]
        )
        p_raw = np.array([np.mean([SIG(known[i][s]) for s in (False, True)]) for i in ids])
        p_cal = np.array(
            [
                SIG(np.mean([known[i][s] - cf[("known", "ab", s)] for s in (False, True)]))
                for i in ids
            ]
        )
        entry = {
            "n": len(ids),
            "known_auc": auc(p_raw, truth),
            "known_accuracy": float(np.mean((p_raw >= 0.5) == truth)),
            "known_auc_calibrated": auc(p_cal, truth),
            "known_accuracy_calibrated": float(np.mean((p_cal >= 0.5) == truth)),
            "content_free_logits": {str(k): v for k, v in cf.items()},
        }
        if model.startswith("Llama"):
            wc = frame(predictions(folder), calibrate=cf)
            t = theta(wc, model)
            t.pop("_frame")
            entry["primary_calibrated"] = {
                k: t[k] for k in ("estimate", "wcr_low", "wcr_high", "mde80")
            }
            entry["brier_N_raw"] = float(((frame(predictions(folder)).N - wc.label) ** 2).mean())
            entry["brier_N_calibrated"] = float(((wc.N - wc.label) ** 2).mean())
        out[model] = entry
    return out


def benchmark():
    recs = read_jsonl(ROOT / "data/processed/sec_confirmation.jsonl")
    items = sorted({k for r in recs for k in r["values"]})

    def ranks(r, pool):
        out = []
        for k in items:
            for j in (0, 1):
                v = r["values"].get(k)
                ref = sorted(
                    Decimal(str(x["values"][k][j]))
                    for x in pool
                    if k in x["values"] and _rank_key(x, j, k) == _rank_key(r, j, k)
                )
                pc = None if v is None else _percentile(ref, Decimal(str(v[j])))
                out.append(float("nan") if pc is None else float(pc))
        return out

    retro = {r["id"]: ranks(r, recs) for r in recs}
    pit = {
        r["id"]: ranks(r, [x for x in recs if x["input_filed"] <= r["input_filed"]]) for r in recs
    }

    def levels(r):
        return [
            (
                math.copysign(math.log1p(abs(float(r["values"][k][j]))), float(r["values"][k][j]))
                if k in r["values"]
                else float("nan")
            )
            for k in items
            for j in (0, 1)
        ]

    feats = {
        "ratios, full precision": lambda r: list(ratio_features(r).values()),
        "ratios and signed log levels": lambda r: list(ratio_features(r).values()) + levels(r),
        "ratios rounded to 0.1% (Q)": lambda r: [
            round(v, 3) if math.isfinite(v) else float("nan") for v in ratio_features(r).values()
        ],
        "retrospective ranks (R as scored)": lambda r: retro[r["id"]],
        "point-in-time ranks": lambda r: pit[r["id"]],
        "current EPS change only": lambda r: [
            float(
                Decimal(str(r["values"]["eps_basic"][1]))
                > Decimal(str(r["values"]["eps_basic"][0]))
            )
        ],
    }
    res = {}
    for name, fx in feats.items():
        preds = walk_forward_predictions(recs, fx)
        p = np.array([r["probability"] for r in preds])
        y = np.array([r["label"] for r in preds])
        res[name] = {"n": len(p), "auc": auc(p, y), "brier": float(np.mean((p - y) ** 2))}
    later = sum(
        any(
            x["input_filed"] > r["input_filed"] and x["input_year"] == r["input_year"] for x in recs
        )
        for r in recs
    )
    res["_retrospective_rank_observations"] = later
    return res


def shift_dispersion():
    rows = []
    for f in (
        "results/llama31_8b_native/predictions.jsonl",
        "results/llama31_8b_placebos/predictions.jsonl",
    ):
        rows += read_jsonl(ROOT / f)
    w = pd.DataFrame(rows).pivot_table(index="id", columns="arm", values="probability")
    d = (w.NC - w.N_p1).dropna()
    return {
        "mean": float(d.mean()),
        "sd": float(d.std()),
        "min": float(d.min()),
        "max": float(d.max()),
        "n": len(d),
    }


def exposure_counts():
    recs = read_jsonl(PANEL)
    out = {}
    for model, cut in CUTOFF.items():
        c = pd.Series([exposure_cell(r, cut) for r in recs]).value_counts().to_dict()
        out[model] = c
    return out


def rho_draws():
    from numeraire.validation import planted_validation

    CONTROL_DIR = ROOT / "results/planted_control"
    assign = read_jsonl(CONTROL_DIR / "assignment/assignments.jsonl")
    base = read_jsonl(CONTROL_DIR / "scores/base/predictions.jsonl")
    preds = []
    for b in ("planted", "flipped"):
        post = read_jsonl(CONTROL_DIR / f"scores/{b}-e60/predictions.jsonl")
        preds += [{**r, "variant": b, "stage": "pre"} for r in base] + [
            {**r, "variant": b, "stage": "post"} for r in post
        ]
    dr = planted_validation(
        preds,
        assign,
        bootstrap=2000,
        seed=2026,
        metric="brier",
        control="flipped",
        trained="N",
        placebo="N_p1",
        return_draws=True,
    )["_draws"]
    return (dr["N"] - dr["N_p1"]) / (dr["N"] - dr["NC"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prim = primary_and_sensitivities()
    rd = rho_draws()
    joint = {m: joint_bound(m, rd) for m in ("Llama 3.1 8B", "Llama 3.1 70B")}
    rep = {
        "primary": prim,
        "joint_bound": joint,
        "validation": validation_and_calibration(),
        "benchmark": benchmark(),
        "shift_dispersion": shift_dispersion(),
        "exposure_counts": exposure_counts(),
        "panel_n": len(read_jsonl(PANEL)),
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {
                "primary": {
                    m: {
                        k: (v["estimate"], v.get("wcr_low"), v.get("wcr_high"))
                        for k, v in r.items()
                    }
                    for m, r in prim.items()
                },
                "joint": joint,
                "benchmark": {k: v for k, v in rep["benchmark"].items()},
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
