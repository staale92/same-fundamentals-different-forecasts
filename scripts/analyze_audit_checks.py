#!/usr/bin/env python3
"""Analyze temporal drift, alternative placebos, and firm prominence.

Also compare probability- and logit-based forecast aggregation and correlate
forecast logits with the known-target probe. Inputs are saved model outputs.
"""

import json

import numpy as np
import pandas as pd

from numeraire.elicitation import up_logit
from numeraire.schema import read_jsonl
from numeraire.scoring import cluster_ols
from analyze_primary import MODELS, OUT, PANEL, ROOT, SIG, frame, predictions, theta
from analyze_presentation import content_free

ROUT = OUT.parent / "audit_checks"
LL = ("Llama 3.1 8B", "Llama 3.1 70B")
RHO = json.loads((ROOT / "results/controls/report.json").read_text())["rho"]["rho"]


def gaps(w, arm="NC", ref="N_p1"):
    f = w.dropna(subset=[arm, ref]).copy()
    f["g"] = (f[arm] - f.label) ** 2 - (f[ref] - f.label) ** 2
    return f[f.cell_llama.isin(["exposed", "unexposed"])]


def diff(f, mask):
    """Mean of g where mask minus mean where not, clustered by firm."""
    x = mask.to_numpy(float)
    s = cluster_ols(
        f.g.to_numpy(),
        np.column_stack([np.ones(len(f)), x]),
        f.firm_id,
        f.period,
        1,
    )
    return {key: s[key] for key in ("estimate", "firm_low", "firm_high")}


def moving_splits(f):
    out = []
    for cell in ("exposed", "unexposed"):
        c = f[f.cell_llama == cell]
        qs = sorted(c.period.unique())
        for k in range(1, len(qs)):
            r = diff(c, c.period.isin(qs[:k]))
            r.update({"cell": cell, "split_after": qs[k - 1]})
            out.append(r)
    return out


def prominence(f):
    selection = json.loads((ROOT / "data/raw/sec_confirmation/selection.json").read_text())
    assets = {
        f'{int(d["cik"]):010d}': d["assets_CY2018Q4I"]
        for d in selection["candidates"]
    }
    firm_assets = pd.Series({k: assets.get(k) for k in f.firm_id.unique()}).dropna()
    top = set(firm_assets[firm_assets >= firm_assets.quantile(2 / 3)].index)
    big = f.firm_id.isin(top).to_numpy(float)
    ex = f.cell_llama.eq("exposed").to_numpy(float)
    X = np.column_stack([np.ones(len(f)), ex, big, ex * big])
    s = cluster_ols(f.g.to_numpy(), X, f.firm_id, f.period, 3)
    by = {}
    for name, mask in (("top_tercile", big == 1), ("bottom_two_terciles", big == 0)):
        g = f[mask]
        by[name] = diff(g, g.cell_llama.eq("exposed")) | {
            "firms": int(g.firm_id.nunique())
        }
    return {
        "triple_difference": {
            key: s[key] for key in ("estimate", "firm_low", "firm_high")
        },
        "theta_by_size": by,
        "missing_assets_firms": int(
            f.firm_id.map(assets).isna().groupby(f.firm_id).first().sum()
        ),
    }


def aggregation(rows, cf):
    recs = {r["id"]: r for r in read_jsonl(PANEL)}
    out = []
    for r in rows:
        if r["arm"] != "N" or r["id"] not in recs:
            continue
        lg = {
            s["swap"]: up_logit("ab", s["swap"], {"A": s["logp_A"], "B": s["logp_B"]})
            for s in r["scores"]
        }
        out.append(
            {
                "y": recs[r["id"]]["label"],
                "prob_avg": float(np.mean([SIG(v) for v in lg.values()])),
                "logit_avg": float(SIG(np.mean(list(lg.values())))),
                "cal_logit_avg": float(
                    SIG(np.mean([lg[s] - cf[("forecast", "ab", s)] for s in lg]))
                ),
                "cal_prob_avg": float(
                    np.mean([SIG(lg[s] - cf[("forecast", "ab", s)]) for s in lg])
                ),
            }
        )
    d = pd.DataFrame(out)
    return {
        k: {
            "brier": float(((d[k] - d.y) ** 2).mean()),
            "share_far": float(((d[k] - 0.5).abs() >= 0.1).mean()),
            "mean_forecast": float(d[k].mean()),
        }
        for k in ("prob_avg", "logit_avg", "cal_prob_avg", "cal_logit_avg")
    } | {"n": len(d)}


def probe_correlation(folder, rows):
    sc = read_jsonl(ROOT / f"results/primary/elicitation_{folder}/scores.jsonl")
    known = {}
    for r in sc:
        if r["task"] == "known" and r["arm"] != "content_free":
            known.setdefault(r["id"], {})[r["swap"]] = up_logit(
                "ab", r["swap"], r["logp"]
            )
    fc = {}
    for r in rows:
        if r["arm"] == "N":
            fc[r["id"]] = np.mean(
                [
                    up_logit("ab", s["swap"], {"A": s["logp_A"], "B": s["logp_B"]})
                    for s in r["scores"]
                ]
            )
    ids = [i for i in fc if i in known and len(known[i]) == 2]
    a = np.array([fc[i] for i in ids])
    b = np.array([np.mean(list(known[i].values())) for i in ids])
    return {
        "n": len(ids),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
    }


def main():
    rep = {}
    for model, folder in MODELS.items():
        rows = predictions(folder)
        w = frame(rows)
        e = {"probe_correlation": probe_correlation(folder, rows)}
        if model in LL:
            f = gaps(w)
            sp = moving_splits(f)
            M = max(abs(x["estimate"]) for x in sp)
            prim = theta(w, model)
            prim.pop("_frame")
            e["moving_splits"] = sp
            e["M_primary_panel"] = M
            e["bound_primary_panel"] = {
                "rho_0": prim["wcr_high"] + M,
                "rho_hat": (prim["wcr_high"] + M) / (1 - RHO),
            }
            alt = {}
            for arm, ref in (
                ("NC", "N_id"),
                ("NC", "NP"),
                ("N_p1", "N_id"),
                ("NP", "N_id"),
                ("N_p2", "N_id"),
            ):
                t = theta(w, model, arm=arm, ref=ref)
                alt[f"{arm} against {ref}"] = {
                    k: t[k]
                    for k in ("estimate", "wcr_low", "wcr_high", "firm_low", "firm_high")
                }
            e["alternative_placebos"] = alt
            e["prominence"] = prominence(f)
            e["aggregation"] = aggregation(rows, content_free(folder))
        rep[model] = e
    ROUT.mkdir(parents=True, exist_ok=True)
    (ROUT / "report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(f"Saved audit checks for {len(rep)} models to {ROUT.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
