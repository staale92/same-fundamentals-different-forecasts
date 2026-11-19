#!/usr/bin/env python3
"""Forecast validation and benchmark analyses from saved outputs only.

1. Validity probe: AUC of the known-target probe (A/B and UP/DOWN).
2. Elicitation: raw and content-free-calibrated A/B and UP/DOWN forecasts
   (AUC, context disagreement, share at least 0.1 from the threshold), and the
   primary contrast under each elicitation.
3. Extrapolation: forecasts against the current EPS change, outcome
   persistence, the extrapolation benchmark, and memory beyond extrapolation.
4. Information cost of R and Q in the walk-forward logistic benchmark.
5. Holm-adjusted placebo family.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from numeraire.schema import read_jsonl, exposure_cell
from numeraire.scoring import cluster_ols
from numeraire.robustness import auc, wild_cluster_restricted, wcr_interval, mde
from numeraire.elicitation import up_logit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/forecast_diagnostics"
PANELS = {
    "pilot": "data/processed/sec_pilot.jsonl",
    "quarterly": "data/processed/sec_quarterly.jsonl",
    "confirmation": "data/processed/sec_confirmation.jsonl",
    "quarterly_confirmation": "data/processed/sec_quarterly_confirmation.jsonl",
}
CUTOFF = {
    "Llama 3.1 8B": "2023-12-31",
    "Llama 3.1 70B": "2023-12-31",
    "Muse Glimmer 30B": "2026-01-04",
    "gpt-oss-120b": "2024-06-01",
}
PROBES = {
    ("Llama 3.1 8B", "pilot"): "llama31_8b_pilot",
    ("Llama 3.1 8B", "quarterly"): "llama31_8b_quarterly",
    ("Muse Glimmer 30B", "pilot"): "muse_glimmer_30b_pilot",
}
AB_RUNS = {
    ("Llama 3.1 8B", "pilot"): [
        "results/llama31_8b/predictions.jsonl",
        "results/llama31_8b_native/predictions.jsonl",
        "results/llama31_8b_placebos/predictions.jsonl",
    ],
    ("Llama 3.1 8B", "quarterly"): [
        "results/llama31_8b_quarterly/predictions.jsonl",
        "results/llama31_8b_quarterly_placebos/predictions.jsonl",
    ],
    ("Muse Glimmer 30B", "pilot"): [
        "results/muse_glimmer_30b/predictions.jsonl",
        "results/muse_glimmer_30b_native/predictions.jsonl",
        "results/muse_glimmer_30b_placebos/predictions.jsonl",
    ],
}
N_RUNS = {  # every saved as-filed forecast, for the extrapolation analysis
    ("Llama 3.1 8B (local)", "pilot"): "results/llama31_8b_native/predictions.jsonl",
    ("Muse Glimmer 30B (local)", "pilot"): "results/muse_glimmer_30b_native/predictions.jsonl",
    ("Llama 3.1 70B", "pilot"): "results/hosted/numeraire-llama31-70b_annual/predictions.jsonl",
    ("gpt-oss-120b", "pilot"): "results/hosted/gptoss120b_annual/predictions.jsonl",
    ("Llama 3.1 8B (local)", "quarterly"): "results/llama31_8b_quarterly/predictions.jsonl",
    (
        "Llama 3.1 70B",
        "quarterly",
    ): "results/hosted/numeraire-llama31-70b_quarterly/predictions.jsonl",
    ("gpt-oss-120b", "quarterly"): "results/hosted/gptoss120b_quarterly/predictions.jsonl",
    (
        "Muse Glimmer 30B",
        "quarterly",
    ): "results/hosted/numeraire-muse-glimmer-30b_quarterly_placebos/predictions.jsonl",
    (
        "Llama 3.1 8B (hosted)",
        "confirmation",
    ): "results/hosted/numeraire-llama31-8b_confirmation/predictions.jsonl",
    (
        "Llama 3.1 70B",
        "confirmation",
    ): "results/hosted/numeraire-llama31-70b_confirmation/predictions.jsonl",
    (
        "Muse Glimmer 30B",
        "confirmation",
    ): "results/hosted/numeraire-muse-glimmer-30b_confirmation_representations/predictions.jsonl",
    (
        "gpt-oss-120b",
        "confirmation",
    ): "results/hosted/gptoss120b_confirmation_representations/predictions.jsonl",
    (
        "Llama 3.1 8B (hosted)",
        "quarterly_confirmation",
    ): "results/hosted/numeraire-llama31-8b_quarterly_confirmation/predictions.jsonl",
    (
        "Llama 3.1 70B",
        "quarterly_confirmation",
    ): "results/hosted/numeraire-llama31-70b_quarterly_confirmation/predictions.jsonl",
    (
        "gpt-oss-120b",
        "quarterly_confirmation",
    ): "results/hosted/gptoss120b_quarterly_confirmation/predictions.jsonl",
    (
        "Muse Glimmer 30B",
        "quarterly_confirmation",
    ): "results/hosted/numeraire-muse-glimmer-30b_quarterly_confirmation/predictions.jsonl",
}


def sig(x):
    return 1 / (1 + np.exp(-x))


def panel(name):
    return {r["id"]: r for r in read_jsonl(ROOT / PANELS[name])}


def cur_change(r):
    a, b = map(float, r["values"]["eps_basic"])
    return b - a


def contexts(rows, cf):
    """Per (id, arm): the two context logits (raw and calibrated). cf maps (task, kind, swap) to the content-free logit."""
    out = {}
    for r in rows:
        lp = r["logp"] if "logp" in r else None
        for sc in ([r] if lp is not None else r["scores"]):
            kind = r.get("kind", "ab")
            task = r.get("task", "forecast")
            swap = sc["swap"]
            logp = sc["logp"] if "logp" in sc else {"A": sc["logp_A"], "B": sc["logp_B"]}
            u = up_logit(kind, swap, logp)
            out.setdefault((r["id"], r["arm"]), {})[swap] = (u, u - cf.get((task, kind, swap), 0.0))
    return out


def summarize(ctx, truth, kind, disagreement_arm=None):
    """Average probabilities for ab_raw, logits otherwise; optionally scope disagreement to one arm."""
    p = {}
    dis = []
    for key, c in ctx.items():
        if len(c) != 2:
            continue
        i = 0 if kind.endswith("raw") else 1
        if kind == "ab_raw":
            p[key] = float(np.mean([sig(c[s][0]) for s in (False, True)]))
        else:
            p[key] = float(sig(np.mean([c[s][i] for s in (False, True)])))
        if disagreement_arm is None or key[1] == disagreement_arm:
            dis.append(abs(sig(c[False][i]) - sig(c[True][i])))
    return p, float(np.mean(dis)) if dis else None


def cf_logits(scores):
    cf = {}
    for r in scores:
        if r["arm"] != "content_free":
            continue
        cf.setdefault((r["task"], r["kind"], r["swap"]), []).append(
            up_logit(r["kind"], r["swap"], r["logp"])
        )
    return {k: float(np.mean(v)) for k, v in cf.items()}


def within_theta(p, arm, ref, records, cutoff, frequency):
    """theta = exposed-minus-unexposed Brier gap of arm over ref (declared windows)."""
    rows = []
    for (i, a), v in p.items():
        if a != arm or (i, ref) not in p:
            continue
        r = records[i]
        cell = exposure_cell(r, cutoff)
        if frequency == "annual":
            cell = (
                "exposed"
                if r["outcome_year"] <= 2022
                else "unexposed" if r["outcome_year"] >= 2024 else "donut"
            )
        if cell not in ("exposed", "unexposed"):
            continue
        y = r["label"]
        rows.append(
            {
                "firm": r["firm_id"],
                "period": r["outcome_year"],
                "x": float(cell == "exposed"),
                "g": (v - y) ** 2 - (p[(i, ref)] - y) ** 2,
            }
        )
    f = pd.DataFrame(rows)
    if f.empty or f.x.nunique() < 2:
        return None
    s = cluster_ols(f.g, np.column_stack([np.ones(len(f)), f.x]), f.firm, f.period, 1)
    lo, hi = wcr_interval(f.g.to_numpy(), f.x.to_numpy(), f.firm, s["estimate"], s["firm_se"])
    return {
        "estimate": s["estimate"],
        "wcr_low": lo,
        "wcr_high": hi,
        "wcr_p": wild_cluster_restricted(f.g, f.x.to_numpy(), f.firm),
        "mde80": mde(s["firm_se"], s["firms"] - 1),
        "n": len(f),
    }


def validity_and_elicitation():
    val, eli, prim = [], [], []
    for (model, pname), d in PROBES.items():
        path = OUT / d / "scores.jsonl"
        if not path.exists():
            continue
        scores = read_jsonl(path)
        recs = panel(pname)
        cf = cf_logits(scores)
        cur = {i: int(cur_change(r) > 0) for i, r in recs.items()}
        for kind in ("ab", "word"):
            rows = [
                r
                for r in scores
                if r["task"] == "known" and r["kind"] == kind and r["arm"] != "content_free"
            ]
            ctx = contexts(rows, cf)
            for variant in (f"{kind}_raw", f"{kind}_cal"):
                p, dis = summarize(
                    ctx, cur, variant if kind == "ab" else variant.replace("word", "w")
                )
                ids = [k[0] for k in p]
                val.append(
                    {
                        "model": model,
                        "panel": pname,
                        "elicitation": variant,
                        "n": len(p),
                        "auc_known": auc(
                            np.array(list(p.values())), np.array([cur[i] for i in ids])
                        ),
                        "accuracy_known": float(
                            np.mean([(v >= 0.5) == bool(cur[i]) for (i, _), v in p.items()])
                        ),
                        "context_disagreement": dis,
                    }
                )
        # Forecasts on N: A/B from the saved forecasts, UP/DOWN from the probe.
        ab = [
            r for f in AB_RUNS[(model, pname)] if (ROOT / f).exists() for r in read_jsonl(ROOT / f)
        ]
        word = [
            r
            for r in scores
            if r["task"] == "forecast" and r["kind"] == "word" and r["arm"] != "content_free"
        ]
        freq = "annual" if pname == "pilot" else "quarterly"
        y = {i: r["label"] for i, r in recs.items()}
        for kind, rows in (("ab", ab), ("word", word)):
            ctx = contexts(rows, cf)
            for variant in (f"{kind}_raw", f"{kind}_cal"):
                p, dis = summarize(
                    ctx,
                    y,
                    variant if kind == "ab" else variant.replace("word", "w"),
                    disagreement_arm="N",
                )
                pn = {k[0]: v for k, v in p.items() if k[1] == "N"}
                if not pn:
                    continue
                ids = list(pn)
                pv = np.array([pn[i] for i in ids])
                eli.append(
                    {
                        "model": model,
                        "panel": pname,
                        "elicitation": variant,
                        "n": len(ids),
                        "auc_outcome": auc(pv, np.array([y[i] for i in ids])),
                        "auc_current_change": auc(pv, np.array([cur[i] for i in ids])),
                        "share_far": float(np.mean(abs(pv - 0.5) >= 0.1)),
                        "brier": float(np.mean((pv - np.array([y[i] for i in ids])) ** 2)),
                        "context_disagreement": dis,
                    }
                )
                for arm, ref in (("NC", "N_p1"), ("NP", "N_p1")):
                    t = (
                        within_theta(p, arm, ref, recs, CUTOFF[model], freq)
                        if model != "Muse Glimmer 30B"
                        else None
                    )
                    if t:
                        prim.append(
                            {
                                "model": model,
                                "panel": pname,
                                "elicitation": variant,
                                "contrast": f"{arm} against {ref}",
                                **t,
                            }
                        )
    return pd.DataFrame(val), pd.DataFrame(eli), pd.DataFrame(prim)


def extrapolation():
    rows, reg, beyond = [], [], []
    for (model, pname), f in N_RUNS.items():
        if not (ROOT / f).exists():
            continue
        recs = panel(pname)
        add = ROOT / f.replace("_quarterly_confirmation/", "_quarterly_confirmation_boundary/")
        pred = [
            r
            for r in read_jsonl(ROOT / f)
            + (read_jsonl(add) if "_quarterly_confirmation/" in f and add.exists() else [])
            if r["arm"] == "N"
        ]
        if not pred:
            continue
        d = pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "firm": recs[r["id"]]["firm_id"],
                    "period": recs[r["id"]]["outcome_year"],
                    "p": r["probability"],
                    "y": recs[r["id"]]["label"],
                    "chg": cur_change(recs[r["id"]]),
                    "exposure": r.get("exposure"),
                }
                for r in pred
            ]
        )
        d["up"] = (d.chg > 0).astype(int)
        d["slog"] = np.sign(d.chg) * np.log1p(d.chg.abs())
        d["logit"] = np.log(d.p.clip(1e-6, 1 - 1e-6) / (1 - d.p.clip(1e-6, 1 - 1e-6)))
        X = np.column_stack([np.ones(len(d)), d.up, d.slog])
        b_up = cluster_ols(d.logit, X, d.firm, d.period, 1)
        b_s = cluster_ols(d.logit, X, d.firm, d.period, 2)
        rows.append(
            {
                "model": model,
                "panel": pname,
                "n": len(d),
                "auc_outcome": auc(d.p, d.y),
                "auc_current_change": auc(d.p, d.up),
                "p_up_given_current_up": d[d.up == 1].y.mean(),
                "p_up_given_current_down": d[d.up == 0].y.mean(),
                "extrapolation_benchmark_auc": auc(d.up.astype(float), d.y),
                "share_far": float(np.mean(abs(d.p - 0.5) >= 0.1)),
                "mean_p_current_up": d[d.up == 1].p.mean(),
                "mean_p_current_down": d[d.up == 0].p.mean(),
            }
        )
        reg.append(
            {
                "model": model,
                "panel": pname,
                "coef_current_up": b_up.get("estimate"),
                "coef_current_up_low": b_up.get("firm_low"),
                "coef_current_up_high": b_up.get("firm_high"),
                "coef_signed_log_change": b_s.get("estimate"),
                "coef_signed_log_change_low": b_s.get("firm_low"),
                "coef_signed_log_change_high": b_s.get("firm_high"),
            }
        )
        e = d[d.exposure.isin(["exposed", "unexposed"])]
        if e.exposure.nunique() == 2:

            def strat(h):
                parts = [(len(g), auc(g.p, g.y)) for _, g in h.groupby("up") if g.y.nunique() == 2]
                return sum(n * a for n, a in parts) / sum(n for n, _ in parts) if parts else np.nan

            point = strat(e[e.exposure == "exposed"]) - strat(e[e.exposure == "unexposed"])
            firms = e.firm.unique()
            rng = np.random.default_rng(2026)
            draws = []
            groups = {f_: e[e.firm == f_] for f_ in firms}
            for _ in range(1000):
                b_ = pd.concat([groups[f_] for f_ in rng.choice(firms, len(firms))])
                if b_.exposure.nunique() == 2:
                    draws.append(
                        strat(b_[b_.exposure == "exposed"]) - strat(b_[b_.exposure == "unexposed"])
                    )
            beyond.append(
                {
                    "model": model,
                    "panel": pname,
                    "difference": point,
                    "low": float(np.nanpercentile(draws, 2.5)),
                    "high": float(np.nanpercentile(draws, 97.5)),
                    "n_exposed": int((e.exposure == "exposed").sum()),
                    "n_unexposed": int((e.exposure == "unexposed").sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(reg), pd.DataFrame(beyond)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


PLACEBO_RUNS = {  # production A/B forecasts with N_p1, NP, N_p2 and N_id
    ("Llama 3.1 8B (local)", "pilot"): AB_RUNS[("Llama 3.1 8B", "pilot")],
    ("Llama 3.1 8B (local)", "quarterly"): AB_RUNS[("Llama 3.1 8B", "quarterly")],
    ("Llama 3.1 8B (hosted)", "confirmation"): [
        "results/hosted/numeraire-llama31-8b_confirmation/predictions.jsonl"
    ],
    ("Llama 3.1 70B", "confirmation"): [
        "results/hosted/numeraire-llama31-70b_confirmation/predictions.jsonl"
    ],
    ("Llama 3.1 8B (hosted)", "quarterly_confirmation"): [
        "results/hosted/numeraire-llama31-8b_quarterly_confirmation/predictions.jsonl",
        "results/hosted/numeraire-llama31-8b_quarterly_confirmation_boundary/predictions.jsonl",
    ],
    ("Llama 3.1 70B", "quarterly_confirmation"): [
        "results/hosted/numeraire-llama31-70b_quarterly_confirmation/predictions.jsonl",
        "results/hosted/numeraire-llama31-70b_quarterly_confirmation_boundary/predictions.jsonl",
    ],
}


def placebo_family():
    out = []
    for (model, pname), files in PLACEBO_RUNS.items():
        rows = [r for f in files if (ROOT / f).exists() for r in read_jsonl(ROOT / f)]
        if not rows:
            continue
        recs = panel(pname)
        p = {(r["id"], r["arm"]): r["probability"] for r in rows}
        freq = "annual" if pname in ("pilot", "confirmation") else "quarterly"
        fam = []
        for arm in ("NP", "N_p2", "N_id"):
            t = within_theta(p, arm, "N_p1", recs, "2023-12-31", freq)
            if t:
                fam.append({"model": model, "panel": pname, "contrast": f"{arm} against N_p1", **t})
        if fam:
            adj = holm(np.array([f["wcr_p"] for f in fam]))
            for f, a in zip(fam, adj):
                f["holm_p"] = float(a)
            out += fam
    return pd.DataFrame(out)


def nodate():
    """Llama 3.1 8B with the template's Today Date line removed, against the standard template."""
    out = []
    for pname, std_files in (
        ("pilot", AB_RUNS[("Llama 3.1 8B", "pilot")]),
        ("quarterly", AB_RUNS[("Llama 3.1 8B", "quarterly")]),
    ):
        path = OUT / f"llama31_8b_{pname}_nodate/scores.jsonl"
        if not path.exists():
            continue
        rows = [
            r
            for r in read_jsonl(path)
            if r["arm"] != "content_free" and r["task"] == "forecast" and r["kind"] == "ab"
        ]
        ctx = contexts(rows, {})
        p, _ = summarize(ctx, None, "ab_raw")
        std = {
            (r["id"], r["arm"]): r["probability"]
            for f in std_files
            if (ROOT / f).exists()
            for r in read_jsonl(ROOT / f)
        }
        common = [k for k in p if k in std and k[1] == "N"]
        recs = panel(pname)
        freq = "annual" if pname == "pilot" else "quarterly"
        entry = {
            "panel": pname,
            "n": len(common),
            "decision_flips_N": float(np.mean([(p[k] >= 0.5) != (std[k] >= 0.5) for k in common])),
            "mean_abs_difference_N": float(np.mean([abs(p[k] - std[k]) for k in common])),
            "auc_N_nodate": auc(
                np.array([p[k] for k in common]), np.array([recs[k[0]]["label"] for k in common])
            ),
            "auc_N_standard": auc(
                np.array([std[k] for k in common]), np.array([recs[k[0]]["label"] for k in common])
            ),
        }
        for label, probs in (("nodate", p), ("standard", std)):
            t = within_theta(probs, "NC", "N_p1", recs, "2023-12-31", freq)
            if t:
                entry.update(
                    {
                        f"theta_NC_{label}": t["estimate"],
                        f"theta_NC_{label}_wcr": [t["wcr_low"], t["wcr_high"]],
                    }
                )
        out.append(entry)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val, eli, prim = validity_and_elicitation()
    val.to_csv(OUT / "validity.csv", index=False)
    eli.to_csv(OUT / "elicitation.csv", index=False)
    prim.to_csv(OUT / "elicitation_contrasts.csv", index=False)
    ext, reg, beyond = extrapolation()
    ext.to_csv(OUT / "extrapolation.csv", index=False)
    reg.to_csv(OUT / "extrapolation_regression.csv", index=False)
    beyond.to_csv(OUT / "memory_beyond_extrapolation.csv", index=False)
    info = {}
    fam = placebo_family()
    fam.to_csv(OUT / "placebo_family.csv", index=False)
    nd = nodate()
    gate = (
        {
            f"{r.model} {r.panel} {r.elicitation}": bool(r.auc_known >= 0.95)
            for r in val.itertuples()
        }
        if len(val)
        else {}
    )
    (OUT / "report.json").write_text(
        json.dumps(
            {"validity_gate_auc_0.95": gate, "information_cost": info, "template_nodate": nd},
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(json.dumps({"gate": gate, "information_cost": info}, indent=2, default=float))


if __name__ == "__main__":
    main()
