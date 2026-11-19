#!/usr/bin/env python3
"""Quarterly within-model contrasts and supported staggered-cutoff rectangles.

The primary comparison is NC against N_p1 on the quarterly confirmation
panel. Secondary rectangles pair Llama 3.1 with gpt-oss-120b across periods
with different stated-cutoff exposure. All inputs are saved token scores.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from numeraire.schema import read_jsonl
from numeraire.scoring import cluster_ols
from numeraire.robustness import (
    auc,
    murphy,
    wild_cluster_restricted,
    wcr_interval,
    sign_flip,
    within_firm_differences,
    mde,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/quarterly"
RUNS = {
    "Llama 3.1 8B": [
        "results/llama31_8b_quarterly/predictions.jsonl",
        "results/llama31_8b_quarterly_placebos/predictions.jsonl",
    ],
    "Llama 3.1 70B": [
        "results/hosted/numeraire-llama31-70b_quarterly/predictions.jsonl",
        "results/hosted/numeraire-llama31-70b_quarterly_placebos/predictions.jsonl",
    ],
    "gpt-oss-120b": [
        "results/hosted/gptoss120b_quarterly/predictions.jsonl",
        "results/hosted/gptoss120b_quarterly_placebos/predictions.jsonl",
    ],
    "Llama 4 Maverick": [
        "results/hosted/numeraire-llama4-maverick_quarterly/predictions.jsonl",
        "results/hosted/numeraire-llama4-maverick_quarterly_placebos/predictions.jsonl",
    ],
    "Muse Glimmer 30B": [
        "results/hosted/numeraire-muse-glimmer-30b_quarterly_placebos/predictions.jsonl"
    ],
}
# Partner exposed quarters (unexposed for Llama 3.1) and post quarters unexposed for both.
# Rectangle specification: 2024Q1 with 2025Q1 and 2026Q1.
PARTNERS = {"gpt-oss-120b": {"primary": (["2024Q1"], ["2025Q1", "2026Q1"])}}
PAIRS = [("NC", "N_p1"), ("NC", "NP")]
EXPECTED = 184  # observations in the panel
# The quarterly confirmation panel contains 107 eligible firms; Maverick is excluded.
if "--confirmation" in sys.argv:
    OUT = ROOT / "results/quarterly_confirmation"
    RUNS = {
        m: [
            f"results/hosted/{d}_quarterly_confirmation/predictions.jsonl",
            f"results/hosted/{d}_quarterly_confirmation_boundary/predictions.jsonl",
        ]
        for m, d in [
            ("Llama 3.1 8B", "numeraire-llama31-8b"),
            ("Llama 3.1 70B", "numeraire-llama31-70b"),
            ("gpt-oss-120b", "gptoss120b"),
            ("Muse Glimmer 30B", "numeraire-muse-glimmer-30b"),
        ]
    }
    PAIRS = [("NC", "N_p1"), ("NC", "NP"), ("NP", "N_p1"), ("N_p2", "N_p1"), ("N_id", "N_p1")]
    EXPECTED = 1592  # EPS changes of at least USD 0.01, compared in decimal arithmetic


def frame(paths):
    parts = [pd.DataFrame(read_jsonl(ROOT / p)) for p in paths if (ROOT / p).exists()]
    if not parts:
        return None
    # Arms and observations can be split across files: stack every row, keep the first score of each (id, arm).
    allrows = pd.concat(parts).drop_duplicates(["id", "arm"])
    w = allrows.pivot(index="id", columns="arm", values="probability")
    meta = allrows.drop_duplicates("id").set_index("id")[
        ["firm_id", "label", "outcome_period", "exposure", "outcome_filed"]
    ]
    return w.join(meta)


def gap(w, arm, ref):
    return (w[arm] - w.label) ** 2 - (w[ref] - w.label) ** 2


def contrast(f):
    x = f.exposure.eq("exposed").to_numpy(float)
    s = cluster_ols(f.g, np.column_stack([np.ones(len(f)), x]), f.firm_id, f.outcome_period, 1)
    if s.get("status") != "ok":
        return {"status": s.get("status")}
    out = {
        k: s[k]
        for k in [
            "estimate",
            "firm_se",
            "firm_low",
            "firm_high",
            "period_low",
            "period_high",
            "firms",
            "periods",
        ]
    }
    out["wcr_p"] = wild_cluster_restricted(f.g, x, f.firm_id)
    out["wcr_low"], out["wcr_high"] = wcr_interval(
        f.g.to_numpy(), x, f.firm_id, s["estimate"], s["firm_se"]
    )
    sf = sign_flip(within_firm_differences(f.rename(columns={"g": "gap"})).to_numpy())
    out.update(
        {
            "signflip_p": sf["p_value"],
            "mde80": mde(s["firm_se"], s["firms"] - 1),
            "n_exposed": int(x.sum()),
            "n_unexposed": int(len(x) - x.sum()),
            "unexposed_periods": int(f[f.exposure.eq("unexposed")].outcome_period.nunique()),
        }
    )
    within = [
        f[(f.label == v) & f.exposure.eq("exposed")].g.mean()
        - f[(f.label == v) & f.exposure.eq("unexposed")].g.mean()
        for v in (0, 1)
    ]
    out["base_rate_adjusted"] = float(np.mean(within))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {m: frame(p) for m, p in RUNS.items()}
    status = {m: "missing" if f is None else f"{len(f)} items" for m, f in frames.items()}
    disc = []
    for m, w in frames.items():
        if w is None:
            continue
        for arm in [a for a in ["N", "NP", "NC", "E", "U", "C"] if a in w]:
            for cell, h in [("all", w)] + list(w.groupby("exposure")):
                h = h.dropna(subset=[arm])
                if len(h):
                    disc.append(
                        {
                            "model": m,
                            "arm": arm,
                            "cell": cell,
                            "n": len(h),
                            "base_rate": h.label.mean(),
                            "mean_p": h[arm].mean(),
                            "auc": auc(h[arm], h.label),
                            **murphy(h[arm], h.label),
                        }
                    )
    pd.DataFrame(disc).to_csv(OUT / "discrimination.csv", index=False)
    skill = {}
    for m, w in frames.items():
        if w is None or "N" not in w:
            continue
        h = w.dropna(subset=["N"])
        firms = h.firm_id.unique()
        rng = np.random.default_rng(2026)
        draws = {"all": [], "exposed": [], "unexposed": [], "difference": []}
        for _ in range(2000):
            b_ = pd.concat([h[h.firm_id == f] for f in rng.choice(firms, len(firms))])
            vals = {
                "all": auc(b_.N, b_.label),
                "exposed": auc(b_[b_.exposure == "exposed"].N, b_[b_.exposure == "exposed"].label),
                "unexposed": auc(
                    b_[b_.exposure == "unexposed"].N, b_[b_.exposure == "unexposed"].label
                ),
            }
            for k, v in vals.items():
                if v is not None:
                    draws[k].append(v)
            if vals["exposed"] is not None and vals["unexposed"] is not None:
                draws["difference"].append(vals["exposed"] - vals["unexposed"])
        point = {
            "all": auc(h.N, h.label),
            "exposed": auc(h[h.exposure == "exposed"].N, h[h.exposure == "exposed"].label),
            "unexposed": auc(h[h.exposure == "unexposed"].N, h[h.exposure == "unexposed"].label),
        }
        point["difference"] = (
            point["exposed"] - point["unexposed"]
            if None not in (point["exposed"], point["unexposed"])
            else None
        )
        skill[m] = {
            k: {
                "auc": point[k],
                "low": float(np.percentile(v, 2.5)) if v else None,
                "high": float(np.percentile(v, 97.5)) if v else None,
            }
            for k, v in draws.items()
        }
        skill[m]["brier_N"] = float(((h.N - h.label) ** 2).mean())
        skill[m]["n"] = len(h)
    rows = []
    for model in ["Llama 3.1 8B", "Llama 3.1 70B", "Muse Glimmer 30B"]:
        w = frames[model]
        if w is None or len(w) != EXPECTED:
            continue
        for arm, ref in PAIRS:
            if not {arm, ref} <= set(w.columns):
                continue
            f = w.assign(g=gap(w, arm, ref)).dropna(subset=["g"])
            f = f[f.exposure.isin(["exposed", "unexposed"])]
            rows.append(
                {
                    "analysis": f"{model} within-model",
                    "contrast": f"{arm} against {ref}",
                    **contrast(f),
                }
            )
            e = f[f.exposure.eq("exposed")].copy()
            e["exposure"] = np.where(e.outcome_period < "2022Q3", "exposed", "unexposed")
            rows.append(
                {
                    "analysis": f"{model} placebo in time (to 2022Q2 vs 2022Q3-2023Q3)",
                    "contrast": f"{arm} against {ref}",
                    **contrast(e),
                }
            )
    gradient = {}
    if frames.get("Llama 3.1 70B") is not None and frames.get("Llama 3.1 8B") is not None:
        for arm, ref in PAIRS:
            if not (
                {arm, ref} <= set(frames["Llama 3.1 70B"].columns)
                and {arm, ref} <= set(frames["Llama 3.1 8B"].columns)
            ):
                continue
            fb, fs = [
                w_.assign(g=gap(w_, arm, ref))[lambda x: x.exposure.isin(["exposed", "unexposed"])]
                for w_ in (frames["Llama 3.1 70B"], frames["Llama 3.1 8B"])
            ]
            t = (
                lambda f: f[f.exposure.eq("exposed")].g.mean()
                - f[f.exposure.eq("unexposed")].g.mean()
            )
            rng = np.random.default_rng(2026)
            firms = fb.firm_id.unique()
            draws = []
            for _ in range(2000):
                pick = rng.choice(firms, len(firms))
                draws.append(
                    t(pd.concat([fb[fb.firm_id == f_] for f_ in pick]))
                    - t(pd.concat([fs[fs.firm_id == f_] for f_ in pick]))
                )
            gradient[f"{arm} against {ref}"] = {
                "difference_70B_minus_8B": t(fb) - t(fs),
                "low": float(np.percentile(draws, 2.5)),
                "high": float(np.percentile(draws, 97.5)),
            }
    rect = {}

    def pool(g, l, pres, posts, arm, ref):
        def cell(w_, q):
            h = w_[w_.outcome_period == q]
            return gap(h, arm, ref).set_axis(h.firm_id).dropna()

        rects = []
        for pre in pres:
            for post in posts:
                cells = [cell(g, pre), cell(g, post), cell(l, pre), cell(l, post)]
                firms = sorted(set.intersection(*[set(c.index) for c in cells]))
                if len(firms) >= 2:
                    rects.append(
                        (
                            f"{pre}/{post}",
                            cells[0][firms] - cells[1][firms] - cells[2][firms] + cells[3][firms],
                        )
                    )
        if not rects:
            return None
        contributions, shares = {}, {}
        for _, q in rects:
            for f_, v in q.items():
                contributions[f_] = contributions.get(f_, 0) + v / len(q) / len(rects)
                shares[f_] = shares.get(f_, 0) + 1 / len(q) / len(rects)
        keys = sorted(contributions)
        sf = sign_flip([contributions[k] for k in keys], [shares[k] for k in keys])
        rng = np.random.default_rng(2026)
        draws = []
        for _ in range(2000):
            counts = pd.Series(rng.choice(keys, len(keys))).value_counts()
            est = [
                np.average(
                    q[counts.index.intersection(q.index)],
                    weights=counts[counts.index.intersection(q.index)],
                )
                for _, q in rects
                if len(counts.index.intersection(q.index))
            ]
            if len(est) == len(rects):
                draws.append(np.mean(est))
        return {
            "pairs": len(rects),
            "rectangles": [
                {"periods": k, "firms": len(q), "estimate": float(q.mean())} for k, q in rects
            ],
            "pooled": sf["estimate"],
            "signflip_p": sf["p_value"],
            "signflip_low": sf["low"],
            "signflip_high": sf["high"],
            "bootstrap_low": float(np.nanpercentile(draws, 2.5)) if draws else None,
            "bootstrap_high": float(np.nanpercentile(draws, 97.5)) if draws else None,
            "firms": sf["firms"],
        }

    audit = {}
    if frames.get("Llama 4 Maverick") is not None:
        q2 = frames["Llama 4 Maverick"][frames["Llama 4 Maverick"].outcome_period == "2024Q2"]
        audit["maverick_2024Q2_filing_dates"] = sorted(q2.outcome_filed.unique().tolist())
        audit["maverick_2024Q2_filed_in_august"] = sorted(
            q2[q2.outcome_filed >= "2024-08-01"].index.tolist()
        )
    for partner, sets in PARTNERS.items():
        g = frames.get(partner)
        for llama in ["Llama 3.1 8B", "Llama 3.1 70B"]:
            l = frames.get(llama)
            if g is None or l is None:
                continue
            for arm, ref in PAIRS:
                if not ({arm, ref} <= set(g.columns) and {arm, ref} <= set(l.columns)):
                    continue
                for kind, (pres, posts) in sets.items():
                    r = pool(g, l, pres, posts, arm, ref)
                    if r:
                        rect[f"{llama} x {partner}: {arm} against {ref} [{kind}]"] = {
                            "llama": llama,
                            "partner": partner,
                            "contrast": f"{arm} against {ref}",
                            "set": kind,
                            **r,
                        }
    pd.DataFrame(rows).to_csv(OUT / "contrasts.csv", index=False)
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "status": status,
                "skill_N": skill,
                "scale_gradient": gradient,
                "rectangles": rect,
                "audit": audit,
                "design": "single-exposed rectangles (partner, Llama 3.1) x (partner-exposed quarter, post quarter); Gamma = G_partner(pre) - G_partner(post) - G_llama(pre) + G_llama(post), d = 1",
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": status,
                "audit": audit,
                "rectangles": {
                    k: (round(v["pooled"], 4), v["signflip_p"], v["pairs"]) for k, v in rect.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
