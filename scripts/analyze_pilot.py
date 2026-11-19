"""Annual pilot analyses from saved forecasts."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from numeraire.schema import read_jsonl, Statement
from numeraire.scoring import cluster_ols
from numeraire.robustness import (
    auc,
    murphy,
    shift_decomposition,
    wild_cluster_restricted,
    wcr_interval,
    sign_flip,
    within_firm_differences,
    mde,
    breakdown,
)
from numeraire.inversion import score_inversion_lenient, TOLERANCES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pilot"
REFERENCE = {a: "N" for a in ("E", "U", "NP", "NC", "N_id", "N_p1", "N_p2")}


def wide(d, column="probability"):
    w = d.pivot(index="id", columns="arm", values=column)
    meta = d.drop_duplicates("id").set_index("id")[["firm_id", "label", "outcome_year", "exposure"]]
    return w.join(meta)


def discrimination(d, model):
    rows = []
    for arm, g in d.groupby("arm"):
        for cell, h in [("all", g)] + list(g.groupby("exposure")):
            y = h.label.to_numpy()
            p = h.probability.to_numpy()
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "cell": cell,
                    "n": len(h),
                    "base_rate": y.mean(),
                    "mean_p": p.mean(),
                    "accuracy": float(np.mean((p >= 0.5) == y)),
                    "auc": auc(p, y),
                    **murphy(p, y),
                }
            )
    return rows


def flips(d, model):
    w = wide(d)
    rows = []
    for arm, ref in REFERENCE.items():
        if arm not in w or ref not in w:
            continue
        m = w[[arm, ref, "label", "firm_id", "outcome_year"]].dropna()
        dist = (m[ref] - 0.5).abs()
        diff = (m[arm] - m.label) ** 2 - (m[ref] - m.label) ** 2
        firm = cluster_ols(diff, np.ones((len(m), 1)), m.firm_id, m.outcome_year, 0)
        bins = pd.cut(
            dist,
            [-1e-9, 0.025, 0.05, 0.1, 0.2, 1],
            labels=["<0.025", "0.025-0.05", "0.05-0.1", "0.1-0.2", ">=0.2"],
        )
        for b, h in m.groupby(bins, observed=True):
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "reference": ref,
                    "distance_bin": str(b),
                    "n": len(h),
                    "flip_rate": float(np.mean((h[arm] >= 0.5) != (h[ref] >= 0.5))),
                }
            )
        rows.append(
            {
                "model": model,
                "arm": arm,
                "reference": ref,
                "distance_bin": "all",
                "n": len(m),
                "flip_rate": float(np.mean((m[arm] >= 0.5) != (m[ref] >= 0.5))),
                "flip_rate_far": (
                    float(np.mean(((m[arm] >= 0.5) != (m[ref] >= 0.5))[dist >= 0.1]))
                    if (dist >= 0.1).any()
                    else None
                ),
                "n_far": int((dist >= 0.1).sum()),
                "mean_abs_shift": float((m[arm] - m[ref]).abs().mean()),
                "firm_low": firm.get("firm_low"),
                "firm_high": firm.get("firm_high"),
                **shift_decomposition(m[arm], m[ref], m.label),
            }
        )
    return rows


def contrast(frame, label, extras=True):
    x = frame.exposure.eq("exposed").to_numpy(float)
    stats = cluster_ols(
        frame.gap, np.column_stack([np.ones(len(frame)), x]), frame.firm_id, frame.outcome_year, 1
    )
    out = {
        "contrast": label,
        "n_exposed": int(x.sum()),
        "n_unexposed": int(len(x) - x.sum()),
        **{
            k: stats.get(k)
            for k in [
                "estimate",
                "firm_se",
                "firm_low",
                "firm_high",
                "period_low",
                "period_high",
                "two_way_low",
                "two_way_high",
                "firms",
            ]
        },
    }
    if stats.get("status") != "ok" or not extras:
        return out
    out["wcr_p"] = wild_cluster_restricted(frame.gap, x, frame.firm_id)
    out["wcr_low"], out["wcr_high"] = wcr_interval(
        frame.gap.to_numpy(), x, frame.firm_id, stats["estimate"], stats["firm_se"]
    )
    s = sign_flip(within_firm_differences(frame).to_numpy())
    out.update(
        {
            "signflip_estimate": s["estimate"],
            "signflip_p": s["p_value"],
            "signflip_low": s["low"],
            "signflip_high": s["high"],
            "signflip_firms": s["firms"],
        }
    )
    out["mde80_firm"] = mde(stats["firm_se"], stats["firms"] - 1)
    out.update(breakdown(stats["estimate"], stats["firm_low"]))
    return out


def inversion(model, folder, panel, sub="inversion"):
    rows = []
    path = ROOT / "results" / folder / sub / "probes.jsonl"
    if not path.exists():
        return []
    for r in read_jsonl(path):
        s = score_inversion_lenient(r["raw_text"], panel[r["id"]])
        rows.append(
            {
                "model": model,
                "arm": r["arm"],
                "kind": "prompt",
                "strict_parse": r["parse_success"],
                "lenient_parse": s["parse_success"],
            }
        )
        rows += [{"model": model, "arm": r["arm"], **f} for f in s["fields"]]
    d = pd.DataFrame(rows)
    out = []
    for arm, g in d.groupby("arm"):
        prompts = g[g.kind == "prompt"]
        fields = g[g.kind != "prompt"]
        out.append(
            {
                "model": model,
                "arm": arm,
                "kind": "prompts",
                "n": len(prompts),
                "strict_parse": int(prompts.strict_parse.sum()),
                "lenient_parse": int(prompts.lenient_parse.sum()),
            }
        )
        for field, h in fields.groupby("field"):
            out.append(
                {
                    "model": model,
                    "arm": arm,
                    "kind": "field",
                    "field": field,
                    "n": len(h),
                    "median_relative_error": h.relative_error.median(),
                    "median_unit_free_error": h.unit_free_relative_error.median(),
                    **{
                        f"unit_free_within_{t:g}": h[f"unit_free_within_{t:g}"].mean()
                        for t in TOLERANCES
                    },
                }
            )
        for kind, h in fields.groupby("kind"):
            out.append(
                {
                    "model": model,
                    "arm": arm,
                    "kind": kind,
                    "n": len(h),
                    "median_relative_error": h.relative_error.median(),
                    "median_unit_free_error": h.unit_free_relative_error.median(),
                    **{f"within_{t:g}": h[f"within_{t:g}"].mean() for t in TOLERANCES},
                    **{
                        f"unit_free_within_{t:g}": h[f"unit_free_within_{t:g}"].mean()
                        for t in TOLERANCES
                    },
                }
            )
    return out


def within_model(w, arm, ref, cutoff, model, label):
    if arm not in w or ref not in w:
        return [{"model": model, "contrast": label, "status": "pending"}]
    w = w.dropna(subset=[arm, ref]).copy()
    y = w.label
    w["gap"] = (w[arm] - y) ** 2 - (w[ref] - y) ** 2
    pre = (w.outcome_year <= 2022) & w.exposure.eq("exposed")
    post = w.outcome_year.isin([2024, 2025]) & w.exposure.eq("unexposed")
    f = w[pre | post].copy()
    base = contrast(f, label)
    rows = [{"model": model, "variant": "primary", **base}]
    # Base-rate robust: contrasts within each outcome, and their average.
    within = {}
    for v in (0, 1):
        h = f[f.label == v]
        within[v] = (
            h[h.exposure.eq("exposed")].gap.mean() - h[h.exposure.eq("unexposed")].gap.mean()
        )
        rows.append(
            {"model": model, "variant": f"within Y={v}", **contrast(h, label, extras=False)}
        )
    rng = np.random.default_rng(2026)
    firms = f.firm_id.unique()
    draws_rw = []
    draws_auc = []

    def auc_gap(h):
        return (auc(h[ref], h.label) or np.nan) - (auc(h[arm], h.label) or np.nan)

    def auc_theta(h):
        return auc_gap(h[h.exposure.eq("exposed")]) - auc_gap(h[h.exposure.eq("unexposed")])

    for _ in range(2000):
        h = pd.concat([f[f.firm_id == g] for g in rng.choice(firms, len(firms))])
        parts = [h[h.label == v] for v in (0, 1)]
        if all(p.exposure.nunique() == 2 for p in parts):
            draws_rw.append(
                np.mean(
                    [
                        p[p.exposure.eq("exposed")].gap.mean()
                        - p[p.exposure.eq("unexposed")].gap.mean()
                        for p in parts
                    ]
                )
            )
        a = auc_theta(h)
        if np.isfinite(a):
            draws_auc.append(a)
    rw = float(np.mean(list(within.values())))
    rows.append(
        {
            "model": model,
            "variant": "base-rate reweighted (mean of within-Y)",
            "contrast": label,
            "estimate": rw,
            "firm_low": float(np.percentile(draws_rw, 2.5)),
            "firm_high": float(np.percentile(draws_rw, 97.5)),
            "bootstrap_valid": len(draws_rw),
        }
    )
    rows.append(
        {
            "model": model,
            "variant": "rank-based (AUC_ref - AUC_arm)",
            "contrast": label,
            "estimate": auc_theta(f),
            "firm_low": float(np.percentile(draws_auc, 2.5)),
            "firm_high": float(np.percentile(draws_auc, 97.5)),
            "bootstrap_valid": len(draws_auc),
        }
    )
    rows.append(
        {
            "model": model,
            "variant": "drop FY2024 (post = FY2025)",
            **contrast(f[f.outcome_year != 2024], label, extras=False),
        }
    )
    # Placebo in time: a fake cutoff inside the exposed years (no exposure change).
    e = w[pre].copy()
    e["exposure"] = np.where(e.outcome_year <= 2016, "exposed", "unexposed")
    rows.append(
        {"model": model, "variant": "placebo-in-time FY2012-16 vs FY2017-22", **contrast(e, label)}
    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = {
        r["id"]: Statement.from_dict(r) for r in read_jsonl(ROOT / "data/processed/sec_pilot.jsonl")
    }
    disc, flip, con, inv = [], [], [], []
    for model, folder in [("Llama 3.1 8B", "llama31_8b"), ("Muse Glimmer 30B", "muse_glimmer_30b")]:
        rows = [
            r
            for suffix in ("", "_native", "_placebos")
            for r in read_jsonl(ROOT / f"results/{folder}{suffix}/predictions.jsonl")
        ]
        d = pd.DataFrame(rows).drop_duplicates(["id", "arm"])
        disc += discrimination(d, model)
        flip += flips(d, model)
        inv += inversion(model, folder, panel, "inversion_nc")
        if model == "Llama 3.1 8B":
            con += within_model(
                wide(d), "NC", "N_p1", "2023-12-31", model, "Primary: NC against N_p1"
            )
    for name, rows in [
        ("discrimination", disc),
        ("flips", flip),
        ("contrasts", con),
        ("inversion_lenient", inv),
    ]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "report.json").write_text(
        json.dumps(
            {"status": {"Llama 3.1 8B": "complete", "Muse Glimmer 30B": "complete"}}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
