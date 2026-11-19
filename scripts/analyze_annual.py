#!/usr/bin/env python3
"""Annual pilot and confirmation analyses from saved hosted-model forecasts.

Compute discrimination, directional flips, within-model Brier contrasts,
temporal placebos, and repeat-scoring variation in the reference arm.
The primary contrast is NC against N_p1; other renderings are sensitivities.
Pilot and confirmation results are reported separately.
"""

from pathlib import Path
import json
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
OUT = ROOT / "results/annual"
HOSTED = {
    "gpt-oss-120b": "gptoss120b",
    "Llama 3.1 70B": "numeraire-llama31-70b",
    "Llama 4 Maverick": "numeraire-llama4-maverick",
    "Llama 3.1 8B (hosted)": "numeraire-llama31-8b",
}
ARMS = ["E", "N", "NC", "NP", "N_id", "N_p1", "N_p2", "U"]
REFERENCE = {
    "N_id": "N",
    "NP": "N",
    "Q": "N",
    "E": "N",
    "N_p1": "N",
    "R": "N",
    "NC": "N",
    "N_p2": "N",
    "U": "N",
}
PAIRS = [
    ("NC", "N_p1"),
    ("NC", "NP"),
    ("NP", "N_p1"),
    ("R", "N_p1"),
    ("Q", "N_p1"),
    ("N_p2", "N_p1"),
    ("N_id", "N_p1"),
]


CONFIRMATION = {
    "Llama 3.1 8B (hosted)": "numeraire-llama31-8b",
    "Llama 3.1 70B": "numeraire-llama31-70b",
    "gpt-oss-120b": "gptoss120b",
    "Muse Glimmer 30B": "numeraire-muse-glimmer-30b",
}


def pivot(path):
    p = ROOT / path / "predictions.jsonl"
    if not p.exists():
        return None
    d = pd.DataFrame(read_jsonl(p))
    return d.pivot(index="id", columns="arm", values="probability").join(
        d.drop_duplicates("id").set_index("id")[["firm_id", "label", "outcome_year", "exposure"]]
    )


def combine(*frames):
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    out = frames[0].copy()
    for f in frames[1:]:
        new = [c for c in f.columns if c not in out.columns]
        out = out.join(f[new], how="outer")
    return out


def boot_auc(w, arm, draws=2000):
    rng = np.random.default_rng(2026)
    firms = w.firm_id.unique()
    out = []
    for _ in range(draws):
        h = pd.concat([w[w.firm_id == f] for f in rng.choice(firms, len(firms))])
        a = auc(h[arm], h.label)
        if a is not None:
            out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def within(w, arm, ref, window="own", drop2024=False):
    """Exposed-minus-unexposed gap contrast. window='llama': the declared Llama 3.1
    windows (exposed FY<=2022, unexposed FY2024-25); 'own': every eligible cell."""
    f = w[w.exposure.isin(["exposed", "unexposed"])].dropna(subset=[arm, ref]).copy()
    if window == "llama":
        f = f[
            ((f.outcome_year <= 2022) & f.exposure.eq("exposed"))
            | (
                f.outcome_year.isin([2025] if drop2024 else [2024, 2025])
                & f.exposure.eq("unexposed")
            )
        ]
    f["gap"] = (f[arm] - f.label) ** 2 - (f[ref] - f.label) ** 2
    x = f.exposure.eq("exposed").to_numpy(float)
    if len(f) == 0 or x.min() == x.max():
        return {"status": "unsupported: one exposure cell"}
    s = cluster_ols(f.gap, np.column_stack([np.ones(len(f)), x]), f.firm_id, f.outcome_year, 1)
    out = {k: s.get(k) for k in ["estimate", "firm_se", "firm_low", "firm_high", "firms"]}
    out["wcr_low"], out["wcr_high"] = wcr_interval(
        f.gap.to_numpy(), x, f.firm_id, s["estimate"], s["firm_se"]
    )
    out["wcr_p"] = wild_cluster_restricted(f.gap, x, f.firm_id)
    out["signflip_p"] = (
        sign_flip(within_firm_differences(f).to_numpy())["p_value"]
        if f.firm_id.nunique() <= 16
        else None
    )
    out["mde80"] = mde(s["firm_se"], s["firms"] - 1)
    out["n_exposed"], out["n_unexposed"] = int(x.sum()), int(len(x) - x.sum())
    out["unexposed_years"] = sorted(
        int(y) for y in f[f.exposure.eq("unexposed")].outcome_year.unique()
    )
    within_y = [
        f[(f.label == v) & f.exposure.eq("exposed")].gap.mean()
        - f[(f.label == v) & f.exposure.eq("unexposed")].gap.mean()
        for v in (0, 1)
    ]
    out["base_rate_adjusted"] = float(np.mean(within_y))
    e = f[f.exposure.eq("exposed")].copy()
    if e.outcome_year.nunique() > 3:
        e["exposure"] = np.where(e.outcome_year <= 2016, "exposed", "unexposed")
        xe = e.exposure.eq("exposed").to_numpy(float)
        if xe.min() != xe.max():
            pt = cluster_ols(
                e.gap, np.column_stack([np.ones(len(e)), xe]), e.firm_id, e.outcome_year, 1
            )
            out["placebo_in_time"] = {k: pt.get(k) for k in ["estimate", "firm_low", "firm_high"]}
    return out


def model_block(w, name, llama):
    window = "llama" if llama else "own"
    entry = {"status": "complete", "n": len(w)}
    for arm in ["N", "E"]:
        entry[f"auc_{arm}"] = auc(w[arm], w.label)
        entry[f"auc_{arm}_ci"] = boot_auc(w, arm)
        entry[f"murphy_{arm}"] = murphy(w[arm], w.label)
    entry["up_rate_N"] = float((w.N >= 0.5).mean())
    entry["flips"] = {
        arm: float(np.mean((w[arm] >= 0.5) != (w[ref] >= 0.5)))
        for arm, ref in REFERENCE.items()
        if arm in w and ref in w
    }
    entry["contrasts"] = {
        f"{a} against {r}": within(w, a, r, window) for a, r in PAIRS if a in w and r in w
    }
    if llama:
        entry["contrasts_drop2024"] = {
            f"{a} against {r}": within(w, a, r, window, drop2024=True)
            for a, r in PAIRS[:1]
            if a in w and r in w
        }
    return entry


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, folder in HOSTED.items():
        first = pivot(f"results/hosted/{folder}_annual")
        nuisance = pivot(f"results/hosted/{folder}_annual_placebos")
        w = combine(first, nuisance)
        if w is None or len(w) != 127 or not set(ARMS) <= set(w.columns):
            raise ValueError(f"Incomplete annual pilot: {name}")
        entry = model_block(w, name, "Llama 3.1" in name)
        repeat = pivot(f"results/hosted/{folder}_annual_N_repeat")
        if nuisance is not None and repeat is not None:
            common = repeat.index.intersection(nuisance.index)
            entry["serving_noise_N"] = {
                "decision_flips": float(
                    np.mean((repeat.loc[common, "N"] >= 0.5) != (nuisance.loc[common, "N"] >= 0.5))
                ),
                "mean_abs_difference": float(
                    (repeat.loc[common, "N"] - nuisance.loc[common, "N"]).abs().mean()
                ),
            }
        summary[name] = entry
    for name, folder in CONFIRMATION.items():
        conf = combine(
            pivot(f"results/hosted/{folder}_confirmation"),
            pivot(f"results/hosted/{folder}_confirmation_representations"),
        )
        need = ["N", "NP", "NC", "N_id", "N_p1", "N_p2", "E", "U", "R", "Q"]
        if (
            conf is None
            or len(conf) != 1186
            or not set(need) <= set(conf.columns)
            or conf[need].isna().any().any()
        ):
            raise ValueError(f"Incomplete annual confirmation: {name}")
        summary[name + " confirmation"] = model_block(conf, name, "Llama 3.1" in name)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print("Annual analyses complete.")


if __name__ == "__main__":
    main()
