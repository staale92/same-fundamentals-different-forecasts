#!/usr/bin/env python3
"""Exploratory temporal and firm-identification sensitivity analyses.

Compute drift with at least three observed quarters on each side of a split,
exclude 2021Q1 or 2021Q1--Q3, and stratify forecast contrasts by firm identification.
Recalibrate drift after excluding 2021Q1--Q3, with minimum segment lengths
of one and three quarters.
Compare identification rates across firm sizes measured by 2018 total assets.
All calculations use saved outputs; these choices are post hoc.
"""

import json

import numpy as np
import pandas as pd

from numeraire.schema import exposure_cell, read_jsonl
from numeraire.scoring import cluster_ols
from analyze_primary import MODELS, PANEL, ROOT, frame, predictions, theta
from analyze_audit_checks import LL, RHO, diff, gaps

OUT = ROOT / "results/sensitivity"
PANDEMIC_COMPARISONS = ("2021Q1", "2021Q2", "2021Q3")


def restricted_splits(f, minimum_quarters=3):
    """Split within each exposure cell, retaining enough periods on both sides."""
    estimates = []
    for cell in ("exposed", "unexposed"):
        subset = f[f.cell_llama == cell]
        quarters = sorted(subset.period.unique())
        for boundary in range(minimum_quarters, len(quarters) - minimum_quarters + 1):
            estimate = diff(subset, subset.period.isin(quarters[:boundary]))
            estimate.update({"cell": cell, "split_after": quarters[boundary - 1]})
            estimates.append(estimate)
    return estimates


def choice_frame(model):
    choices = pd.DataFrame(
        read_jsonl(ROOT / f"results/identification/{model}/choice_scores.jsonl")
    )
    records = {row["id"]: row for row in read_jsonl(PANEL)}
    choices["pred"] = choices.logp.map(lambda scores: max(scores, key=scores.get))
    choices["ok"] = choices.pred.eq(choices.true_letter).astype(float)
    choices["p_true"] = [
        np.exp(scores[letter]) / sum(np.exp(value) for value in scores.values())
        for scores, letter in zip(choices.logp, choices.true_letter)
    ]
    choices["cell"] = choices.id.map(
        lambda observation: exposure_cell(records[observation], "2023-12-31")
    )
    return choices


def mean_interval(data):
    result = cluster_ols(
        data.ok.to_numpy(float), np.ones((len(data), 1)), data.firm_id, data.id, 0
    )
    return {
        "rate": result["estimate"],
        "low": result["firm_low"],
        "high": result["firm_high"],
        "n": int(len(data)),
    }


def identification_strata(f, identified):
    f = f.assign(
        id70_ok=f.index.map(identified.ok),
        id70_p=f.index.map(identified.p_true),
    )
    if f[["id70_ok", "id70_p"]].isna().any().any():
        raise ValueError("Missing identification scores for forecast observations")
    median = f.id70_p.median()
    strata = {}
    for name, mask in (
        ("identified", f.id70_ok.eq(1)),
        ("not_identified", f.id70_ok.eq(0)),
        ("p_true_above_median", f.id70_p > median),
        ("p_true_below_median", f.id70_p <= median),
    ):
        subset = f[mask]
        strata[name] = diff(subset, subset.cell_llama.eq("exposed")) | {
            "n": int(len(subset)),
            "firms": int(subset.firm_id.nunique()),
        }
    exposed = f.cell_llama.eq("exposed").to_numpy(float)
    correct = f.id70_ok.to_numpy(float)
    result = cluster_ols(
        f.g.to_numpy(),
        np.column_stack([np.ones(len(f)), exposed, correct, exposed * correct]),
        f.firm_id,
        f.period,
        3,
    )
    strata["interaction_identified"] = {
        key: result[key] for key in ("estimate", "firm_low", "firm_high")
    }
    return strata


def identification_comparisons(choices):
    result = {}
    for arm, subset in choices.groupby("arm"):
        result[arm] = {
            cell: mean_interval(subset[subset.cell == cell])
            for cell in ("exposed", "unexposed")
        } | {"all": mean_interval(subset)}
    for cell in ("exposed", "unexposed"):
        subset = choices[choices.arm.isin(["N", "NC"]) & (choices.cell == cell)]
        estimate = cluster_ols(
            subset.ok.to_numpy(),
            np.column_stack([np.ones(len(subset)), subset.arm.eq("N").to_numpy(float)]),
            subset.firm_id,
            subset.id,
            1,
        )
        result[f"N_minus_NC_{cell}"] = {
            "estimate": estimate["estimate"],
            "low": estimate["firm_low"],
            "high": estimate["firm_high"],
        }
    subset = choices[
        choices.arm.isin(["N", "NC"]) & choices.cell.isin(["exposed", "unexposed"])
    ]
    native = subset.arm.eq("N").to_numpy(float)
    exposed = subset.cell.eq("exposed").to_numpy(float)
    estimate = cluster_ols(
        subset.ok.to_numpy(),
        np.column_stack([np.ones(len(subset)), native, exposed, native * exposed]),
        subset.firm_id,
        subset.id,
        3,
    )
    result["drop_difference_exposed_minus_unexposed"] = {
        "estimate": estimate["estimate"],
        "low": estimate["firm_low"],
        "high": estimate["firm_high"],
    }
    result["accuracy_by_true_letter"] = (
        choices.groupby("true_letter").ok.mean().round(3).to_dict()
    )
    result["true_letter_shares"] = (
        choices.true_letter.value_counts(normalize=True).round(3).to_dict()
    )
    return result


def identification_by_size(choices):
    """Compare as-filed identification in the top asset tercile with other firms."""
    selection = json.loads((ROOT / "data/raw/sec_confirmation/selection.json").read_text())
    assets = pd.Series(
        {
            f'{int(firm["cik"]):010d}': firm["assets_CY2018Q4I"]
            for firm in selection["candidates"]
        }
    )
    firm_assets = assets.reindex(sorted(choices.firm_id.unique())).dropna()
    top = set(firm_assets[firm_assets >= firm_assets.quantile(2 / 3)].index)
    native = choices[choices.arm == "N"]
    return {
        "top_tercile": mean_interval(native[native.firm_id.isin(top)]),
        "bottom_two_terciles": mean_interval(native[~native.firm_id.isin(top)]),
    }


def exclusion_result(forecasts, model, periods):
    estimate = theta(forecasts, model, mask=~forecasts.period.isin(periods))
    return {
        key: estimate[key]
        for key in (
            "estimate", "wcr_low", "wcr_high", "firm_low", "firm_high",
            "two_way_low", "two_way_high",
        )
    } | {"signflip_p": estimate["firm_level"]["signflip_p"]}


def main():
    report = {
        "llama_70b_identification_strata": {},
        "identification": {},
        "drift_min3": {},
        "drift_restricted": {},
        "theta_without_2021Q1": {},
        "theta_without_2020_comparisons": {},
    }
    primary = json.loads((ROOT / "results/primary/report.json").read_text())["primary"]
    choices = {model: choice_frame(model) for model in ("llama31-8b", "llama31-70b")}
    identified = choices["llama31-70b"].query("arm == 'N'").set_index("id")
    for model in LL:
        forecasts = frame(predictions(MODELS[model]))
        f = gaps(forecasts)
        splits = restricted_splits(f)
        drift = max(abs(estimate["estimate"]) for estimate in splits)
        upper = primary[model]["primary"]["wcr_high"]
        report["drift_min3"][model] = {
            "splits": splits,
            "M": drift,
            "bound_rho_0": upper + drift,
            "bound_rho_hat": (upper + drift) / (1 - RHO),
        }
        restricted = f[~f.period.isin(PANDEMIC_COMPARISONS)]
        splits_min3 = restricted_splits(restricted, minimum_quarters=3)
        splits_any = restricted_splits(restricted, minimum_quarters=1)
        report["drift_restricted"][model] = {
            "M_min3": max(abs(split["estimate"]) for split in splits_min3),
            "M_any": max(abs(split["estimate"]) for split in splits_any),
            "splits_min3": splits_min3,
            "splits_any": splits_any,
        }
        report["theta_without_2021Q1"][model] = exclusion_result(
            forecasts, model, ["2021Q1"]
        )
        report["theta_without_2020_comparisons"][model] = exclusion_result(
            forecasts, model, PANDEMIC_COMPARISONS
        )
        report["llama_70b_identification_strata"][model] = identification_strata(
            f, identified
        )
    for model, data in choices.items():
        report["identification"][model] = identification_comparisons(data)
        if model == "llama31-70b":
            report["identification"][model]["N_by_size"] = identification_by_size(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"Saved temporal and identification sensitivities to {OUT.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
