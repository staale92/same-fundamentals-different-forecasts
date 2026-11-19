from __future__ import annotations
import numpy as np
import pandas as pd
from .analysis import _draw_weights, _interval


def planted_validation(
    predictions,
    assignments,
    bootstrap=2000,
    seed=2026,
    metric="accuracy",
    control="shuffled",
    trained="E",
    placebo="U",
    return_draws=False,
):
    """Randomized pre/post x planted/control x treatment-control accuracy effects.

    Required prediction fields: id, arm, stage (pre/post), variant
    (planted or the named control branch), probability, label. The control is
    'shuffled' (permuted labels) or 'flipped' (every treated label inverted).
    Supply one key type/model/regime per call. Both branches must be evaluated
    before and after fine-tuning.
    """
    p = pd.DataFrame(predictions).copy()
    assignments = pd.DataFrame(assignments)
    if control not in ("shuffled", "flipped"):
        raise ValueError("Unknown control branch")
    if control != "shuffled":
        if p.variant.eq("shuffled").any():
            raise ValueError("Mixed control branches")
        p["variant"] = p.variant.replace({control: "shuffled"})
    keys = ["id", "arm", "stage", "variant"]
    if p.duplicated(keys).any():
        raise ValueError("Multiple predictions per planted validation cell")
    if assignments.id.duplicated().any():
        raise ValueError("Duplicate assignments")
    if p.groupby("id").label.nunique().gt(1).any():
        raise ValueError("Labels differ across validation branches")
    if "label" in assignments:
        matched = (
            p[["id", "label"]]
            .drop_duplicates()
            .merge(assignments[["id", "label"]], on="id", suffixes=("_prediction", "_assignment"))
        )
        if (matched.label_prediction != matched.label_assignment).any():
            raise ValueError("Prediction and assignment labels disagree")
    if set(p.id) - set(assignments.id):
        raise ValueError("Predictions contain unassigned observations")
    if not p.stage.isin(["pre", "post"]).all() or not p.variant.isin(["planted", "shuffled"]).all():
        raise ValueError("Expected pre/post stages and planted/shuffled branches")
    if not p.probability.between(0, 1).all() or not p.label.isin([0, 1]).all():
        raise ValueError("Invalid predictions")
    if metric == "accuracy":
        p["correct"] = (p.probability.ge(0.5) == p.label).astype(float)
    elif metric == "brier":
        p["correct"] = -((p.probability - p.label) ** 2)
    elif metric == "log":
        clipped = np.clip(p.probability, 1e-12, 1 - 1e-12)
        p["correct"] = p.label * np.log(clipped) + (1 - p.label) * np.log1p(-clipped)
    else:
        raise ValueError("Unknown validation score")
    wide = p.pivot(index=["id", "arm"], columns=["variant", "stage"], values="correct")
    required = [("planted", "post"), ("planted", "pre"), ("shuffled", "post"), ("shuffled", "pre")]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(required)).dropna()
    if wide.empty:
        raise ValueError("No complete pre/post planted/shuffled support")
    delta = wide.to_numpy() @ np.array([1.0, -1.0, -1.0, 1.0])
    result = wide.index.to_frame(index=False)
    result["delta"] = delta
    result = result.merge(
        assignments[["id", "firm_id", "treated"]], on="id", validate="many_to_one"
    )
    if not result.treated.isin([True, False, 0, 1]).all():
        raise ValueError("Treatment assignment must be binary")
    support = result.groupby("id").arm.nunique()
    all_arms = sorted(result.arm.unique())
    complete_ids = support[support == len(all_arms)].index
    result = result[result.id.isin(complete_ids)].copy()
    if result.empty or set(result.treated.unique()) != {False, True}:
        raise ValueError("Both assignment groups need common arm support")
    firms = pd.Index(sorted(result.firm_id.astype(str).unique()))
    result["firm_id"] = result.firm_id.astype(str)
    columns = ["arm", "treated"]
    sums = (
        result.pivot_table(index="firm_id", columns=columns, values="delta", aggfunc="sum")
        .reindex(firms)
        .fillna(0)
    )
    counts = (
        result.pivot_table(index="firm_id", columns=columns, values="delta", aggfunc="count")
        .reindex(index=firms, columns=sums.columns)
        .fillna(0)
    )
    weights = _draw_weights(len(firms), bootstrap, seed)
    numer = weights @ sums.to_numpy()
    denom = weights @ counts.to_numpy()
    draws = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    means = sums.sum() / counts.sum()
    lookup = {tuple(c): j for j, c in enumerate(sums.columns)}
    effects = {}
    effect_draws = {}
    for arm in all_arms:
        if (arm, True) not in lookup or (arm, False) not in lookup:
            raise ValueError("Missing randomized treatment cell")
        effects[arm] = float(means[arm, True] - means[arm, False])
        effect_draws[arm] = draws[:, lookup[arm, True]] - draws[:, lookup[arm, False]]
    if trained not in effects or placebo not in effects:
        raise ValueError(f"{trained} and {placebo} are needed for recovery")
    rows = []
    for arm in all_arms:
        # Compare each arm with the specified trained and placebo arms.
        removed = effects[placebo] - effects[arm]
        removed_draws = effect_draws[placebo] - effect_draws[arm]
        error = removed - effects[trained]
        reference_removed = effects[trained] - effects[arm]
        reference_draws = effect_draws[trained] - effect_draws[arm]
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "randomized_effect": effects[arm],
                "randomized_effect_interval": _interval(effect_draws[arm]),
                "total_planted_effect": effects[trained],
                "trained_minus_arm_effect": reference_removed,
                "placebo_minus_arm_effect": removed,
                "recovery_error_vs_total": error,
                "recovery_error_vs_removed_effect": removed - reference_removed,
                **_interval(removed_draws),
                "recovery_error_interval": _interval(removed_draws - effect_draws[trained]),
                "removed_effect_recovery_interval": _interval(removed_draws - reference_draws),
                **(
                    {
                        "trained_arm": trained,
                        "placebo_arm": placebo,
                        "removal_interval": _interval(reference_draws),
                    }
                    if (trained, placebo) != ("E", "U")
                    else {}
                ),
            }
        )
    out = {
        "n_observations": int(result.id.nunique()),
        "n_firms": len(firms),
        "metric": metric,
        "effects": rows,
        "interpretation": "The trained-arm effect is the treated-minus-control pre/post performance gain, net of the specified label-control branch. Interpreting a placebo-minus-arm contrast as total recall requires placebo retention and complete arm removal.",
    }
    if return_draws:
        out["_draws"] = (
            effect_draws  # bootstrap draws of each arm's randomized effect (not serialized)
        )
    return out
