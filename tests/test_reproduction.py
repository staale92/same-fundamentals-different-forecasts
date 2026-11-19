from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from numeraire.schema import Statement, read_jsonl
from numeraire.transforms import render, _percentile
from numeraire.token_inference import conditional_probability
from numeraire.robustness import (
    WEBB,
    wild_cluster_restricted,
    wcr_interval,
    _cr1_slope,
    sign_flip,
    within_firm_differences,
)

ROOT = Path(__file__).resolve().parents[1]


def reference_wcr(y, x, clusters, b0=0, draws=199, seed=2026):
    _, groups = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    beta, se = _cr1_slope(y, x, groups)
    restricted = y - b0 * x
    intercept = restricted.mean()
    residual = restricted - intercept
    rng = np.random.default_rng(seed)
    weights = WEBB[rng.integers(0, 6, size=(draws, groups.max() + 1))]
    ys = intercept + b0 * x[None, :] + residual[None, :] * weights[:, groups]
    bs, ses = _cr1_slope(ys, x, groups)
    return float(np.mean(abs((bs - b0) / ses) >= abs((beta[0] - b0) / se[0])))


@pytest.mark.parametrize("firms", [6, 11, 107])
def test_cluster_aggregation_reproduces_observation_bootstrap(firms):
    rng = np.random.default_rng(52)
    g = np.repeat(np.arange(firms), 5)
    x = rng.integers(0, 2, len(g)).astype(float)
    y = rng.normal(0, 0.1, len(g)) + 0.02 * x
    for null in (-0.04, 0, 0.04):
        assert wild_cluster_restricted(y, x, g, null, draws=199) == reference_wcr(y, x, g, null)
    b, se = _cr1_slope(y, x, g)
    grid = np.linspace(b[0] - 6 * se[0], b[0] + 6 * se[0], 21)
    accepted = [v for v in grid if reference_wcr(y, x, g, v) > 0.05]
    expected = (min(accepted), max(accepted)) if accepted else (None, None)
    assert wcr_interval(y, x, g, b[0], se[0], draws=199, grid=21) == expected


@pytest.mark.parametrize("arm", ["N", "NP", "NC", "N_id", "N_p1", "N_p2", "E", "U"])
def test_exact_arms_recover_the_original_economic_values(arm):
    s = Statement.from_dict(read_jsonl(ROOT / "data/processed/sec_pilot.jsonl")[0])
    r = render(s, arm)
    assert r.recover_native() == s.values


def test_zero_rank_is_a_valid_observation():
    assert _percentile([Decimal(1), Decimal(2)], Decimal(1)) == 0
    assert _percentile([Decimal(1), Decimal(2)], Decimal(2)) == 100


def test_exact_cent_observations_are_retained():
    reference_ids = {
        r["id"] for r in read_jsonl(ROOT / "data/processed/planted_rank_reference.jsonl")
    }
    primary = read_jsonl(ROOT / "data/processed/sec_quarterly_confirmation.jsonl")
    boundary_cases = [r for r in primary if r["id"] not in reference_ids]
    assert len(boundary_cases) == 5
    assert all(
        abs(Decimal(r["provenance"]["eps_change"])) == Decimal(".01") for r in boundary_cases
    )


def test_sign_flip_estimate_is_equal_weighted_firm_mean():
    frame = pd.DataFrame(
        {
            "firm_id": ["a", "a", "a", "b", "b"],
            "exposure": ["exposed", "exposed", "unexposed", "exposed", "unexposed"],
            "gap": [0.1, 0.3, 0.1, 0.5, 0.2],
        }
    )
    result = sign_flip(within_firm_differences(frame).to_numpy())
    assert result["estimate"] == pytest.approx(0.2)


def test_label_swap_preserves_direction():
    assert conditional_probability(-0.1, -2, False) == pytest.approx(
        conditional_probability(-2, -0.1, True)
    )


def test_executed_control_assignment_and_training_texts(tmp_path):
    from prepare_control_data import prepare

    output = prepare(tmp_path / "control")
    actual = ROOT / "results/planted_control"
    assert read_jsonl(output / "assignments.jsonl") == read_jsonl(
        actual / "assignment/assignments.jsonl"
    )
    for name in ("planted_train", "flipped_train", "planted_chat", "flipped_chat"):
        assert read_jsonl(output / f"{name}.jsonl") == read_jsonl(actual / f"{name}.jsonl")


def test_context_disagreement_uses_the_reported_arm():
    from analyze_forecasts import summarize

    ctx = {
        ("one", "N"): {False: (0.0, 0.0), True: (0.0, 0.0)},
        ("one", "NC"): {False: (4.0, 4.0), True: (-4.0, -4.0)},
    }
    predictions, disagreement = summarize(ctx, None, "ab_raw", disagreement_arm="N")
    pooled_predictions, pooled_disagreement = summarize(ctx, None, "ab_raw")
    assert predictions == pooled_predictions
    assert disagreement == 0.0
    assert pooled_disagreement > 0.4


def test_directional_flip_ties_match_up_at_half_convention():
    from analyze_presentation import ARMS, flips

    w = pd.DataFrame({"N": [0.5, 0.4, 0.6], **{arm: [0.6, 0.5, 0.4] for arm in ARMS}})
    for value in flips(w).values():
        assert value["flips"] == 2
        assert value["reference_ties"] == 1
        assert value["arm_ties"] == 1
    w = pd.DataFrame({"N": [0.5], **{arm: [0.6] for arm in ARMS}})
    for value in flips(w).values():
        assert value["flip_rate"] == 0.0
        assert value["flip_rate_ties_down"] == 1.0
