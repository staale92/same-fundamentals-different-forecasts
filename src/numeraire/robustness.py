from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def murphy(probability, label, bins=10):
    """Binned Murphy decomposition: Brier ~ reliability - resolution + uncertainty.

    Equal-width bins on [0, 1]; ``within_bin`` is the exact remainder so the
    four terms sum to the Brier score.
    """
    p = np.asarray(probability, float)
    y = np.asarray(label, float)
    index = np.minimum((p * bins).astype(int), bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in np.unique(index):
        m = index == b
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    brier = float(np.mean((p - y) ** 2))
    unc = float(base * (1 - base))
    return {
        "brier": brier,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": unc,
        "within_bin": brier - (rel - res + unc),
    }


def auc(probability, label):
    y = np.asarray(label)
    return float(roc_auc_score(y, probability)) if 0 < y.sum() < len(y) else None


def shift_decomposition(p_arm, p_ref, label):
    """Brier_k - Brier_ref = E[(f_k-f_r)(f_k+f_r-2Y)], split into the part
    implied by a uniform mean shift s and the remainder."""
    pk, pr, y = (np.asarray(v, float) for v in (p_arm, p_ref, label))
    total = float(np.mean((pk - y) ** 2 - (pr - y) ** 2))
    s = float(np.mean(pk - pr))
    shift = s * (2 * pr.mean() + s - 2 * y.mean())
    return {
        "delta_brier": total,
        "mean_shift": s,
        "shift_component": float(shift),
        "remainder": total - float(shift),
    }


def _cr1_slope(y, x, groups):
    """Slope and CR1 standard error of y on [1, x], vectorized over rows of y."""
    y = np.atleast_2d(y)
    xc = x - x.mean()
    sxx = xc @ xc
    b = (y - y.mean(axis=1, keepdims=True)) @ xc / sxx
    a = y.mean(axis=1) - b * x.mean()
    e = y - a[:, None] - b[:, None] * x[None, :]
    G = groups.max() + 1
    n = len(x)
    scores = np.zeros((y.shape[0], G))
    np.add.at(scores.T, groups, (xc[None, :] * e).T)
    var = (scores**2).sum(axis=1) / sxx**2 * (G / (G - 1)) * ((n - 1) / (n - 2))
    return b, np.sqrt(var)


def _restricted_statistics(y, x, clusters, draws, seed):
    """Sufficient statistics for the same Webb WCR draws at every null value.

    Restricted residuals are (y - mean(y)) - b0 * (x - mean(x)).
    Their cluster scores and bootstrap variance are linear and quadratic in b0.
    Aggregating before resampling avoids a draws-by-observations array for each
    point of the inversion grid, without changing the bootstrap experiment.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    _, groups = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    b, se = _cr1_slope(y, x, groups)
    xc, yc = x - x.mean(), y - y.mean()
    n, G = len(x), int(groups.max() + 1)
    sxx = xc @ xc
    sums_y = np.bincount(groups, weights=yc, minlength=G)
    sums_x = np.bincount(groups, weights=xc, minlength=G)
    sums_xy = np.bincount(groups, weights=xc * yc, minlength=G)
    sums_xx = np.bincount(groups, weights=xc * xc, minlength=G)
    rng = np.random.default_rng(seed)
    weights = WEBB[rng.integers(0, 6, size=(draws, G))]
    a_y, a_x = weights @ sums_y / n, weights @ sums_x / n
    b_y, b_x = weights @ sums_xy / sxx, weights @ sums_xx / sxx
    score_y = weights * sums_xy - a_y[:, None] * sums_x - b_y[:, None] * sums_xx
    score_x = weights * sums_xx - a_x[:, None] * sums_x - b_x[:, None] * sums_xx
    factor = G / (G - 1) * (n - 1) / (n - 2) / sxx**2
    quadratic = (
        np.sum(score_y**2, axis=1) * factor,
        np.sum(score_y * score_x, axis=1) * factor,
        np.sum(score_x**2, axis=1) * factor,
    )
    return b[0], se[0], b_y, b_x, quadratic


def _restricted_pvalue(statistics, b0):
    b, se, b_y, b_x, (v_y, v_xy, v_x) = statistics
    variance = v_y - 2 * b0 * v_xy + b0**2 * v_x
    # A negative value of roundoff size can arise from polynomial cancellation.
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = (b_y - b0 * b_x) / standard_error
    return float(np.mean(np.abs(t_star) >= abs((b - b0) / se)))


def wild_cluster_restricted(y, x, clusters, b0=0.0, draws=9999, seed=2026):
    """Restricted bootstrap-t p-value using Webb's six-point weights."""
    return _restricted_pvalue(_restricted_statistics(y, x, clusters, draws, seed), b0)


def wcr_interval(y, x, clusters, estimate, se, draws=9999, seed=2026, level=0.95, grid=241):
    """Invert the declared +/-6-SE, 241-point grid with common seeded draws."""
    statistics = _restricted_statistics(y, x, clusters, draws, seed)
    values = np.linspace(estimate - 6 * se, estimate + 6 * se, grid)
    accepted = [v for v in values if _restricted_pvalue(statistics, v) > 1 - level]
    return (float(min(accepted)), float(max(accepted))) if accepted else (None, None)


def sign_flip(contributions, shares=None, level=0.95):
    """Exact firm sign-flip randomization for theta = sum_f c_f.

    Each c_f must be mean-zero under H0 (e.g. a within-firm difference or a
    rectangle double difference). Under H0: theta = theta0 the centered terms
    c_f - theta0 * share_f are symmetric; ``shares`` are each firm's weight in
    theta (summing to one; equal by default). Enumerates all 2^G patterns for
    G <= 16; beyond that, the same randomization distribution is approximated by
    19,999 seeded random patterns plus the identity (a Monte Carlo randomization test).
    """
    c = np.asarray(contributions, float)
    G = len(c)
    a = np.full(G, 1 / G) if shares is None else np.asarray(shares, float)
    if G <= 16:
        signs = np.array(list(product([-1, 1], repeat=G)), float)
    else:
        signs = np.vstack(
            [np.ones(G), np.random.default_rng(2026).choice([-1.0, 1.0], size=(19999, G))]
        )
    theta = c.sum()

    def p_value(theta0):
        centered = c - theta0 * a
        return float(np.mean(np.abs(signs @ centered) >= abs(centered.sum()) - 1e-12))

    span = 8 * np.sqrt((c**2).sum()) + 1e-12
    grid = np.linspace(theta - span, theta + span, 641)
    accepted = [g for g in grid if p_value(g) > 1 - level]
    return {
        "estimate": float(theta),
        "p_value": p_value(0.0),
        "low": float(min(accepted)) if accepted else None,
        "high": float(max(accepted)) if accepted else None,
        "firms": G,
        "method": "exact" if G <= 16 else "monte carlo (19,999 draws)",
    }


def within_firm_differences(frame, value="gap", exposed="exposed"):
    """Per-firm exposed-minus-unexposed mean differences, for firms in both cells.

    A common presentation mean cancels within firm, so each difference is
    mean-zero under no exposure effect; their average is the firm-level estimand.
    """
    m = frame.groupby(["firm_id", frame.exposure.eq(exposed)])[value].mean().unstack()
    m = m.dropna(subset=[c for c in (True, False) if c in m])
    if True not in m or False not in m:
        return pd.Series(dtype=float)
    return (m[True] - m[False]) / len(m)


def mde(se, df, power=0.8, level=0.95):
    """Minimum detectable effect for a two-sided t test with df degrees of freedom."""
    return float((student_t.ppf(1 - (1 - level) / 2, df) + student_t.ppf(power, df)) * se)


def breakdown(estimate, ci_low):
    """Smallest M at which the lower-only bound (point or interval) is nonpositive."""
    return {
        "point_breakdown_M": max(0.0, float(estimate)),
        "interval_breakdown_M": max(0.0, float(ci_low)),
    }
