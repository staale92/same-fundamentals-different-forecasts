from __future__ import annotations
import numpy as np
from scipy.stats import t


def losses(probability, label, metric="brier"):
    p, y = np.asarray(probability, float), np.asarray(label, float)
    if (
        p.shape != y.shape
        or not np.isfinite(p).all()
        or not np.isin(y, [0, 1]).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Invalid probabilities or binary labels")
    if metric == "brier":
        return (p - y) ** 2
    if metric == "log":
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    raise ValueError("Unknown proper score")


def cluster_ols(y, X, firms, periods, coefficient=0):
    """OLS with CR1 firm, period, and two-way inclusion-exclusion variances.

    This is a sensitivity analysis, not a few-cluster remedy. A negative
    two-way variance is reported as undefined; it is never clipped to zero.
    Intersections include all observations within a firm-period cell.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if (
        X.ndim != 2
        or len(y) != len(X)
        or len(y) <= X.shape[1]
        or np.linalg.matrix_rank(X) < X.shape[1]
    ):
        return {"status": "insufficient_rank_or_observations"}
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    influence = (X * residual[:, None]) @ bread
    f = np.asarray(firms).astype(str)
    p = np.asarray(periods).astype(str)

    def variance(keys):
        _, inverse = np.unique(keys, return_inverse=True, axis=0)
        G = int(inverse.max() + 1)
        if G < 2:
            return np.nan, G
        sums = np.zeros((G, k))
        np.add.at(sums, inverse, influence)
        v = (sums.T @ sums) * (G / (G - 1)) * ((n - 1) / (n - k))
        return float(v[coefficient, coefficient]), G

    vf, nf = variance(f)
    vp, np_ = variance(p)
    vi, ni = variance(np.column_stack([f, p]))
    out = {
        "status": "ok",
        "estimate": float(beta[coefficient]),
        "n": n,
        "firms": nf,
        "periods": np_,
        "variance_firm": vf,
        "variance_period": vp,
        "variance_intersection": vi,
    }
    for name, v, df in [
        ("firm", vf, nf - 1),
        ("period", vp, np_ - 1),
        ("two_way", vf + vp - vi, min(nf, np_) - 1),
    ]:
        valid = np.isfinite(v) and v >= 0 and df >= 1
        se = float(np.sqrt(v)) if valid else None
        radius = float(t.ppf(0.975, df) * se) if valid else None
        out.update(
            {
                name + "_se": se,
                name + "_df": df,
                name + "_low": float(beta[coefficient]) - radius if valid else None,
                name + "_high": float(beta[coefficient]) + radius if valid else None,
                name
                + "_status": "ok" if valid else "undefined_negative_variance_or_too_few_clusters",
            }
        )
    return out


def recall_bounds(theta, ci_low, ci_high, M, placebo_nonnegative=False):
    """Identification set and conservative sampling envelope, in score units.

    With nonnegative U recall removal, theta=L_C-L_U+drift only supplies
    a lower bound. Negative lower bounds are retained: no sign restriction
    on C's contribution is silently added.
    """
    if M < 0 or not np.isfinite([theta, ci_low, ci_high, M]).all() or ci_low > ci_high:
        raise ValueError("Invalid bounded-drift inputs")
    return {
        "M": float(M),
        "lower": float(theta - M),
        "upper": None if placebo_nonnegative else float(theta + M),
        "confidence_lower": float(ci_low - M),
        "confidence_upper": None if placebo_nonnegative else float(ci_high + M),
        "upper_identified": not placebo_nonnegative,
        "assumption": "L_U >= 0" if placebo_nonnegative else "L_U = 0",
    }
