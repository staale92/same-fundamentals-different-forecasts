from __future__ import annotations
import numpy as np


def brier_decomposition(probabilities, labels, weights=None):
    """Exact finite-support identity; each row is one observation, columns arms."""
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    if p.ndim != 2 or y.shape != (len(p),) or p.shape[1] == 0:
        raise ValueError("Expected probabilities (n, arms) and labels (n,)")
    w = np.ones(p.shape[1]) / p.shape[1] if weights is None else np.asarray(weights, float)
    if w.shape != (p.shape[1],) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("Weights must be nonnegative and sum to one")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Brier decomposition requires finite complete support")
    mean = p @ w
    dispersion = ((p - mean[:, None]) ** 2) @ w
    mean_loss = ((p - y[:, None]) ** 2) @ w
    ensemble_loss = (mean - y) ** 2
    return {
        "probability": mean,
        "dispersion": dispersion,
        "mean_arm_brier": mean_loss,
        "ensemble_brier": ensemble_loss,
        "residual": mean_loss - ensemble_loss - dispersion,
    }


def _draw_weights(n_firms, bootstrap, seed):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_firms, np.ones(n_firms) / n_firms, size=bootstrap).astype(float)


def _interval(draws):
    valid = np.asarray(draws)[np.isfinite(draws)]
    if len(valid) < 20:
        return {"ci_low": None, "ci_high": None, "bootstrap_valid": int(len(valid))}
    lo, hi = np.quantile(valid, [0.025, 0.975])
    return {"ci_low": float(lo), "ci_high": float(hi), "bootstrap_valid": int(len(valid))}
