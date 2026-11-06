"""Summary statistics of hedging P&L with Monte Carlo standard errors.

All estimators are sample means over i.i.d. paths, so the standard error of a
mean is std/sqrt(n). For ratios such as the variance reduction 1 - A/B we use
the delta method on the paired pathwise quantities (the same paths enter A and
B, which is what makes the comparison precise).
"""

from __future__ import annotations

import math

import numpy as np


def mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def ratio_se(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """Estimate mean(a)/mean(b) and its delta-method SE for paired samples."""
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    mean_b = float(np.mean(b))
    r = float(np.mean(a)) / mean_b
    influence = (a - r * b) / mean_b
    return r, float(np.std(influence, ddof=1) / math.sqrt(len(a)))


def pnl_summary(
    payoff: np.ndarray,
    gain: np.ndarray,
    cost: np.ndarray,
    capital: float,
    baseline_centered_sq: np.ndarray,
) -> dict[str, float]:
    """Statistics of the short-call P&L e = capital + gain - cost - payoff.

    ``baseline_centered_sq`` is (payoff - mean payoff)^2, the no-hedge squared
    error, used for the variance reduction exactly as in the upstream script
    (``summarize_errors``): VR = 1 - Var(e) / Var(payoff), each variance being
    the centered second moment.
    """
    error = capital + gain - cost - payoff
    centered_sq = (error - np.mean(error)) ** 2
    mse_values = error**2
    mean_err, mean_err_se = mean_se(error)
    var_err, var_err_se = mean_se(centered_sq)
    mse, mse_se = mean_se(mse_values)
    ratio, ratio_err = ratio_se(centered_sq, baseline_centered_sq)
    cost_mean, cost_se = mean_se(cost)
    return {
        "mean_pnl": mean_err,
        "mean_pnl_se": mean_err_se,
        "std_pnl": math.sqrt(var_err),
        "std_pnl_se": var_err_se / (2.0 * math.sqrt(var_err)),
        "rmse": math.sqrt(mse),
        "rmse_se": mse_se / (2.0 * math.sqrt(mse)),
        "variance_reduction": 1.0 - ratio,
        "variance_reduction_se": ratio_err,
        "mean_cost": cost_mean,
        "mean_cost_se": cost_se,
    }


def block_bootstrap_se(values: np.ndarray, block: int, reps: int = 2000,
                       seed: int = 0) -> float:
    """Moving-block bootstrap SE of the mean of a serially dependent series.

    Used for the real-data backtest, where options written on consecutive days
    overlap and their hedging errors are strongly autocorrelated: the i.i.d.
    formula std/sqrt(n) would then understate the uncertainty. Blocks of
    ``block`` consecutive observations (at least the overlap length) are
    resampled with replacement (Kunsch 1989).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if block < 1 or block > n:
        raise ValueError("need 1 <= block <= n")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    # Block sums via cumulative sums: mean of a resample = sum of drawn blocks / n.
    csum = np.concatenate([[0.0], np.cumsum(x)])
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    sums = csum[starts + block] - csum[starts]
    # Trim the last block so each resample has exactly n observations.
    excess = n_blocks * block - n
    if excess:
        last = starts[:, -1]
        sums[:, -1] = csum[last + block - excess] - csum[last]
    means = sums.sum(axis=1) / n
    return float(np.std(means, ddof=1))
