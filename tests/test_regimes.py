"""Checks for src/regimes.py: causality (no look-ahead) and correctness.

Run: python -m pytest tests/test_regimes.py -q
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from src import regimes as rg

SPLIT = pd.Timestamp("2018-12-31")


@pytest.fixture(scope="module")
def fitted():
    panel = rg.load_panel("2010-07-19")
    params = rg.calibrate(panel[panel.index <= SPLIT])
    return panel, params, rg.label(panel, params)


# --- 1. No look-ahead ----------------------------------------------------------


def test_labels_are_invariant_to_truncation(fitted):
    """The label at t computed from data up to t equals the full-sample label at t."""
    panel, params, full = fitted
    rng = np.random.default_rng(0)
    dates = list(rng.choice(panel.index[30:], size=12, replace=False)) + [SPLIT, panel.index[-1]]
    for t in dates:
        t = pd.Timestamp(t)
        if t not in panel.index:
            continue
        cut = rg.label(panel[panel.index <= t], params)
        for col in (*rg.METHODS, "ms_prob_stress", "rv21"):
            a, b = cut.loc[t, col], full.loc[t, col]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, (t, col, a, b)


def test_future_data_cannot_change_past_labels(fitted):
    """Scramble everything after a date: labels up to that date are unchanged."""
    panel, params, full = fitted
    t = pd.Timestamp("2021-06-30")
    shocked = panel.copy()
    after = shocked.index > t
    rng = np.random.default_rng(1)
    shocked.loc[after, "logret"] = rng.normal(0, 0.05, after.sum())
    shocked.loc[after, "vix"] = 80.0
    lab = rg.label(shocked, params)
    before = full.index <= t
    for col in (*rg.METHODS, "ms_prob_stress"):
        pd.testing.assert_series_equal(lab.loc[before, col], full.loc[before, col])


def test_calibration_ignores_test_period(fitted):
    """Parameters depend only on train rows: altering test rows leaves them unchanged."""
    panel, params, _ = fitted
    shocked = panel.copy()
    shocked.loc[shocked.index > SPLIT, "vix"] *= 3
    again = rg.calibrate(shocked[shocked.index <= SPLIT])
    assert again["vix_q75"] == params["vix_q75"] and again["rv21_q75"] == params["rv21_q75"]
    assert again["markov_switching"]["mu_pct"] == params["markov_switching"]["mu_pct"]


def test_trailing_and_forward_windows():
    r = pd.Series(np.arange(1.0, 41.0) / 1000)
    rv = rg.trailing_rv(r, 5)
    assert math.isclose(rv.iloc[10], math.sqrt(252 * np.mean(r.iloc[6:11] ** 2)))  # up to t
    assert rv.iloc[:4].isna().all()
    fwd = rg.forward_rv(r, 5)
    assert math.isclose(fwd.iloc[10], math.sqrt(252 * np.mean(r.iloc[11:16] ** 2)))  # after t
    assert fwd.iloc[-5:].isna().all()


# --- 2. Correctness ------------------------------------------------------------


def test_hamilton_filter_matches_bruteforce():
    """Filtered probabilities and likelihood equal a direct sum over all 2^T paths."""
    rng = np.random.default_rng(2)
    r = rng.normal(0, 1.2, 8)
    mu, sigma = np.array([0.1, -0.2]), np.array([0.7, 1.6])
    P = np.array([[0.95, 0.05], [0.1, 0.9]])
    filtered, loglik = rg.hamilton_filter(r, mu, sigma, P)
    pi1 = P[0, 1] / (P[0, 1] + P[1, 0])
    pi = np.array([1 - pi1, pi1])
    dens = lambda x, s: math.exp(-0.5 * ((x - mu[s]) / sigma[s]) ** 2) / (sigma[s] * math.sqrt(2 * math.pi))
    for t in range(len(r)):
        weight = np.zeros(2)
        for path in itertools.product((0, 1), repeat=t + 1):
            # The filter starts from the stationary law before the first observation.
            w = sum(pi[s0] * P[s0, path[0]] for s0 in (0, 1))
            for k in range(1, t + 1):
                w *= P[path[k - 1], path[k]]
            for k in range(t + 1):
                w *= dens(r[k], path[k])
            weight[path[-1]] += w
        assert np.allclose(filtered[t], weight / weight.sum(), atol=1e-12)
        if t == len(r) - 1:
            assert math.isclose(loglik, math.log(weight.sum()), rel_tol=1e-10)


def test_markov_switching_fit_recovers_known_parameters():
    """Simulate a known 2-state chain; the MLE should recover it approximately."""
    rng = np.random.default_rng(3)
    n, P = 3000, np.array([[0.98, 0.02], [0.04, 0.96]])
    mu, sigma = np.array([0.05, -0.1]), np.array([0.7, 1.6])
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = rng.random() < P[s[t - 1], 1]
    r = mu[s] + sigma[s] * rng.standard_normal(n)
    fit = rg.fit_markov_switching(r, starts=2)
    assert np.allclose(fit["sigma_pct"], sigma, rtol=0.1), fit["sigma_pct"]
    assert abs(fit["P"][0][0] - 0.98) < 0.015 and abs(fit["P"][1][1] - 0.96) < 0.03, fit["P"]


def test_hysteresis_rule():
    x = pd.Series([10, 21, 19, 18, 17, 25, np.nan, 16, 22])
    lab = rg.hysteresis_label(x, enter=20, leave=17.5)
    expected = [0, 1, 1, 1, 0, 1, np.nan, 0, 1]
    assert np.allclose(lab.to_numpy(), expected, equal_nan=True)


def test_spells():
    lab = pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
                    index=pd.date_range("2020-01-01", periods=10))
    sp = rg.spells(lab)
    assert sp["days"].tolist() == [2, 1, 3]
