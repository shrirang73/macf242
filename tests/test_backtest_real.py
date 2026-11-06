"""Checks for the real-data backtest (src/backtest_real.py) and block bootstrap.

Run: python -m pytest tests/test_backtest_real.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import backtest_real as bt
from src import metrics
from src import rebalancing as rb


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@pytest.fixture(scope="module")
def panel():
    return bt.load_panel()


# --- 1. Pricing and a hand-computed option -----------------------------------------


def test_bs_delta_is_derivative_of_price():
    s, k, tau, vol, h = 1.03, 1.0, 15 / 252, 0.18, 1e-5
    fd = (bt.bs_call(s + h, k, tau, vol) - bt.bs_call(s - h, k, tau, vol)) / (2 * h)
    assert abs(fd - bt.bs_delta(s, k, tau, vol)) < 1e-8


def test_one_option_by_hand(panel):
    """Every-day rebalancing, no cost: P&L equals a scalar loop written from scratch."""
    maturity = 21
    starts = bt.option_starts(panel, "test", maturity)
    i0 = int(starts[100])
    prices = panel["nifty"].to_numpy()[i0:i0 + maturity + 1]
    vix = panel["vix"].to_numpy()[i0:i0 + maturity + 1] / 100
    s = prices / prices[0]
    gain = 0.0
    for k in range(maturity):
        tau = (maturity - k) / 252
        d1 = (math.log(s[k]) + 0.5 * vix[k] ** 2 * tau) / (vix[k] * math.sqrt(tau))
        gain += norm_cdf(d1) * (s[k + 1] - s[k])
    tau0 = maturity / 252
    d1 = 0.5 * vix[0] * math.sqrt(tau0)
    premium = norm_cdf(d1) - norm_cdf(d1 - vix[0] * math.sqrt(tau0))
    labels = bt.label_variants(panel, ["vix_hyst"])
    stock, vol, _ = bt.build_paths(panel, np.array([i0]), maturity, labels)
    phi, _ = rb.held_positions(bt.target_deltas(stock, vol), rb.FixedFrequency(1))
    ours = rb.trading_outcome(phi, stock)["gain"][0]
    assert abs(ours - gain) < 1e-12
    assert abs(bt.bs_call(1.0, 1.0, tau0, vix[0]) - premium) < 1e-12


# --- 2. No look-ahead and clean periods ------------------------------------------


def test_option_windows_stay_inside_period(panel):
    for period in ("train", "test"):
        starts = bt.option_starts(panel, period, 21)
        split = panel["split"].to_numpy()
        assert all((split[s:s + 22] == period).all() for s in starts)
    assert panel.index[bt.option_starts(panel, "train", 21)[-1] + 21] <= pd.Timestamp("2018-12-31")
    assert panel.index[bt.option_starts(panel, "test", 21)[0]] >= pd.Timestamp("2019-01-01")


def test_positions_ignore_future_data(panel):
    """Changing prices, vols and labels after step k leaves positions up to k unchanged."""
    labels = bt.label_variants(panel, ["vix_hyst"])
    starts = bt.option_starts(panel, "test", 21)[:300]
    stock, vol, lab = bt.build_paths(panel, starts, 21, labels)
    k = 9
    stock2, vol2, lab2 = stock.copy(), vol.copy(), lab["vix_hyst"].copy()
    stock2[:, k + 1:] *= 1.3
    vol2[:, k + 1:] = 0.8
    lab2[:, k + 1:] = 1 - lab2[:, k + 1:]
    for rule, a, b in ((rb.VarianceTriggered(10, 2, 0.5), lab["vix_hyst"], lab2),
                       (rb.NoTradeBand(0.05), None, None), (rb.FixedFrequency(3), None, None)):
        p1, _ = rb.held_positions(bt.target_deltas(stock, vol), rule, a)
        p2, _ = rb.held_positions(bt.target_deltas(stock2, vol2), rule, b)
        assert np.array_equal(p1[:, :k + 1], p2[:, :k + 1])


def test_placebo_lag_is_causal(panel):
    v = bt.label_variants(panel, ["vix_hyst"])
    lab, lag = v["vix_hyst"], v[f"lag{bt.PLACEBO_LAG}:vix_hyst"]
    assert np.isnan(lag[: bt.PLACEBO_LAG]).all()
    assert np.array_equal(lag[bt.PLACEBO_LAG:], lab[: -bt.PLACEBO_LAG], equal_nan=True)
    flip = v["flip:vix_hyst"]
    ok = ~np.isnan(lab)
    assert np.array_equal(flip[ok], 1 - lab[ok])


# --- 3. Block bootstrap -------------------------------------------------------------


def test_block_bootstrap_iid_matches_formula():
    x = np.random.default_rng(0).normal(size=4000)
    iid = x.std(ddof=1) / math.sqrt(len(x))
    for block in (1, 20):
        assert abs(metrics.block_bootstrap_se(x, block) / iid - 1) < 0.1


def test_block_bootstrap_ar1_inflation():
    """AR(1) with phi = 0.8: long-run SE / iid SE -> sqrt((1+phi)/(1-phi)) = 3."""
    rng = np.random.default_rng(1)
    phi, n = 0.8, 40000
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    ratio = metrics.block_bootstrap_se(x, 200) / (x.std(ddof=1) / math.sqrt(n))
    assert 2.5 < ratio < 3.4, ratio


# --- 4. End to end in a Black--Scholes world ---------------------------------------


def test_pipeline_in_black_scholes_world():
    """GBM path, zero drift, VIX = true vol: the hedged seller breaks even on average,
    and halving the rebalancing frequency roughly doubles the error variance."""
    rng = np.random.default_rng(2)
    n, sigma = 30000, 0.2
    logret = -0.5 * sigma**2 / 252 + sigma / math.sqrt(252) * rng.standard_normal(n)
    nifty = 10000 * np.exp(np.cumsum(logret))
    dates = pd.bdate_range("2000-01-03", periods=n)
    fake = pd.DataFrame({"split": "test", "nifty": nifty, "vix": 100 * sigma,
                         "lab": (rng.random(n) < 0.3).astype(float)}, index=dates)
    res = bt.evaluate_period(fake, "test", 21, ["lab"])
    err = {m: res["premium"] + res[f"gain|fixed/every={m}"] - res["payoff"] for m in (1, 2, 21)}
    se = metrics.block_bootstrap_se(err[1], 42)
    assert abs(err[1].mean()) < 4 * se
    ratio = err[2].var() / err[1].var()
    assert 1.6 < ratio < 2.4, ratio
    assert err[21].var() > 5 * err[1].var()  # static hedge is far worse
