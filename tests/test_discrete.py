"""Correctness checks for src/ (run: python -m pytest tests -q).

Each test compares our code with something independent of it: the unmodified
upstream code, a brute-force loop, an exact identity, or a leading-order
asymptotic formula derived in the test's docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr

from src import hedge_paths, rebalancing as rb
from src.experiment_discrete import CUTOFF, MATURITY, draw_sources, setup_hedge
from src.upstream import base, hedging

STEPS = 128
PATHS = 400


@pytest.fixture(scope="module")
def market_and_hedge():
    model = base.Model()
    lambdas, weights, riccati, capital, vol = setup_hedge(model, STEPS)
    rng = np.random.default_rng(np.random.SeedSequence(7))
    market = base.simulate_original_market(
        model, MATURITY, STEPS, *draw_sources(rng, PATHS, STEPS)
    )
    return model, market, lambdas, weights, riccati, capital, vol


# --- 1. Agreement with the upstream code ---------------------------------------


def test_gkw_holdings_reproduce_upstream_gain(market_and_hedge):
    """Rebalancing every step must give exactly the upstream GKW gain."""
    model, market, lambdas, weights, riccati, capital, _ = market_and_hedge
    up_capital, up_gain = hedging.evaluate_policy(
        model, MATURITY, market, lambdas, weights, CUTOFF, 0.0, riccati
    )
    our_capital, theta = hedge_paths.kernel_hedge_holdings(
        model, MATURITY, market, lambdas, weights, CUTOFF, 0.0, riccati
    )
    phi, _ = rb.held_positions(theta, rb.FixedFrequency(1))
    gain = rb.trading_outcome(phi, np.asarray(market["stock"]))["gain"]
    assert our_capital == up_capital == capital
    assert np.max(np.abs(gain - up_gain)) < 1e-12


def test_shifted_kernel_holdings_reproduce_upstream(market_and_hedge):
    """Same check for a shifted kernel (epsilon = T/64), covering that branch."""
    model, market, lambdas, weights, _, _, _ = market_and_hedge
    eps = MATURITY / 64
    ric = base.solve_riccati_product_integration(
        model, 0.5 + 1j * lambdas, MATURITY, STEPS, eps
    )
    _, up_gain = hedging.evaluate_policy(model, MATURITY, market, lambdas, weights, 40.0, eps, ric)
    _, theta = hedge_paths.kernel_hedge_holdings(model, MATURITY, market, lambdas, weights, 40.0, eps, ric)
    gain = rb.trading_outcome(theta, np.asarray(market["stock"]))["gain"]
    assert np.max(np.abs(gain - up_gain)) < 1e-12


def test_bs_holdings_reproduce_upstream_loop(market_and_hedge):
    """Our BS delta equals the inline loop in the upstream script."""
    _, market, *_, vol = market_and_hedge
    stock = np.asarray(market["stock"])
    dstock = np.diff(stock, axis=1)
    up_gain = np.zeros(len(stock))
    for k in range(STEPS):  # copied from hedging_performance_experiment.py
        tau = MATURITY - k * MATURITY / STEPS
        stdev = vol * math.sqrt(tau)
        d1 = np.log(stock[:, k]) / stdev + 0.5 * stdev
        up_gain += ndtr(d1) * dstock[:, k]
    theta = hedge_paths.black_scholes_delta_holdings(stock, MATURITY, vol)
    gain = rb.trading_outcome(theta, stock)["gain"]
    assert np.max(np.abs(gain - up_gain)) < 1e-12


def test_gkw_holdings_are_sensible(market_and_hedge):
    """Loose sanity bounds on the GKW holdings.

    The variance-optimal holding is NOT confined to [0, 1]: with rho < 0 it
    equals the stock delta plus a negative vega/correlation correction, which
    can make it slightly negative for out-of-the-money paths. We checked that
    such values persist when the Fourier cutoff is raised from 80 to 320, so
    they are not only truncation error (see notes/robustness_log.md). Near
    maturity the R = 80 holdings carry visible truncation error (up to ~0.2
    shares in the last 10% of the life), hence the wide bounds.
    """
    model, market, lambdas, weights, riccati, *_ = market_and_hedge
    _, theta = hedge_paths.kernel_hedge_holdings(
        model, MATURITY, market, lambdas, weights, CUTOFF, 0.0, riccati
    )
    assert np.all(np.isfinite(theta))
    assert np.all((theta[:, 0] > 0.0) & (theta[:, 0] < 1.0))  # at t_0 the ATM hedge is a fraction of a share
    assert theta.min() > -0.25 and theta.max() < 1.25


# --- 2. Rules against brute-force loops ----------------------------------------


def brute_force(target, should_trade):
    """Reference implementation: explicit loop over paths and times."""
    phi = np.empty_like(target)
    for p in range(target.shape[0]):
        held = None
        for k in range(target.shape[1]):
            if k == 0 or should_trade(p, k, held, target[p, k]):
                held = target[p, k]
            phi[p, k] = held
    return phi


@pytest.fixture(scope="module")
def random_target():
    rng = np.random.default_rng(1)
    target = np.cumsum(rng.normal(scale=0.03, size=(30, 64)), axis=1) + 0.5
    proxy = rng.lognormal(mean=-4.8, sigma=0.5, size=(30, 64))
    return target, proxy


@pytest.mark.parametrize("every", [1, 3, 8, 64, 100])
def test_fixed_frequency(random_target, every):
    target, _ = random_target
    phi, _ = rb.held_positions(target, rb.FixedFrequency(every))
    expected = brute_force(target, lambda p, k, held, t: k % every == 0)
    assert np.array_equal(phi, expected)


def test_variance_triggered(random_target):
    target, proxy = random_target
    thr = float(np.median(proxy))
    rule = rb.VarianceTriggered(16, 4, thr)
    phi, _ = rb.held_positions(target, rule, proxy)
    expected = brute_force(
        target, lambda p, k, held, t: k % 16 == 0 or (k % 4 == 0 and proxy[p, k] > thr)
    )
    assert np.array_equal(phi, expected)
    # Degenerate cases: infinite threshold = calm calendar; zero = stress calendar.
    calm, _ = rb.held_positions(target, rb.VarianceTriggered(16, 4, np.inf), proxy)
    assert np.array_equal(calm, rb.held_positions(target, rb.FixedFrequency(16))[0])
    stress, _ = rb.held_positions(target, rb.VarianceTriggered(16, 4, -1.0), proxy)
    assert np.array_equal(stress, rb.held_positions(target, rb.FixedFrequency(4))[0])


@pytest.mark.parametrize("width", [0.0, 0.02, 0.1])
def test_band(random_target, width):
    target, _ = random_target
    phi, _ = rb.held_positions(target, rb.NoTradeBand(width))
    expected = brute_force(target, lambda p, k, held, t: abs(t - held) > width)
    assert np.array_equal(phi, expected)
    if width == 0.0:
        assert np.array_equal(phi, target)


def test_rules_are_predictable(random_target):
    """Changing data after t_k must not change positions up to t_k."""
    target, proxy = random_target
    k = 30
    target2, proxy2 = target.copy(), proxy.copy()
    target2[:, k + 1:] += 1.0
    proxy2[:, k + 1:] *= 10.0
    for rule in (rb.FixedFrequency(4), rb.VarianceTriggered(16, 4, float(np.median(proxy))),
                 rb.NoTradeBand(0.02)):
        a, _ = rb.held_positions(target, rule, proxy)
        b, _ = rb.held_positions(target2, rule, proxy2)
        assert np.array_equal(a[:, : k + 1], b[:, : k + 1])


# --- 3. Gains and costs: exact identities ----------------------------------------


def test_static_hedge_gain(market_and_hedge):
    """Rebalancing only at t_0 gives gain theta_0 (S_T - S_0)."""
    model, market, lambdas, weights, riccati, *_ = market_and_hedge
    stock = np.asarray(market["stock"])
    _, theta = hedge_paths.kernel_hedge_holdings(
        model, MATURITY, market, lambdas, weights, CUTOFF, 0.0, riccati
    )
    phi, _ = rb.held_positions(theta, rb.FixedFrequency(STEPS))
    out = rb.trading_outcome(phi, stock)
    assert np.allclose(out["gain"], theta[:, 0] * (stock[:, -1] - stock[:, 0]), atol=1e-14)
    assert np.all(out["trades"] == 0) and np.all(out["notional_rebalance"] == 0)


def test_costs_by_hand():
    stock = np.array([[1.0, 1.1, 0.9, 1.2]])
    phi = np.array([[0.5, 0.7, 0.7]])
    out = rb.trading_outcome(phi, stock)
    assert math.isclose(out["gain"][0], 0.5 * 0.1 + 0.7 * -0.2 + 0.7 * 0.3)
    assert math.isclose(out["notional_initial"][0], 0.5)
    assert math.isclose(out["notional_rebalance"][0], 1.1 * 0.2)  # trade at t_1 only
    assert math.isclose(out["notional_terminal"][0], 1.2 * 0.7)
    assert out["trades"][0] == 1
    kappa = 1e-3
    assert math.isclose(rb.total_cost(out, kappa)[0], kappa * (0.5 + 0.22 + 0.84))
    assert rb.total_cost(out, 0.0)[0] == 0.0
    assert math.isclose(rb.total_cost(out, kappa, False, False)[0], kappa * 0.22)


def test_spot_variance_proxy_uses_only_past(market_and_hedge):
    model, market, *_ = market_and_hedge
    A = np.asarray(market["integrated_variance"])
    dA = np.asarray(market["dintegrated_variance"])
    h = MATURITY / STEPS
    one = rb.spot_variance_proxy(market, MATURITY, model.v0, window=1)
    assert np.all(one[:, 0] == model.v0)
    assert np.allclose(one[:, 5], dA[:, 4] / h)  # increment over [t_4, t_5]
    w = 8
    prox = rb.spot_variance_proxy(market, MATURITY, model.v0, window=w)
    assert np.allclose(prox[:, 3], (A[:, 3] - A[:, 0]) / (3 * h))  # short history
    assert np.allclose(prox[:, 20], (A[:, 20] - A[:, 12]) / (w * h))
    assert np.allclose(prox[:, 20], dA[:, 12:20].mean(axis=1) / h)
    # Altering the clock after t_k leaves the proxy up to t_k unchanged.
    k = 40
    altered = dict(market)
    A2 = A.copy()
    A2[:, k + 1:] += 1.0
    altered["integrated_variance"] = A2
    prox2 = rb.spot_variance_proxy(altered, MATURITY, model.v0, window=w)
    assert np.array_equal(prox[:, : k + 1], prox2[:, : k + 1])


# --- 4. Analysis: matched-turnover comparison ------------------------------------


def test_matched_turnover_estimator(monkeypatch):
    """A copy of a fixed rule must give exactly zero; a rule whose notional sits
    between two fixed rules gets the interpolated (mixture) benchmark."""
    from src import analysis

    rng = np.random.default_rng(5)
    n = 1000
    payoff = rng.random(n)
    gains = {"fixed/every=1": rng.normal(size=n), "fixed/every=2": rng.normal(size=n)}
    notion = {"fixed/every=1": np.full(n, 4.0), "fixed/every=2": np.full(n, 2.0)}
    data = {"payoff": payoff}
    for r in gains:
        data[f"gain|GKW|{r}"] = gains[r]
        data[f"notional|GKW|{r}"] = notion[r].astype(np.float32)
    # Copy of every=2, and a rule with notional 3 (midway, so p = 0.5).
    data["gain|GKW|band/half_width=0.1"] = gains["fixed/every=2"]
    data["notional|GKW|band/half_width=0.1"] = notion["fixed/every=2"].astype(np.float32)
    data["gain|GKW|band/half_width=0.2"] = np.zeros(n)
    data["notional|GKW|band/half_width=0.2"] = np.full(n, 3.0, dtype=np.float32)
    config = {"gkw_capital": 0.1}
    monkeypatch.setattr(analysis, "COMBOS", [("GKW", r) for r in
                                             ["fixed/every=1", "fixed/every=2",
                                              "band/half_width=0.1", "band/half_width=0.2"]])
    copy = analysis.matched_turnover(data, config, "GKW", "band/half_width=0.1", 0.0)
    assert copy["mse_minus_matched_fixed"] == 0.0
    mid = analysis.matched_turnover(data, config, "GKW", "band/half_width=0.2", 1e-3)
    e = lambda g, k_notional: 0.1 + g - 1e-3 * k_notional - payoff
    expected = np.mean(e(np.zeros(n), 3.0) ** 2) - 0.5 * (
        np.mean(e(gains["fixed/every=2"], 2.0) ** 2) + np.mean(e(gains["fixed/every=1"], 4.0) ** 2))
    assert math.isclose(mid["mse_minus_matched_fixed"], expected, rel_tol=1e-6)


# --- 5. Independent check of the discretisation engine in Black--Scholes -------


def test_gbm_discrete_hedging_error_matches_leading_order():
    """In GBM with the true-vol BS delta, the discrete-hedging error variance
    has the leading-order form (Ito/Taylor expansion of the call price C):

        increment of error over [t_k, t_{k+1}] ~ (1/2) Gamma_k S_k^2 (sigma^2 h - R_k^2),
        R_k = return over the step, so Var(R_k^2) ~ 2 sigma^4 h^2, hence
        Var(e) ~ sum_k (1/2) E[Gamma_k^2 S_k^4] sigma^4 h^2      (O(h)).

    We check (i) Var(e) is close to this prediction for fine grids and (ii) the
    log-log slope of Var(e) against h is close to 1. This validates the
    rebalancing/gain code on a model where the answer is known.
    """
    sigma, T, n_fine, paths = 0.2, 0.25, 512, 20_000
    rng = np.random.default_rng(3)
    h = T / n_fine
    logret = -0.5 * sigma**2 * h + sigma * math.sqrt(h) * rng.standard_normal((paths, n_fine))
    stock = np.exp(np.concatenate([np.zeros((paths, 1)), np.cumsum(logret, axis=1)], axis=1))
    payoff = np.maximum(stock[:, -1] - 1.0, 0.0)
    price = hedging.black_scholes_call(1.0, 1.0, T, sigma)
    delta = hedge_paths.black_scholes_delta_holdings(stock, T, sigma)
    gamma = hedge_paths.black_scholes_gamma(stock, T, sigma)

    variances, predictions, hs = [], [], []
    for every in (32, 16, 8, 4):
        phi, _ = rb.held_positions(delta, rb.FixedFrequency(every))
        error = price + rb.trading_outcome(phi, stock)["gain"] - payoff
        variances.append(np.var(error))
        H = every * h  # rebalancing interval
        idx = np.arange(0, n_fine, every)
        g, s = gamma[:, idx], stock[:, idx]
        predictions.append(np.sum(np.mean(0.5 * g**2 * s**4, axis=0)) * sigma**4 * H**2)
        hs.append(H)
    ratios = np.array(variances) / np.array(predictions)
    slope = np.polyfit(np.log(hs), np.log(variances), 1)[0]
    # Leading-order formula should be accurate to ~10% once there are >= 64 dates.
    assert np.all(np.abs(ratios[1:] - 1.0) < 0.12), ratios
    assert 0.85 < slope < 1.15, slope
