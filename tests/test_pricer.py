import numpy as np
import pytest

from heston.pricer import (HestonParams, bs_call, call_price, implied_vol,
                           mc_call, put_price)

P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
S, r, q = 100.0, 0.03, 0.01


def test_reduces_to_black_scholes_when_vol_of_vol_tiny():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-3, rho=0.0)
    for K in (80, 100, 120):
        assert call_price(S, K, 1.0, r, q, p) == pytest.approx(
            bs_call(S, K, 1.0, r, q, 0.2), abs=1e-4)


def test_put_call_parity_and_bounds():
    K, T = 105.0, 0.5
    c, pu = call_price(S, K, T, r, q, P), put_price(S, K, T, r, q, P)
    assert c - pu == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)
    assert max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) <= c <= S * np.exp(-q * T)


@pytest.mark.parametrize("K", [90.0, 100.0, 115.0])
def test_matches_monte_carlo(K):
    f = call_price(S, K, 1.0, r, q, P)
    mc, se = mc_call(S, K, 1.0, r, q, P, n_paths=200_000, n_steps=300, seed=1)
    assert abs(f - mc) < 4 * se + 0.03  # 0.03 allows for Euler discretisation bias


def test_negative_rho_gives_downside_skew():
    iv = lambda K: implied_vol(call_price(S, K, 1.0, r, q, P), S, K, 1.0, r, q)
    assert iv(85) > iv(100) > iv(115)
