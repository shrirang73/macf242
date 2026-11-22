import numpy as np

from heston import simulate as sim
from heston.hedge import heston_deltas
from heston.pricer import HestonParams, call_price

P = HestonParams(v0=0.03, kappa=3.0, theta=0.03, xi=0.6, rho=-0.6)


def test_implied_vol_inverts_black():
    m = np.array([0.9, 1.0, 1.1]); tau = np.full(3, 21 / 252); sig = np.array([0.25, 0.18, 0.15])
    from scipy.special import ndtr
    sd = sig * np.sqrt(tau); d1 = (-np.log(m) + 0.5 * sd ** 2) / sd
    price = ndtr(d1) - m * ndtr(d1 - sd)
    assert np.allclose(sim.implied_vol_vec(price, m, tau, np.full(3, 0.4)), sig, atol=1e-8)


def test_per_path_deltas_match_scalar_version():
    # vectorised per-path variances give exactly the scalar-parameter results
    m = np.array([0.95, 1.0, 1.04]); v = np.array([0.01, 0.03, 0.08]); tau = 10 / 252
    price, d, dmv = sim.deltas_paths(m, tau, P, v)
    for i in range(3):
        pi = HestonParams(v[i], P.kappa, P.theta, P.xi, P.rho)
        ref_d, ref_mv = heston_deltas(m[i:i + 1], np.array([tau]), pi)
        assert abs(d[i] - ref_d[0]) < 1e-6 and abs(dmv[i] - ref_mv[0]) < 1e-6
        assert abs(price[i] - call_price(1.0, m[i], tau, 0, 0, pi, umax=2000)) < 1e-6


def test_simulated_call_price_matches_formula():
    sim.N_PATHS, sim.SUB = 40_000, 10
    F, v = sim.simulate_paths(P, np.random.default_rng(1))
    pay = np.maximum(F[:, -1] - 1.0, 0.0)
    ref = call_price(1.0, 1.0, sim.DAYS / 252, 0, 0, P)
    assert abs(pay.mean() - ref) < 4 * pay.std() / np.sqrt(len(pay)) + 2e-4
