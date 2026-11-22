import numpy as np
import pytest

from heston.calib import calibrate, call_vec, model_iv
from heston.pricer import HestonParams, call_price

TRUE = HestonParams(v0=0.03, kappa=2.5, theta=0.04, xi=0.7, rho=-0.65)


def test_vectorised_matches_adaptive_quadrature():
    F = 100.0
    K = np.array([85.0, 100.0, 118.0, 85.0, 100.0, 118.0])
    T = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    got = call_vec(F, K, T, TRUE)
    ref = [call_price(F, k, t, 0.0, 0.0, TRUE, umax=2000) for k, t in zip(K, T)]
    assert np.allclose(got, ref, atol=1e-5)


def test_calibration_recovers_fit_on_synthetic_chain():
    F = 100.0
    K = np.concatenate([F * np.exp(np.linspace(-0.12, 0.12, 13))] * 3)
    T = np.repeat([0.08, 0.16, 0.24], 13)
    iv = model_iv(F, K, T, TRUE)
    p, rmse, ok = calibrate(F, K, T, iv, prior=None, ridge=0.0)
    assert rmse < 5e-4                       # < 5 bp of vol on noise-free data
    assert p.rho < -0.3 and p.xi > 0.3       # sign and rough size of the skew
