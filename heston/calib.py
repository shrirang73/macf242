"""Fast vectorised Heston pricing on forwards and daily calibration.

Pricing uses r = q = 0 on the forward F (Black-76 convention of the data), with a
fixed Gauss-Legendre grid for the Gil-Pelaez integrals so a whole chain is
priced in one vectorised call. Accuracy is tested against adaptive quadrature.

Calibration: least squares on vega-scaled price errors (~ implied-vol errors),
over all monthly expiries of a day, warm-started from the previous day with a
small ridge towards it (kappa and theta are weakly identified from a few
maturities; the ridge keeps the parameter path stable and is part of the
documented method, not a tuned result).
"""
from __future__ import annotations

from dataclasses import astuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from .pricer import HestonParams
from .data import black_iv

_U_MAX, _N = 30.0, 128    # in units of 1/total-vol; error ~1e-10 (tests/test_calib.py)
_X, _W = np.polynomial.legendre.leggauss(_N)
U = 0.5 * _U_MAX * (_X + 1.0)       # nodes on (0, U_MAX)
W = 0.5 * _U_MAX * _W


def _cf(u, T, p: HestonParams, v0=None):
    """E[exp(i u ln(S_T/F))], little-trap form, zero drift.

    v0 overrides p.v0 and may be an array broadcastable with u (one variance per
    option); the characteristic function is exp(C + D v0), linear in v0 inside."""
    i = 1j
    a = p.kappa * p.theta
    b = p.kappa - p.rho * p.xi * i * u
    d = np.sqrt(b * b + p.xi ** 2 * (u * u + i * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = a / p.xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / p.xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * (p.v0 if v0 is None else v0))


def call_vec(F, K, T, p: HestonParams, v0=None):
    """Heston call prices for arrays K, T (same shape), scalar F; r = q = 0.

    Optional v0: array of current variances, one per option (simulation use)."""
    K, T = np.atleast_1d(K).astype(float), np.atleast_1d(T).astype(float)
    k = np.log(K / F)[:, None]                  # (n, 1)
    Tn = T[:, None]
    v = p.v0 if v0 is None else np.atleast_1d(v0).astype(float)[:, None]
    # Frequency grid scaled by each option's total volatility sqrt(avg var * T):
    # the integrand's decay scale is then the same for every maturity, so a small
    # fixed number of Gauss-Legendre nodes is accurate from 5 days to 1 year.
    scale = np.sqrt(0.5 * (v + p.theta) * Tn)
    u = U[None, :] / scale                      # (n, N)
    w = W[None, :] / scale
    cf2 = _cf(u, Tn, p, v)
    cf1 = _cf(u - 1j, Tn, p, v)                 # cf at -i is 1 when drift is zero
    ker = np.exp(-1j * u * k) / (1j * u)
    p1 = 0.5 + (w * np.real(ker * cf1)).sum(1) / np.pi
    p2 = 0.5 + (w * np.real(ker * cf2)).sum(1) / np.pi
    return F * p1 - K * p2


def vega_black(F, K, T, vol):
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    return F * np.sqrt(T) * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)


def model_iv(F, K, T, p: HestonParams):
    prices = call_vec(F, K, T, p)
    return np.array([black_iv(c, F, k, t) for c, k, t in zip(prices, K, T)])


LOWER = np.array([1e-4, 0.1, 1e-4, 0.05, -0.99])
UPPER = np.array([1.0, 15.0, 1.0, 3.0, 0.99])
DEFAULT = HestonParams(v0=0.025, kappa=3.0, theta=0.03, xi=0.6, rho=-0.6)


def _clip(x):
    return np.clip(x, LOWER + 1e-9, UPPER - 1e-9)


def calibrate(F, K, T, iv, prior: HestonParams | None = None, ridge: float = 0.02):
    """Fit Heston to one day's chain. Returns (params, rmse_iv, ok)."""
    vega = vega_black(F, K, T, iv)
    # Price of each option implied by its own IV (robust to the call/put conversion).
    from .data import black_call
    target = black_call(F, K, T, iv)
    scale = np.maximum(vega, 1e-8)
    x_prior = np.array(astuple(prior if prior is not None else DEFAULT))
    # parameters scaled so the ridge treats them comparably
    s = np.array([0.05, 5.0, 0.05, 1.0, 1.0])

    def resid(x):
        pm = HestonParams(*x)
        r = (call_vec(F, K, T, pm) - target) / scale          # ~ vol error
        pen = ridge * (x - x_prior) / s * np.sqrt(len(K)) * 0.01
        return np.concatenate([r, pen])

    def fit(x0):
        try:
            return least_squares(resid, _clip(x0), bounds=(LOWER, UPPER), x_scale=s, max_nfev=40)
        except Exception:
            return None

    def score(sol):
        pm = HestonParams(*sol.x)
        return float(np.sqrt(np.nanmean((model_iv(F, K, T, pm) - iv) ** 2)))

    # Warm start from the previous day; only if that fit is poor (> 2 vol points)
    # or there is no prior, also try the default start and keep the better fit.
    best, rmse = None, np.inf
    first = fit(x_prior if prior is not None else np.array(astuple(DEFAULT)))
    if first is not None:
        best, rmse = first, score(first)
    if best is None or rmse > 0.02:
        second = fit(np.array(astuple(DEFAULT)))
        if second is not None:
            r2 = score(second)
            if r2 < rmse:
                best, rmse = second, r2
    if best is None:
        return prior or DEFAULT, np.nan, False
    return HestonParams(*best.x), rmse, bool(np.isfinite(rmse))
