"""Heston (1993) European option pricing by Fourier inversion.

Model under Q (rate r, dividend yield q):
    dS = (r-q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1,W2> = rho dt

Uses the "little Heston trap" characteristic function (Albrecher et al. 2007),
which is continuous in the complex plane, and the Gil-Pelaez inversion
    C = S e^{-qT} P1 - K e^{-rT} P2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats


@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot/variance correlation

    @property
    def feller(self) -> float:
        """2*kappa*theta - xi^2 ; positive means variance never hits zero."""
        return 2 * self.kappa * self.theta - self.xi ** 2


def _cf(u, T, S, r, q, p: HestonParams):
    """Characteristic function E[exp(i u ln S_T)] (little-trap form)."""
    i = 1j
    x = np.log(S)
    a = p.kappa * p.theta
    b = p.kappa - p.rho * p.xi * i * u
    d = np.sqrt(b * b + p.xi ** 2 * (u * u + i * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (i * u * (x + (r - q) * T)
         + a / p.xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g))))
    D = (b - d) / p.xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0)


def call_price(S, K, T, r, q, p: HestonParams, umax: float = 200.0) -> float:
    """European call price."""
    lnK = np.log(K)

    def integrand(u, j):
        if j == 1:
            num = _cf(u - 1j, T, S, r, q, p) / _cf(-1j, T, S, r, q, p)
        else:
            num = _cf(u, T, S, r, q, p)
        return np.real(np.exp(-1j * u * lnK) * num / (1j * u))

    P = []
    for j in (1, 2):
        val, _ = integrate.quad(integrand, 1e-10, umax, args=(j,), limit=400)
        P.append(0.5 + val / np.pi)
    return S * np.exp(-q * T) * P[0] - K * np.exp(-r * T) * P[1]


def put_price(S, K, T, r, q, p: HestonParams) -> float:
    """European put via put-call parity."""
    return call_price(S, K, T, r, q, p) - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_call(S, K, T, r, q, sigma) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def implied_vol(price, S, K, T, r, q) -> float:
    """Black-Scholes implied volatility by Brent root-finding."""
    f = lambda s: bs_call(S, K, T, r, q, s) - price
    return optimize.brentq(f, 1e-4, 5.0)


def mc_call(S, K, T, r, q, p: HestonParams, n_paths=200_000, n_steps=400, seed=0):
    """Full-truncation Euler Monte Carlo (Lord et al. 2010). Returns (price, s.e.)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    lnS = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        lnS += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
    pay = np.exp(-r * T) * np.maximum(np.exp(lnS) - K, 0.0)
    return pay.mean(), pay.std(ddof=1) / np.sqrt(n_paths)
