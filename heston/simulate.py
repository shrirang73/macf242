"""Stage 4: hedging in a simulated Heston world (model-consistent benchmark).

Question: if NIFTY really followed Heston, how much would the minimum-variance
delta gain over the Black-Scholes and plain Heston deltas, and how does the
rebalancing interval matter? Comparing this with the real-data results tells us
whether the small real-data gain is a property of the hedge or of model error.

Design (common random numbers across hedges and frequencies):
  * Heston paths for the forward F under Q (zero rates), full-truncation Euler
    with 20 sub-steps per trading day (Lord et al. 2010).
  * Sell an ATM call with 21 trading days to expiry at its Heston price.
  * Hedge every h days (h = 0.25, 1, 2, 5) with
       bs_own    : Black-76 delta at the option's own Heston implied vol at that time
                   (what a trader reading the market smile would use),
       heston    : dC/dF with variance held fixed,
       heston_mv : dC/dF + rho*xi/F * dC/dv   (minimum-variance delta, eq. (2)).
  * Error e = C0 - (F_T-K)^+ + sum phi_k (F_{k+1}-F_k), per unit of F0, no costs.

Two parameter sets:  'calibrated' = median of our daily calibrations (2019-26),
'time-series' = the team's VIX/NIFTY estimate (kappa, theta, xi, rho) with v0 = theta.

Usage:  python -m heston.simulate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .calib import call_vec
from .data import ROOT, YEAR
from .hedge import heston_deltas
from .pricer import HestonParams

N_PATHS, DAYS, SUB = 10_000, 21, 20
FREQS = [0.25, 1, 2, 5]           # rebalancing interval in trading days
SEED = 20261119


def scenarios() -> dict:
    p = pd.read_csv(ROOT / "results" / "params.csv", parse_dates=["date"])
    t = p[p.date >= "2019-01-01"]
    med = t[["v0", "kappa", "theta", "xi", "rho"]].median()
    return {
        "calibrated": HestonParams(*med.astype(float)),
        "time-series": HestonParams(v0=0.0337, kappa=8.60, theta=0.0337, xi=0.341, rho=-0.526),
    }


def simulate_paths(p: HestonParams, rng):
    """Return F (N, DAYS*SUB+1) and v on the same grid; F0 = 1."""
    n = DAYS * SUB
    dt = 1.0 / YEAR / SUB
    lnF = np.zeros((N_PATHS, n + 1))
    v = np.empty((N_PATHS, n + 1))
    v[:, 0] = p.v0
    for i in range(n):
        z1 = rng.standard_normal(N_PATHS)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * rng.standard_normal(N_PATHS)
        vp = np.maximum(v[:, i], 0.0)
        lnF[:, i + 1] = lnF[:, i] - 0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v[:, i + 1] = v[:, i] + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
    return np.exp(lnF), np.maximum(v, 0.0)


def implied_vol_vec(price, m, tau, guess):
    """Vectorised Black-76 implied vol (F = 1, strike m) by safeguarded Newton."""
    lo, hi = np.full_like(price, 1e-3), np.full_like(price, 3.0)
    s = np.clip(guess, 0.02, 2.0)
    for _ in range(60):
        sd = s * np.sqrt(tau)
        d1 = (-np.log(m) + 0.5 * sd ** 2) / sd
        c = ndtr(d1) - m * ndtr(d1 - sd)
        f = c - price
        lo = np.where(f < 0, s, lo)
        hi = np.where(f > 0, s, hi)
        vega = np.sqrt(tau) * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        step = s - f / np.maximum(vega, 1e-12)
        s = np.where((step > lo) & (step < hi), step, 0.5 * (lo + hi))
    return s


def deltas_paths(m, tau, p: HestonParams, v, h=1e-3, chunk=2500):
    """(price, delta, MV delta) per path; m and v arrays, scalar tau.

    Same formulas as hedge.heston_deltas, but with a different current variance on
    every path: delta = c - m c'(m) (homogeneity), dC/dv by a 5% central bump."""
    out = [np.empty(len(m)) for _ in range(3)]
    for s in range(0, len(m), chunk):
        sl = slice(s, s + chunk)
        mm, vv = m[sl], v[sl]
        tt = np.full(len(mm), tau)
        c = call_vec(1.0, mm, tt, p, v0=vv)
        cp = (call_vec(1.0, mm * (1 + h), tt, p, v0=vv)
              - call_vec(1.0, mm * (1 - h), tt, p, v0=vv)) / (2 * mm * h)
        hv = 0.05 * vv
        dcdv = (call_vec(1.0, mm, tt, p, v0=vv + hv) - call_vec(1.0, mm, tt, p, v0=vv - hv)) / (2 * hv)
        delta = c - mm * cp
        out[0][sl], out[1][sl], out[2][sl] = c, delta, delta + p.rho * p.xi * dcdv
    return out


def hedge_errors(p: HestonParams, F, v, h):
    """Hedging errors (per unit F0) for the three hedges at interval h days."""
    every = int(round(h * SUB))
    idx = np.arange(0, DAYS * SUB + 1, every)
    if idx[-1] != DAYS * SUB:
        idx = np.append(idx, DAYS * SUB)
    K = 1.0
    T = DAYS / YEAR
    c0 = float(call_vec(1.0, np.array([K]), np.array([T]), p)[0])
    payoff = np.maximum(F[:, -1] - K, 0.0)
    gains = {k: np.zeros(N_PATHS) for k in ("bs_own", "heston", "heston_mv")}
    for a, b in zip(idx[:-1], idx[1:]):
        tau = max(T - a / SUB / YEAR, 1.0 / YEAR / SUB)
        Fa, va = F[:, a], np.maximum(v[:, a], 1e-6)
        m = K / Fa
        dF = F[:, b] - Fa
        # Heston price and deltas with each path's own current variance (vectorised)
        price, dh, dmv = deltas_paths(m, tau, p, va)
        iv = implied_vol_vec(price, m, tau, np.sqrt(va))
        sd = iv * np.sqrt(tau)
        dbs = ndtr((-np.log(m) + 0.5 * sd ** 2) / sd)
        gains["bs_own"] += dbs * dF
        gains["heston"] += dh * dF
        gains["heston_mv"] += dmv * dF
    return {k: c0 - payoff + g for k, g in gains.items()}, c0


def bootstrap_ratio(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    point = 100 * (a.var() / b.var() - 1)
    draws = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(a), len(a))
        draws[i] = 100 * (a[j].var() / b[j].var() - 1)
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    rows = []
    for name, p in scenarios().items():
        rng = np.random.default_rng(SEED)
        F, v = simulate_paths(p, rng)
        for h in FREQS:
            e, c0 = hedge_errors(p, F, v, h)
            for hedge, x in e.items():
                r = dict(scenario=name, interval_days=h, hedge=hedge, sd_bp=1e4 * x.std(ddof=1),
                         mean_bp=1e4 * x.mean(), premium_bp=1e4 * c0,
                         **{f"param_{k}": getattr(p, k) for k in ("v0", "kappa", "theta", "xi", "rho")})
                if hedge != "bs_own":
                    pt, lo, hi = bootstrap_ratio(x, e["bs_own"])
                    r.update(vs_bs_pct=pt, vs_bs_lo=lo, vs_bs_hi=hi)
                rows.append(r)
            print(f"{name:12s} h={h:<5} " + "  ".join(f"{k}={1e4 * x.std():6.1f}" for k, x in e.items()), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "simulation.csv", index=False)
    return out


if __name__ == "__main__":
    main()
