"""Full holding paths theta(t_k) for the hedges compared in the project.

The upstream function ``hedging.evaluate_policy`` computes the variance-optimal
(GKW) holding theta(t_k) at every grid time but only returns the accumulated
gain sum_k theta(t_k) (S_{k+1} - S_k). For discrete rebalancing we need the
holdings themselves, so ``kernel_hedge_holdings`` repeats the upstream loop
line for line and stores theta instead of summing it. Keeping the arithmetic
identical lets the test suite check that, when we rebalance at every grid step,
our gain equals the upstream gain to rounding error.

Notation (paper, Sec. 5.1): grid t_k = k h, h = T/N; theta(t_k) is the number
of shares held over (t_k, t_{k+1}] and depends only on information up to t_k,
so it is predictable.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .upstream import base


def kernel_hedge_holdings(
    model: base.Model,
    maturity: float,
    market: dict,
    lambdas: np.ndarray,
    weights: np.ndarray,
    cutoff: float,
    epsilon: float,
    riccati: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray]:
    """Return the Fourier initial capital and the holdings matrix theta.

    Mirrors ``hedging_performance_experiment.evaluate_policy`` (upstream).
    ``epsilon`` is the kernel shift (0 = the original fractional kernel) and
    ``cutoff`` the Fourier truncation R. Output ``theta`` has shape
    (paths, steps); column k is the holding over (t_k, t_{k+1}].
    """
    x = np.asarray(market["x"])
    stock = np.asarray(market["stock"])
    driver = np.asarray(market["driver"])
    dz = np.asarray(market["dz"])
    steps = stock.shape[1] - 1
    # Integrated history of the (possibly shifted) kernel against the fixed
    # driver Z: this is the information the hedger uses at each t_k.
    history = base.integrated_history_statistic(model, driver, maturity, steps, epsilon)
    dhistory = np.diff(history, axis=1)
    psi, forcing = riccati
    u = 0.5 + 1j * lambdas  # Lewis contour Re(w) = 1/2
    coefficients = base.call_coefficients(lambdas, weights)
    mask = lambdas <= cutoff + 1.0e-12
    u0 = base.initial_memory(model, forcing, maturity, steps)
    capital = base.capital_by_cutoff(u0, coefficients, lambdas, np.array([cutoff]))[0]
    u_state = np.broadcast_to(u0, (len(stock), len(lambdas))).copy()
    theta = np.empty((len(stock), steps))

    for k in range(steps):
        remaining = steps - k
        # GKW projection coefficient for each Fourier node at time-to-go T-t_k.
        beta, _ = base.projection_beta(model, u, psi[remaining])
        transformed = base.clipped_exponential(u_state)
        h_over_s = np.exp((u[None, :] - 1.0) * x[:, k, None]) * transformed
        node_value = h_over_s * beta[None, :] * coefficients[None, :]
        theta[:, k] = 1.0 - np.real(np.sum(node_value[:, mask], axis=1))
        # Advance the log-memory U to t_{k+1} (paper eq. after (73)).
        u_state += (
            psi[remaining][None, :] * dz[:, k, None]
            - forcing[remaining][None, :] * dhistory[:, k, None]
        )
    return float(capital), theta


def black_scholes_delta_holdings(
    stock: np.ndarray, maturity: float, implied_vol: float, strike: float = 1.0
) -> np.ndarray:
    """Black--Scholes call delta N(d1) at each grid time (zero rate).

    Same formula as the upstream benchmark (which assumes strike 1); shape
    (paths, steps).
    """
    steps = stock.shape[1] - 1
    theta = np.empty((stock.shape[0], steps))
    for k in range(steps):
        tau = maturity - k * maturity / steps
        stdev = implied_vol * math.sqrt(tau)
        d1 = np.log(stock[:, k] / strike) / stdev + 0.5 * stdev
        theta[:, k] = ndtr(d1)
    return theta


def black_scholes_gamma(
    stock: np.ndarray, maturity: float, vol: float, strike: float = 1.0
) -> np.ndarray:
    """Black--Scholes call gamma at each grid time; used only in the GBM check."""
    steps = stock.shape[1] - 1
    gamma = np.empty((stock.shape[0], steps))
    for k in range(steps):
        tau = maturity - k * maturity / steps
        stdev = vol * math.sqrt(tau)
        d1 = np.log(stock[:, k] / strike) / stdev + 0.5 * stdev
        gamma[:, k] = np.exp(-0.5 * d1 * d1) / (
            math.sqrt(2.0 * math.pi) * stock[:, k] * stdev
        )
    return gamma
