"""Discrete rebalancing rules and proportional transaction costs.

Setting. The market is simulated on a fine grid t_k = k h, k = 0..N. A hedge
supplies a *target* holding theta_k (shares over (t_k, t_{k+1}]); in the
continuous-time limit the hedger would hold theta_k at every step. A
rebalancing rule decides at which grid times the position is actually reset to
the target; in between the previous position phi is kept. The rule may only
use information available at t_k, so phi stays predictable.

For a short call hedged with initial capital x, the hedger's terminal P&L is

    e = x + sum_k phi_k (S_{k+1} - S_k) - C - (S_N - K)^+,

with proportional costs C = kappa * sum_k S_k |phi_k - phi_{k-1}| (phi_{-1} = 0,
so buying the initial position is charged) plus kappa * S_N |phi_{N-1}| for
liquidating the stock at maturity. ``kappa`` is a fraction of traded notional
(e.g. 5e-4 = 5 basis points). Costs are linear in kappa for a fixed rule, so we
store the traded notional and apply any kappa afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Rules. Each rule maps (target holdings, state) -> held positions phi.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedFrequency:
    """Rebalance to the target every ``every`` fine steps (t_0 included)."""

    every: int

    @property
    def label(self) -> str:
        return f"fixed/every={self.every}"


@dataclass(frozen=True)
class VarianceTriggered:
    """Regime-aware rule: rebalance more often when spot variance is high.

    At t_k the rule rebalances if k is a multiple of ``calm_every`` or if
    (k is a multiple of ``stress_every`` and the variance proxy at t_k exceeds
    ``threshold``). The proxy is the trailing realised variance from
    ``spot_variance_proxy``, which is known at t_k (no look-ahead).
    """

    calm_every: int
    stress_every: int
    threshold: float

    @property
    def label(self) -> str:
        return (
            f"variance/calm={self.calm_every},stress={self.stress_every},"
            f"thr={self.threshold:.4g}"
        )


@dataclass(frozen=True)
class NoTradeBand:
    """Rebalance to the target only when |target - held| > ``half_width`` shares.

    Classical benchmark for hedging with proportional costs (cf. Whalley &
    Wilmott 1997, who trade to the band edge; here we trade to the target).
    """

    half_width: float

    @property
    def label(self) -> str:
        return f"band/half_width={self.half_width:g}"


Rule = FixedFrequency | VarianceTriggered | NoTradeBand


def rebalance_mask(rule: Rule, steps: int, spot_variance: np.ndarray | None = None,
                   paths: int | None = None) -> np.ndarray:
    """Boolean (paths, steps) array: True where the rule resets to the target.

    Only for the calendar/state rules; the band rule depends on the held
    position and is handled in ``held_positions``.
    """
    k = np.arange(steps)
    if isinstance(rule, FixedFrequency):
        if rule.every < 1:
            raise ValueError("every must be >= 1")
        row = (k % rule.every) == 0
        n_paths = paths if paths is not None else 1
        return np.broadcast_to(row, (n_paths, steps)).copy()
    if isinstance(rule, VarianceTriggered):
        if spot_variance is None:
            raise ValueError("VarianceTriggered needs the spot-variance proxy")
        if rule.stress_every < 1 or rule.calm_every < rule.stress_every:
            raise ValueError("need 1 <= stress_every <= calm_every")
        calm = (k % rule.calm_every) == 0
        stress_slot = (k % rule.stress_every) == 0
        mask = calm[None, :] | (stress_slot[None, :] & (spot_variance > rule.threshold))
        mask[:, 0] = True  # always set up the hedge at t_0
        return mask
    raise TypeError(f"no calendar mask for rule {rule!r}")


def held_positions(target: np.ndarray, rule: Rule,
                   spot_variance: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``rule`` to the target holdings.

    Returns (phi, traded) where phi has the shape of ``target`` and ``traded``
    is a boolean array marking the grid times at which a trade happens.
    """
    paths, steps = target.shape
    if isinstance(rule, NoTradeBand):
        phi = np.empty_like(target)
        traded = np.zeros(target.shape, dtype=bool)
        phi[:, 0] = target[:, 0]
        traded[:, 0] = True
        for k in range(1, steps):
            move = np.abs(target[:, k] - phi[:, k - 1]) > rule.half_width
            phi[:, k] = np.where(move, target[:, k], phi[:, k - 1])
            traded[:, k] = move
        return phi, traded

    mask = rebalance_mask(rule, steps, spot_variance, paths)
    # Index of the most recent rebalancing time <= k (forward fill).
    last = np.where(mask, np.arange(steps)[None, :], 0)
    np.maximum.accumulate(last, axis=1, out=last)
    phi = np.take_along_axis(target, last, axis=1)
    return phi, mask


def spot_variance_proxy(market: dict, maturity: float, v0: float,
                        window: int = 1) -> np.ndarray:
    """Trailing realised-variance proxy known at t_k (no look-ahead).

    proxy_k = (A_k - A_{k-w}) / (w h) with w = min(window, k), and proxy_0 = v0,
    where A is the integrated-variance clock of the upstream simulation
    (``market['integrated_variance']``, a running maximum, so A is
    non-decreasing).

    A one-step window (w = 1) is a poor regime signal here: the running-max clock
    is flat on about 62% of fine steps (upstream diagnostic
    ``running_max_adjustment_fraction``), so the one-step proxy is exactly zero
    most of the time. A window of T/16 is zero on only ~4% of (path, time) points.
    """
    A = np.asarray(market["integrated_variance"])
    paths, steps = A.shape[0], A.shape[1] - 1
    if window < 1:
        raise ValueError("window must be >= 1")
    h = maturity / steps
    k = np.arange(1, steps)
    lo = np.maximum(k - window, 0)
    proxy = np.empty((paths, steps))
    proxy[:, 0] = v0
    proxy[:, 1:] = (A[:, k] - A[:, lo]) / ((k - lo) * h)
    return proxy


# ---------------------------------------------------------------------------
# Gains and costs.
# ---------------------------------------------------------------------------


def trading_outcome(phi: np.ndarray, stock: np.ndarray) -> dict[str, np.ndarray]:
    """Pathwise trading gain and traded notional for held positions ``phi``.

    stock has shape (paths, steps + 1). Returns per-path arrays:
      gain               sum_k phi_k (S_{k+1} - S_k)
      notional_initial   S_0 |phi_0|
      notional_rebalance sum_{k>=1} S_k |phi_k - phi_{k-1}|
      notional_terminal  S_N |phi_{N-1}|  (liquidation at maturity)
      trades             number of k >= 1 with phi_k != phi_{k-1}
    """
    dstock = np.diff(stock, axis=1)
    gain = np.sum(phi * dstock, axis=1)
    dphi = np.diff(phi, axis=1)
    return {
        "gain": gain,
        "notional_initial": stock[:, 0] * np.abs(phi[:, 0]),
        "notional_rebalance": np.sum(stock[:, 1:-1] * np.abs(dphi), axis=1),
        "notional_terminal": stock[:, -1] * np.abs(phi[:, -1]),
        "trades": np.count_nonzero(dphi, axis=1).astype(np.int64),
    }


def total_cost(outcome: dict[str, np.ndarray], kappa: float,
               include_initial: bool = True, include_terminal: bool = True) -> np.ndarray:
    """Proportional cost kappa x traded notional (per path)."""
    notional = outcome["notional_rebalance"].copy()
    if include_initial:
        notional += outcome["notional_initial"]
    if include_terminal:
        notional += outcome["notional_terminal"]
    return kappa * notional
