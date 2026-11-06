"""Discrete rebalancing with proportional costs in the rough Hawkes--Heston model.

Extension of the terminal-loss experiment of Wang & Wang (2026, repo script
numerics/hedging_performance_experiment.py). The market, the hedges and the
random numbers are the upstream ones; what is new is that the hedge is traded
only at the times chosen by a rebalancing rule and pays proportional costs.

Design
------
* Market: upstream ``simulate_original_market`` on N fine steps (default 512),
  T = 0.25, ATM call, upstream parameters. Random numbers are drawn exactly as
  in the upstream script (same SeedSequence children, same draw order), so with
  the default seed the paths coincide with the upstream run.
* Hedges (targets): the variance-optimal GKW hedge with the unshifted kernel
  and Fourier cutoff R = 80 ("GKW"), and the Black--Scholes delta with the
  implied volatility that matches the GKW capital ("BS"), both as upstream.
* Rules: fixed frequency, variance-triggered (regime-aware) and no-trade band;
  see ``rebalancing.py``. All rules and hedges are applied to the same paths
  (common random numbers), so differences between them have small paired SEs.
* Costs: kappa x traded notional; for each (hedge, rule) we store traded
  notional and apply every kappa afterwards.

Outputs (``--out``, default results/discrete_rebalancing/):
  summary.csv    one row per (hedge, rule, kappa)
  config.json    run configuration, thresholds, timing and diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from . import hedge_paths, metrics, rebalancing as rb
from .upstream import PROJECT_ROOT, base, hedging

MATURITY = 0.25
STRIKE = 1.0
CUTOFF = 80.0
FOURIER_PANELS = np.array(
    [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0]
)
MAX_EVENTS = 8  # Poisson thresholds per path, as upstream
# Higher-accuracy GKW variant used as a robustness check (stable at 512 steps;
# the explicit Riccati scheme diverges for R = 320 on this grid).
HI_CUTOFF = 160.0
HI_PANELS = np.concatenate([FOURIER_PANELS, [100.0, 120.0, 140.0, 160.0]])
HI_ORDER = 8


def draw_sources(rng: np.random.Generator, paths: int, steps: int):
    """Random inputs in the exact order used by the upstream script."""
    z1 = rng.standard_normal((paths, steps), dtype=np.float64)
    z2 = rng.standard_normal((paths, steps), dtype=np.float64)
    thresholds = np.cumsum(rng.exponential(size=(paths, MAX_EVENTS)), axis=1)
    marks = rng.exponential(size=(paths, MAX_EVENTS))
    return z1, z2, thresholds, marks


def setup_hedge(model: base.Model, steps: int, cutoff: float = CUTOFF,
                panels: np.ndarray = FOURIER_PANELS, order: int = 4):
    """Fourier nodes, Riccati solution, GKW capital and BS implied vol.

    Defaults are the upstream choices (R = 80, 4 Gauss points per panel).
    """
    lambdas, weights = base.fourier_rule(order, panels)
    u = 0.5 + 1j * lambdas
    riccati = base.solve_riccati_product_integration(model, u, MATURITY, steps, 0.0)
    capital = float(
        base.capital_by_cutoff(
            base.initial_memory(model, riccati[1], MATURITY, steps),
            base.call_coefficients(lambdas, weights),
            lambdas,
            np.array([cutoff]),
        )[0]
    )
    implied_vol = brentq(
        lambda vol: hedging.black_scholes_call(1.0, STRIKE, MATURITY, vol) - capital,
        1.0e-5,
        2.0,
    )
    return lambdas, weights, riccati, capital, implied_vol


def variance_thresholds(model: base.Model, steps: int, seed: int,
                        quantiles: tuple[float, ...], window: int,
                        pilot_paths: int = 2000):
    """Calibrate stress thresholds on an independent pilot sample.

    Using separate paths keeps the thresholds deterministic constants for the
    main run (no information from the evaluation paths leaks into the rule).
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    market = base.simulate_original_market(
        model, MATURITY, steps, *draw_sources(rng, pilot_paths, steps)
    )
    proxy = rb.spot_variance_proxy(market, MATURITY, model.v0, window)
    return {q: float(np.quantile(proxy[:, 1:], q)) for q in quantiles}


def build_rules(steps: int, thresholds: dict[float, float]) -> list[rb.Rule]:
    # Dense fixed-frequency grid (powers of 2 and 1.5 x powers of 2). Rules are
    # compared with fixed-frequency hedging at equal turnover by interpolating
    # between neighbouring fixed rules; a sparse grid would make that benchmark
    # artificially weak because the error-turnover curve is convex.
    rules: list[rb.Rule] = []
    candidates = sorted({2**j for j in range(12)} | {3 * 2**j for j in range(11)})
    for every in candidates:
        if every <= steps:
            rules.append(rb.FixedFrequency(every))
    for q, thr in thresholds.items():
        for calm in (16, 32, 64, 128):
            if calm <= steps:
                rules.append(rb.VarianceTriggered(calm, max(calm // 4, 1), thr))
    for width in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        rules.append(rb.NoTradeBand(width))
    return rules


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--kappas", type=float, nargs="+",
                        default=[0.0, 1.0e-4, 5.0e-4, 1.0e-3])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--proxy-window", type=float, default=1.0 / 16.0,
                        help="trailing realised-variance window as a fraction of T")
    parser.add_argument("--no-per-path", action="store_true",
                        help="skip saving per-path arrays (needed for paired SEs)")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "results" / "discrete_rebalancing")
    args = parser.parse_args(argv)
    if args.steps & (args.steps - 1):
        raise ValueError("--steps must be a power of two")

    started = time.time()
    model = base.Model()
    lambdas, weights, riccati, capital, implied_vol = setup_hedge(model, args.steps)
    # Robustness variant: higher Fourier cutoff and quadrature order. The R = 80
    # holdings carry truncation error near maturity; this shows whether our
    # conclusions depend on it. Same capital convention: its own Fourier capital.
    hi = setup_hedge(model, args.steps, cutoff=HI_CUTOFF, panels=HI_PANELS, order=HI_ORDER)
    window = max(1, round(args.proxy_window * args.steps))
    thresholds = variance_thresholds(model, args.steps, args.seed,
                                     tuple(args.quantiles), window)
    rules = build_rules(args.steps, thresholds)
    hedges = ("GKW", "GKW_R160", "BS")
    capitals = {"GKW": capital, "GKW_R160": hi[3], "BS": capital}

    keys = ("gain", "notional_initial", "notional_rebalance", "notional_terminal", "trades")
    store = {(h, r.label): {k: [] for k in keys} for h in hedges for r in rules}
    payoff_parts, jumps = [], 0
    n_batches = math.ceil(args.paths / args.batch_size)
    children = np.random.SeedSequence(args.seed).spawn(n_batches)

    for b, start in enumerate(range(0, args.paths, args.batch_size)):
        count = min(args.batch_size, args.paths - start)
        rng = np.random.default_rng(children[b])
        market = base.simulate_original_market(
            model, MATURITY, args.steps, *draw_sources(rng, count, args.steps)
        )
        jumps += int(market["jump_count"])
        stock = np.asarray(market["stock"])
        payoff_parts.append(np.maximum(stock[:, -1] - STRIKE, 0.0))
        proxy = rb.spot_variance_proxy(market, MATURITY, model.v0, window)
        targets = {
            "GKW": hedge_paths.kernel_hedge_holdings(
                model, MATURITY, market, lambdas, weights, CUTOFF, 0.0, riccati
            )[1],
            "GKW_R160": hedge_paths.kernel_hedge_holdings(
                model, MATURITY, market, hi[0], hi[1], HI_CUTOFF, 0.0, hi[2]
            )[1],
            "BS": hedge_paths.black_scholes_delta_holdings(stock, MATURITY, implied_vol, STRIKE),
        }
        for h in hedges:
            for rule in rules:
                phi, _ = rb.held_positions(targets[h], rule, proxy)
                outcome = rb.trading_outcome(phi, stock)
                for k in keys:
                    store[(h, rule.label)][k].append(outcome[k])
        print(f"batch {b + 1}/{n_batches} done, {time.time() - started:.0f}s", flush=True)

    payoff = np.concatenate(payoff_parts)
    baseline_sq = (payoff - payoff.mean()) ** 2
    rows = []
    zero = np.zeros_like(payoff)
    for kappa in args.kappas:
        rows.append({"hedge": "none", "rule": "none", "kappa": kappa, "mean_trades": 0.0,
                     **metrics.pnl_summary(payoff, zero, zero, capital, baseline_sq)})
        for h in hedges:
            for rule in rules:
                out = {k: np.concatenate(v) for k, v in store[(h, rule.label)].items()}
                cost = rb.total_cost(out, kappa)
                rows.append({
                    "hedge": h, "rule": rule.label, "kappa": kappa,
                    "mean_trades": float(out["trades"].mean()),
                    **metrics.pnl_summary(payoff, out["gain"], cost, capitals[h], baseline_sq),
                    "mean_notional_rebalance": float(out["notional_rebalance"].mean()),
                })

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_per_path:
        # Per-path outcomes for paired comparisons (analysis.py). Notional and
        # trades are stored in reduced precision; gains stay float64.
        arrays = {"payoff": payoff}
        for (h, label), parts in store.items():
            tag = f"{h}|{label}"
            arrays[f"gain|{tag}"] = np.concatenate(parts["gain"])
            arrays[f"notional|{tag}"] = (
                np.concatenate(parts["notional_initial"])
                + np.concatenate(parts["notional_rebalance"])
                + np.concatenate(parts["notional_terminal"])
            ).astype(np.float32)
            arrays[f"trades|{tag}"] = np.concatenate(parts["trades"]).astype(np.int16)
        np.savez_compressed(args.out / "per_path.npz", **arrays)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "paths": args.paths, "batch_size": args.batch_size, "steps": args.steps,
        "seed": args.seed, "maturity": MATURITY, "strike": STRIKE, "cutoff": CUTOFF,
        "kappas": args.kappas, "gkw_capital": capital, "gkw_r160_capital": hi[3],
        "bs_implied_vol": implied_vol,
        "variance_thresholds": {str(q): v for q, v in thresholds.items()},
        "proxy_window_steps": window,
        "payoff_mean": float(payoff.mean()), "total_jumps": jumps,
        "rules": [r.label for r in rules], "runtime_seconds": time.time() - started,
        "cost_convention": "kappa * notional; initial purchase and terminal liquidation included",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
