"""Real-data backtest: hedging synthetic ATM NIFTY calls with discrete rebalancing.

Question
--------
On historical NIFTY 50 paths, does regime-aware rebalancing (trade more often in
stress, less in calm) hedge an option better than fixed-frequency rebalancing at
the same transaction cost? This is the real-data counterpart of
``experiment_discrete.py``. Plan option (a): a crude proxy that uses no options
data.

Design
------
* Options: at every close t0 a 1-month (21 trading days) ATM European call on
  NIFTY is sold. Its whole life must lie inside one period (train or test), so
  the two periods never share information.
* Premium and hedge: Black--Scholes with implied vol = India VIX/100. The premium
  uses VIX at t0; the hedge holds the BS delta computed with VIX at each
  rebalancing close. Zero interest rate and dividends: a simplification, since
  in practice the hedge would be NIFTY futures (see notes/backtest_log.md).
  Prices are divided by S(t0), so all P&L is per unit of notional.
* Rebalancing at daily closes, with the rules from ``rebalancing.py``:
    fixed       every m trading days
    regime      every m_stress days if the regime label at the close is stress,
                else every m_calm days (VarianceTriggered with the 0/1 label as
                proxy and threshold 0.5)
    band        trade only if |BS delta - position| > h
  Labels come from ``regimes.py`` (train-calibrated; dated t uses data up to t).
* Placebos, to check that any gain comes from the regime information:
    flip:L      labels inverted (trade more often in calm)
    lag252:L    label from 252 trading days earlier (causal but stale)
* Costs: kappa x traded notional, including the initial trade and the final
  liquidation (same convention as the simulation).

Outputs (results/backtest_real/<tag>/): per_option.npz, config.json.
Analysis: ``backtest_analysis.py``.

Usage: python -m src.backtest_real [--maturity 21] [--labels vix_hyst rv21_q75 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import rebalancing as rb
from .upstream import PROJECT_ROOT

LABELS_CSV = PROJECT_ROOT / "results" / "regimes" / "regime_labels.csv"
OUT = PROJECT_ROOT / "results" / "backtest_real"
YEAR = 252  # trading days per year, for time to maturity
PLACEBO_LAG = 252

FIXED_EVERY = (1, 2, 3, 4, 5, 7, 10, 14, 21)
REGIME_PAIRS = ((2, 1), (3, 1), (5, 1), (7, 1), (10, 1), (21, 1),
                (5, 2), (10, 2), (21, 2), (10, 5), (21, 5), (21, 7))
BANDS = (0.02, 0.05, 0.1, 0.2)


# ---------------------------------------------------------------------------
# Black--Scholes (zero rate), vectorised
# ---------------------------------------------------------------------------


def bs_d1(s, k, tau, vol):
    return (np.log(s / k) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))


def bs_call(s, k, tau, vol):
    d1 = bs_d1(s, k, tau, vol)
    return s * ndtr(d1) - k * ndtr(d1 - vol * np.sqrt(tau))


def bs_delta(s, k, tau, vol):
    return ndtr(bs_d1(s, k, tau, vol))


# ---------------------------------------------------------------------------
# Option paths from the historical panel
# ---------------------------------------------------------------------------


def load_panel(path: Path = LABELS_CSV) -> pd.DataFrame:
    panel = pd.read_csv(path, index_col=0, parse_dates=True)
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("panel dates must be unique and increasing")
    return panel


def label_variants(panel: pd.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    """Label series (aligned with the panel rows) for each method and placebo."""
    out = {}
    for name in names:
        lab = panel[name].to_numpy(dtype=float)
        out[name] = lab
        out[f"flip:{name}"] = np.where(np.isnan(lab), np.nan, 1.0 - lab)
        lagged = np.full_like(lab, np.nan)
        lagged[PLACEBO_LAG:] = lab[:-PLACEBO_LAG]  # value from 252 days earlier (causal)
        out[f"lag{PLACEBO_LAG}:{name}"] = lagged
    return out


def option_starts(panel: pd.DataFrame, period: str, maturity: int) -> np.ndarray:
    """Row positions t0 whose option life [t0, t0 + maturity] lies inside ``period``."""
    in_period = (panel["split"] == period).to_numpy()
    n = len(panel)
    starts = [i for i in range(n - maturity)
              if in_period[i] and in_period[i + maturity]
              and in_period[i:i + maturity + 1].all()]
    return np.array(starts, dtype=int)


def build_paths(panel: pd.DataFrame, starts: np.ndarray, maturity: int,
                labels: dict[str, np.ndarray]):
    """Normalised price paths S/S(t0), vol paths and label paths per option."""
    idx = starts[:, None] + np.arange(maturity + 1)[None, :]
    price = panel["nifty"].to_numpy()[idx]
    stock = price / price[:, :1]
    vol = panel["vix"].to_numpy()[idx] / 100.0
    lab = {name: series[idx[:, :-1]] for name, series in labels.items()}  # decision dates
    return stock, vol, lab


def target_deltas(stock: np.ndarray, vol: np.ndarray, strike: float = 1.0) -> np.ndarray:
    """BS delta at each decision close k = 0..N-1, with that day's implied vol."""
    n_steps = stock.shape[1] - 1
    tau = (n_steps - np.arange(n_steps)) / YEAR
    return bs_delta(stock[:, :-1], strike, tau[None, :], vol[:, :-1])


# ---------------------------------------------------------------------------
# Rules and evaluation
# ---------------------------------------------------------------------------


def build_rules(label_names: list[str], maturity: int) -> list[tuple[str, rb.Rule, str | None]]:
    """(name, rule, label variant or None)."""
    rules = [(f"fixed/every={m}", rb.FixedFrequency(m), None)
             for m in FIXED_EVERY if m <= maturity]
    for name in label_names:
        for calm, stress in REGIME_PAIRS:
            if calm <= maturity:
                rule = rb.VarianceTriggered(calm, stress, 0.5)
                rules.append((f"regime[{name}]/calm={calm},stress={stress}", rule, name))
    rules += [(f"band/h={h:g}", rb.NoTradeBand(h), None) for h in BANDS]
    return rules


def evaluate_period(panel: pd.DataFrame, period: str, maturity: int,
                    label_names: list[str]) -> dict:
    labels = label_variants(panel, label_names)
    variants = list(labels)  # includes placebos
    starts = option_starts(panel, period, maturity)
    stock, vol, lab = build_paths(panel, starts, maturity, labels)
    tau0 = maturity / YEAR
    premium = bs_call(1.0, 1.0, tau0, vol[:, 0])
    payoff = np.maximum(stock[:, -1] - 1.0, 0.0)
    target = target_deltas(stock, vol)
    out = {"start_dates": panel.index[starts].strftime("%Y-%m-%d").to_numpy().astype("U10"),
           "premium": premium, "payoff": payoff}
    for name, rule, variant in build_rules(variants, maturity):
        proxy = lab[variant] if variant is not None else None
        phi, _ = rb.held_positions(target, rule, proxy)
        res = rb.trading_outcome(phi, stock)
        out[f"gain|{name}"] = res["gain"]
        out[f"notional|{name}"] = (res["notional_initial"] + res["notional_rebalance"]
                                   + res["notional_terminal"])
        out[f"trades|{name}"] = res["trades"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--maturity", type=int, default=21, help="trading days")
    parser.add_argument("--labels", nargs="+",
                        default=["vix_hyst", "vix_q75", "rv21_q75", "ms_filter"])
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or OUT / f"maturity_{args.maturity}"
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.labels_csv)
    config = {"maturity_days": args.maturity, "labels": args.labels,
              "labels_csv": str(args.labels_csv), "year_days": YEAR,
              "placebo_lag": PLACEBO_LAG, "periods": {}}
    for period in ("train", "test"):
        res = evaluate_period(panel, period, args.maturity, args.labels)
        np.savez_compressed(out / f"per_option_{period}.npz", **res)
        config["periods"][period] = {
            "options": int(len(res["premium"])),
            "first_start": str(res["start_dates"][0]), "last_start": str(res["start_dates"][-1]),
            "mean_premium": float(res["premium"].mean()),
            "mean_payoff": float(res["payoff"].mean()),
        }
        print(period, config["periods"][period])
    (out / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
