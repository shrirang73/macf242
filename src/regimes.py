"""Calm/stress regime labels for NIFTY 50 with a chronological split and no look-ahead.

Timing convention
-----------------
Every label dated t uses only data up to and including the close of day t, so it
may drive a decision taken at that close (e.g. how often to rebalance over
(t, t+1]). It must never be compared with returns up to t as if it were a
forecast: outcomes are always measured strictly after t.

Calibration
-----------
All thresholds and model parameters are estimated on the TRAIN period only
(default: sample start .. 2018-12-31). On the TEST period (2019-01-01 ..) they are
applied unchanged.

Methods (stress = 1, calm = 0)
------------------------------
vix_q75        India VIX close > 75th percentile of train VIX.
vix_hyst       Same, with hysteresis: enter stress above the train 75th
               percentile, leave only below the train 60th percentile.
rv21_q75       Trailing 21-day realised volatility of NIFTY log returns
               (annualised, returns up to t) > its train 75th percentile.
               Real-data analogue of the trailing realised-variance rule in the
               simulation (src/rebalancing.py).
ms_filter      Two-state Gaussian Markov-switching model for daily log returns
               (Hamilton 1989), fitted by maximum likelihood on train. Stress =
               filtered probability P(high-vol state | returns up to t) > 0.5.
               Filtered, never smoothed: smoothed probabilities use future data.

Usage: python -m src.regimes [--split 2018-12-31] [--start 2010-07-19]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .upstream import PROJECT_ROOT

PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "results" / "regimes"
TRADING_DAYS = 252
RV_WINDOW = 21


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_panel(start: str) -> pd.DataFrame:
    """NIFTY close, log return and India VIX on NIFTY trading days from ``start``.

    VIX is missing on one NIFTY special session (2024-03-02); it is
    forward-filled from the previous close, which is known at t (causal).
    """
    nifty = pd.read_csv(PROCESSED / "nifty50.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(PROCESSED / "indiavix.csv", index_col=0, parse_dates=True)
    panel = pd.DataFrame({"nifty": nifty["close"], "special_session": nifty["special_session"]})
    panel["logret"] = np.log(panel["nifty"]).diff()
    panel["vix"] = vix["close"].reindex(panel.index)
    panel["vix_filled"] = panel["vix"].isna()
    panel["vix"] = panel["vix"].ffill()
    panel["vix_verified"] = vix["verified"].reindex(panel.index).fillna(False).astype(bool)
    panel = panel[panel.index >= pd.Timestamp(start)]
    return panel.iloc[1:]  # first row has no return inside the sample


def trailing_rv(logret: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Annualised realised vol from the last ``window`` returns up to t."""
    return np.sqrt(TRADING_DAYS * (logret**2).rolling(window, min_periods=window).mean())


# ---------------------------------------------------------------------------
# Threshold rules
# ---------------------------------------------------------------------------


def threshold_label(x: pd.Series, threshold: float) -> pd.Series:
    label = (x > threshold).astype(float)
    return label.where(x.notna())


def hysteresis_label(x: pd.Series, enter: float, leave: float) -> pd.Series:
    """Stress from the first close above ``enter`` until the first close below
    ``leave`` (sequential, so causal). Starts calm."""
    if leave > enter:
        raise ValueError("leave threshold must not exceed enter threshold")
    out, state = [], 0.0
    for value in x.to_numpy():
        if np.isnan(value):
            out.append(np.nan)
            continue
        if state == 0.0 and value > enter:
            state = 1.0
        elif state == 1.0 and value < leave:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=x.index)


# ---------------------------------------------------------------------------
# Two-state Markov-switching model (Hamilton filter)
# ---------------------------------------------------------------------------


def _unpack(theta):
    mu = np.array([theta[0], theta[1]])
    sigma = np.exp([theta[2], theta[3]])
    p00, p11 = expit(theta[4]), expit(theta[5])
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])  # P[i, j] = P(s_t = j | s_{t-1} = i)
    return mu, sigma, P


def hamilton_filter(r: np.ndarray, mu, sigma, P):
    """Filtered state probabilities P(s_t | r_1..r_t) and the log-likelihood.

    Starts from the stationary distribution of P. Only past and current
    observations enter each step, so the output at t is causal by construction.
    """
    # Scalar arithmetic: two states, so this is much faster than small arrays.
    p00, p01, p10, p11 = float(P[0, 0]), float(P[0, 1]), float(P[1, 0]), float(P[1, 1])
    m0, m1, s0, s1 = float(mu[0]), float(mu[1]), float(sigma[0]), float(sigma[1])
    c0, c1 = 1.0 / (s0 * math.sqrt(2 * math.pi)), 1.0 / (s1 * math.sqrt(2 * math.pi))
    xi1 = p01 / (p01 + p10)  # stationary probability of state 1
    xi0 = 1.0 - xi1
    filtered = np.empty((len(r), 2))
    loglik = 0.0
    for t, x in enumerate(r.tolist()):
        prior0 = p00 * xi0 + p10 * xi1          # P(s_t = 0 | r_1..r_{t-1})
        prior1 = p01 * xi0 + p11 * xi1
        z0, z1 = (x - m0) / s0, (x - m1) / s1
        joint0 = prior0 * c0 * math.exp(-0.5 * z0 * z0)
        joint1 = prior1 * c1 * math.exp(-0.5 * z1 * z1)
        total = joint0 + joint1                 # density of r_t given the past
        loglik += math.log(total)
        xi0, xi1 = joint0 / total, joint1 / total
        filtered[t, 0], filtered[t, 1] = xi0, xi1
    return filtered, loglik


def fit_markov_switching(r: np.ndarray, starts: int = 8, seed: int = 0) -> dict:
    """MLE on the training returns (in percent) from several starting points.

    Returns the best fit, with state 1 = the higher-volatility (stress) state.
    """
    rng = np.random.default_rng(seed)
    sd = float(np.std(r))
    best, all_logliks = None, []
    for i in range(starts):
        theta0 = np.array([
            0.05 + 0.02 * rng.standard_normal(), -0.05 + 0.02 * rng.standard_normal(),
            math.log(sd * rng.uniform(0.5, 0.8)), math.log(sd * rng.uniform(1.3, 2.0)),
            logit(rng.uniform(0.9, 0.99)), logit(rng.uniform(0.8, 0.97)),
        ])
        res = minimize(lambda th: -hamilton_filter(r, *_unpack(th))[1], theta0,
                       method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-7})
        all_logliks.append(-float(res.fun))
        if best is None or res.fun < best.fun:
            best = res
    mu, sigma, P = _unpack(best.x)
    if sigma[0] > sigma[1]:  # order states so that 1 = stress
        mu, sigma, P = mu[::-1], sigma[::-1], P[::-1, ::-1]
    fits = {"mu_pct": mu.tolist(), "sigma_pct": sigma.tolist(),
            "P": P.tolist(), "loglik_train": -float(best.fun),
            "converged": bool(best.success), "n_train": int(len(r)),
            # Multi-start check: if several starts reach the same best value,
            # the optimum is unlikely to be a poor local one.
            "loglik_all_starts": sorted(all_logliks, reverse=True),
            "expected_duration_days": [1 / (1 - P[0, 0]), 1 / (1 - P[1, 1])]}
    return fits


# ---------------------------------------------------------------------------
# Build labels
# ---------------------------------------------------------------------------


def calibrate(train: pd.DataFrame) -> dict:
    """Everything estimated from the training rows only."""
    rv = trailing_rv(train["logret"])
    r_pct = 100 * train["logret"].to_numpy()
    return {
        "vix_q75": float(train["vix"].quantile(0.75)),
        "vix_q60": float(train["vix"].quantile(0.60)),
        "rv21_q75": float(rv.quantile(0.75)),
        "markov_switching": fit_markov_switching(r_pct),
        "train_first": str(train.index[0].date()), "train_last": str(train.index[-1].date()),
        "train_rows": int(len(train)),
    }


def label(panel: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the (train-calibrated) rules to the whole panel, causally."""
    out = panel.copy()
    out["rv21"] = trailing_rv(out["logret"])
    out["vix_q75"] = threshold_label(out["vix"], params["vix_q75"])
    out["vix_hyst"] = hysteresis_label(out["vix"], params["vix_q75"], params["vix_q60"])
    out["rv21_q75"] = threshold_label(out["rv21"], params["rv21_q75"])
    ms = params["markov_switching"]
    filtered, _ = hamilton_filter(100 * out["logret"].to_numpy(), np.array(ms["mu_pct"]),
                                  np.array(ms["sigma_pct"]), np.array(ms["P"]))
    out["ms_prob_stress"] = filtered[:, 1]
    out["ms_filter"] = (out["ms_prob_stress"] > 0.5).astype(float)
    return out


METHODS = ("vix_q75", "vix_hyst", "rv21_q75", "ms_filter")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def spells(lab: pd.Series) -> pd.DataFrame:
    """Stress spells (start, end, length in trading days)."""
    s = lab.fillna(0).astype(int)
    change = s.diff().fillna(s.iloc[0]).ne(0).cumsum()
    groups = s.groupby(change)
    rows = [{"start": g.index[0], "end": g.index[-1], "days": len(g)}
            for _, g in groups if g.iloc[0] == 1]
    return pd.DataFrame(rows)


def forward_rv(logret: pd.Series, horizon: int) -> pd.Series:
    """Annualised realised vol over (t, t+horizon]: strictly AFTER t."""
    future_sq = (logret**2).shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    return np.sqrt(TRADING_DAYS * future_sq)


def block_bootstrap_ratio(x: np.ndarray, lab: np.ndarray, block: int = 20,
                          reps: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Ratio mean(x | stress) / mean(x | calm) with a moving-block bootstrap CI.

    Blocks keep the serial dependence of daily data (both x and the labels are
    persistent), so the CI is not overly optimistic.
    """
    ok = ~np.isnan(x) & ~np.isnan(lab)
    x, lab = x[ok], lab[ok]
    ratio = x[lab == 1].mean() / x[lab == 0].mean()
    rng = np.random.default_rng(seed)
    n = len(x)
    starts_max = n - block
    stats = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.integers(0, starts_max, size=n // block + 1)])[:n]
        xb, lb = x[idx], lab[idx]
        if (lb == 1).sum() and (lb == 0).sum():
            stats.append(xb[lb == 1].mean() / xb[lb == 0].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(ratio), float(lo), float(hi)


def diagnostics(lab: pd.DataFrame, split: pd.Timestamp) -> dict:
    out = {}
    fwd5 = forward_rv(lab["logret"], 5)
    fwd21 = forward_rv(lab["logret"], 21)
    for name, part in (("train", lab.index <= split), ("test", lab.index > split)):
        block = lab[part]
        d = {"rows": int(len(block)), "first": str(block.index[0].date()),
             "last": str(block.index[-1].date())}
        for m in METHODS:
            series = block[m]
            sp = spells(series)
            entry = {
                "share_stress": float(series.mean()),
                "switches": int(series.fillna(0).diff().abs().sum()),
                "stress_spells": int(len(sp)),
                "median_spell_days": float(sp["days"].median()) if len(sp) else 0.0,
            }
            for h, fwd in ((5, fwd5), (21, fwd21)):
                r, lo, hi = block_bootstrap_ratio(fwd[part].to_numpy(), series.to_numpy())
                entry[f"fwd_rv{h}_stress_over_calm"] = [r, lo, hi]
            d[m] = entry
        # Pairwise agreement between methods (share of days with equal labels).
        d["agreement"] = {f"{a}~{b}": float((block[a] == block[b]).mean())
                          for i, a in enumerate(METHODS) for b in METHODS[i + 1:]}
        out[name] = d
    return out


def plot(lab: pd.DataFrame, split: pd.Timestamp, path: Path):
    """NIFTY (log scale) and India VIX in two stacked panels, stress shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    shade = spells(lab["vix_hyst"])
    for ax in axes:
        for _, s in shade.iterrows():
            ax.axvspan(s["start"], s["end"], color="#eb6834", alpha=0.18, linewidth=0)
        ax.axvline(split, color="#52514e", lw=1, ls="--")
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#898781", labelcolor="#52514e")
    axes[0].semilogy(lab.index, lab["nifty"], color="#2a78d6", lw=1.0)
    axes[0].set_ylabel("NIFTY 50 (log scale)", color="#52514e")
    axes[1].plot(lab.index, lab["vix"], color="#2a78d6", lw=0.8)
    axes[1].set_ylabel("India VIX", color="#52514e")
    axes[0].text(split, axes[0].get_ylim()[1], "  train | test", va="top", color="#52514e", fontsize=8)
    axes[0].set_title("Shaded: stress regime (VIX with hysteresis, train-calibrated)",
                      fontsize=9, color="#0b0b0b", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-07-19",
                        help="sample start (default: first NSE-verified VIX day)")
    parser.add_argument("--split", default="2018-12-31", help="last day of the train period")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    split = pd.Timestamp(args.split)

    panel = load_panel(args.start)
    params = calibrate(panel[panel.index <= split])
    lab = label(panel, params)
    lab["split"] = np.where(lab.index <= split, "train", "test")
    diag = diagnostics(lab, split)

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["split", "nifty", "logret", "vix", "vix_verified", "vix_filled", "special_session",
            "rv21", "ms_prob_stress", *METHODS]
    lab[cols].to_csv(args.out / "regime_labels.csv")
    test_spells = spells(lab.loc[lab.index > split, "vix_hyst"])
    summary = {"start": args.start, "split": args.split, "params": params, "diagnostics": diag,
               "test_stress_spells_vix_hyst": [
                   {"start": str(r.start.date()), "end": str(r.end.date()), "days": int(r.days)}
                   for r in test_spells.itertuples()]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(lab, split, args.out / "fig_regimes")
    print(json.dumps(summary, indent=1)[:7000])


if __name__ == "__main__":
    main()
