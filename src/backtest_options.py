"""Backtest with real NIFTY option prices (NSE F&O bhavcopy) and a futures hedge.

Plan option (b). Same question as ``backtest_real.py`` (fixed vs regime-aware vs
band rebalancing, with costs, train-calibrated regime labels, train-to-test
protocol and placebos), but with traded prices instead of synthetic ones:

* Option: at each close t0, sell the monthly NIFTY call whose expiry E is
  closest to 21 trading days away (at least 10), at the listed strike nearest to
  the same-expiry futures price F(t0). The premium is that call's closing
  price, and the call must have traded at t0 (contracts > 0).
* Hedge: the same-expiry NIFTY future, held at the Black-76 delta. Futures
  P&L = sum phi_k (F_{k+1} - F_k); margin financing is ignored. At expiry the
  future settles at the index close, and the call pays (F_E - K)^+ with the same
  final settlement price (checked: expiring futures settle == index close).
* Delta volatility (``--vol``):
    own  the option's own Black-76 implied vol from its close, if it traded that
         day and the price is inside the bounds; otherwise the ATM implied vol of
         the same expiry; otherwise the previous day's vol (main choice);
    vix  India VIX/100 (robustness; same as the synthetic backtest).
* Prices are divided by F(t0), so P&L is per unit of notional and comparable
  with the synthetic backtest. Costs are on futures turnover only: the option
  trade is the same for every rule, so it cannot change the ranking.

Option lives vary (about 10-32 trading days), so options are grouped by life
length for the vectorised rule engine, then put back in start-date order (the
block bootstrap needs time order).

Outputs: results/backtest_options/<vol>/per_option_{train,test}.npz and
config.json, in the format read by ``backtest_analysis.py``.

Usage: python -m src.backtest_options [--vol own|vix]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_real as bt
from . import fo_data as fd
from . import rebalancing as rb
from .upstream import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "backtest_options"
TARGET_DAYS, MIN_DAYS = 21, 10


class Market:
    """Fast lookups into the bhavcopy extract."""

    def __init__(self, fo: pd.DataFrame, calendar: pd.DatetimeIndex):
        fut, opt = fd.split_fo(fo)
        self.calendar = calendar
        # Relabelled contracts (expiry moved for a holiday): treat as one contract.
        self.aliases = self._relabels(fut, calendar)
        if self.aliases:
            fut = fut.assign(expiry=fut["expiry"].replace(self.aliases))
            opt = opt.assign(expiry=opt["expiry"].replace(self.aliases))
        self.fut_close = fut.set_index(["date", "expiry"])["close"].to_dict()
        self.fut_settle = fut.set_index(["date", "expiry"])["settle"].to_dict()
        calls = opt[(opt["instrument"] == "CE") & (opt["contracts"] > 0)]
        self.call_close = calls.set_index(["date", "expiry", "strike"])["close"].to_dict()
        self.strikes = {k: np.sort(g["strike"].to_numpy())
                        for k, g in calls.groupby(["date", "expiry"])}
        self.expiries = {d: sorted(g["expiry"].unique()) for d, g in fut.groupby("date")}
        self.actual_expiry = self._actual_expiries(fut, calendar)

    @staticmethod
    def _relabels(fut: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
        """Old label -> new label for contracts whose expiry label changed.

        Example: the June 2023 future is labelled 2023-06-29 up to 2023-06-27,
        then 2023-06-28 on its (moved) expiry day. Rule: label L is not a
        trading day, and a label L2 within 7 calendar days of L first appears on
        the trading day right after L's last appearance.
        """
        first = fut.groupby("expiry")["date"].min()
        last = fut.groupby("expiry")["date"].max()
        aliases = {}
        for label in first.index:
            if label in calendar or last[label] >= label:
                continue
            pos = calendar.searchsorted(last[label], side="right")
            if pos >= len(calendar):
                continue
            nxt = calendar[pos]
            new = [l2 for l2 in first.index if l2 != label and first[l2] == nxt
                   and abs((l2 - label).days) <= 7]
            if len(new) == 1:
                aliases[label] = new[0]
        return aliases

    def _actual_expiries(self, fut: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
        """Map each futures expiry label to the date the contract really expired.

        The bhavcopy keeps the original label when an expiry is moved for a
        holiday (e.g. label 2014-04-24, last traded and settled 2014-04-23), and
        some labels are later replaced (2014-02-27 -> 2014-02-26). Rule: the
        label itself if it is a trading day; otherwise the contract's last
        appearance, *if* it settled there at the index close (a real final
        settlement); otherwise None (stale label, contract not used).
        """
        close = None
        out = {}
        for label, g in fut.groupby("expiry"):
            if label in calendar:
                out[label] = label
                continue
            last = g["date"].max()
            settle = g.loc[g["date"] == last, "settle"].iloc[0]
            if close is None:
                close = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv",
                                    index_col=0, parse_dates=True)["close"]
            index_close = close.get(last, np.nan)
            out[label] = last if abs(settle / index_close - 1) < 1e-6 else None
        return out

    def days_to(self, day, expiry) -> int:
        return fd.trading_days_between(self.calendar, day, expiry)

    @lru_cache(maxsize=None)
    def atm_iv(self, day, expiry) -> float:
        f = self.fut_close.get((day, expiry))
        strikes = self.strikes.get((day, expiry))
        n = self.days_to(day, self.actual_expiry.get(expiry) or expiry)
        if f is None or strikes is None or n <= 0:
            return float("nan")
        k = float(strikes[np.abs(strikes - f).argmin()])
        return fd.implied_vol(self.call_close[(day, expiry, k)], f, k, n / fd.YEAR)

    def own_iv(self, day, expiry, strike) -> float:
        price = self.call_close.get((day, expiry, strike))
        f = self.fut_close.get((day, expiry))
        n = self.days_to(day, self.actual_expiry.get(expiry) or expiry)
        if price is None or f is None or n <= 0:
            return float("nan")
        return fd.implied_vol(price, f, strike, n / fd.YEAR)


def make_option(mkt: Market, day: pd.Timestamp, vix: pd.Series, vol_source: str):
    """One option position from start day ``day``; (record, None) or (None, reason)."""
    labels = [e for e in mkt.expiries.get(day, []) if mkt.actual_expiry.get(e) is not None]
    candidates = [(abs(mkt.days_to(day, mkt.actual_expiry[e]) - TARGET_DAYS), e) for e in labels
                  if mkt.days_to(day, mkt.actual_expiry[e]) >= MIN_DAYS]
    if not candidates:
        return None, "no expiry with >= 10 days"
    expiry = min(candidates)[1]          # contract label (key for prices)
    actual = mkt.actual_expiry[expiry]   # date it really expires
    f0 = mkt.fut_close.get((day, expiry))
    strikes = mkt.strikes.get((day, expiry))
    if f0 is None or strikes is None:
        return None, "no traded call or future at start"
    strike = float(strikes[np.abs(strikes - f0).argmin()])
    premium = mkt.call_close[(day, expiry, strike)]
    if not np.isfinite(mkt.own_iv(day, expiry, strike)):
        return None, "start price outside no-arbitrage bounds"
    i0 = mkt.calendar.get_loc(day)
    n = mkt.days_to(day, actual)
    days = mkt.calendar[i0:i0 + n + 1]
    if days[-1] != actual:
        return None, "expiry is not a trading day in the calendar"
    fut = [mkt.fut_close.get((d, expiry)) for d in days[:-1]] + [mkt.fut_settle.get((actual, expiry))]
    if any(v is None or not np.isfinite(v) for v in fut):
        return None, "missing futures price on the path"
    vols, fallback = [], 0
    for d in days[:-1]:
        if vol_source == "vix":
            vols.append(vix.get(d, np.nan) / 100)
            continue
        v = mkt.own_iv(d, expiry, strike)
        if not np.isfinite(v):
            fallback += 1
            v = mkt.atm_iv(d, expiry)
        if not np.isfinite(v):
            v = vols[-1] if vols else mkt.own_iv(day, expiry, strike)
        vols.append(v)
    if not np.all(np.isfinite(vols)):
        return None, "missing volatility"
    fut = np.array(fut)
    return {"start": day, "expiry": actual, "label": expiry, "n": n, "f": fut / f0, "strike": strike / f0,
            "premium": premium / f0, "vol": np.array(vols), "dates": days,
            "own_iv_fallback_share": fallback / n}, None


def evaluate_group(options: list[dict], labels: dict[str, pd.Series], rules) -> dict:
    """All options with the same life n, vectorised over options."""
    n = options[0]["n"]
    stock = np.stack([o["f"] for o in options])               # (P, n+1), normalised futures
    vol = np.stack([o["vol"] for o in options])               # (P, n)
    strike = np.array([o["strike"] for o in options])[:, None]
    tau = (n - np.arange(n))[None, :] / fd.YEAR
    target = fd.black_delta(stock[:, :-1], strike, tau, vol)
    lab = {name: np.stack([series.reindex(o["dates"][:-1]).to_numpy(dtype=float) for o in options])
           for name, series in labels.items()}
    out = {"premium": np.array([o["premium"] for o in options]),
           "payoff": np.maximum(stock[:, -1] - strike[:, 0], 0.0)}
    for name, rule, variant in rules:
        phi, _ = rb.held_positions(target, rule, lab[variant] if variant else None)
        res = rb.trading_outcome(phi, stock)
        out[f"gain|{name}"] = res["gain"]
        out[f"notional|{name}"] = (res["notional_initial"] + res["notional_rebalance"]
                                   + res["notional_terminal"])
        out[f"trades|{name}"] = res["trades"]
    return out


def run_period(mkt: Market, panel: pd.DataFrame, period: str, label_names, vol_source: str):
    in_period = panel.index[panel["split"] == period]
    first, last = in_period[0], in_period[-1]
    vix = panel["vix"]
    variants = bt.label_variants(panel, label_names)
    labels = {k: pd.Series(v, index=panel.index) for k, v in variants.items()}
    options, reasons = [], Counter()
    for day in in_period:
        if day not in mkt.expiries:
            reasons["no bhavcopy for start day"] += 1
            continue
        opt, why = make_option(mkt, day, vix, vol_source)
        if opt is None:
            reasons[why] += 1
        elif opt["expiry"] > last:
            reasons["expiry after end of period"] += 1
        else:
            options.append(opt)
    rules = bt.build_rules(list(labels), max(o["n"] for o in options))
    by_n: dict[int, list[int]] = {}
    for i, o in enumerate(options):
        by_n.setdefault(o["n"], []).append(i)
    results: dict[str, np.ndarray] = {}
    for n, idx in by_n.items():
        group = evaluate_group([options[i] for i in idx], labels, rules)
        for key, values in group.items():
            results.setdefault(key, np.empty(len(options), dtype=values.dtype))[idx] = values
    results["start_dates"] = np.array([o["start"].strftime("%Y-%m-%d") for o in options], dtype="U10")
    results["n_days"] = np.array([o["n"] for o in options])
    results["strike_over_f0"] = np.array([o["strike"] for o in options])
    results["iv0"] = np.array([o["vol"][0] for o in options])
    results["own_iv_fallback_share"] = np.array([o["own_iv_fallback_share"] for o in options])
    summary = {"options": len(options), "first_start": str(results["start_dates"][0]),
               "last_start": str(results["start_dates"][-1]), "excluded": dict(reasons),
               "life_days": {"min": int(results["n_days"].min()), "median": float(np.median(results["n_days"])),
                             "max": int(results["n_days"].max())},
               "strike_over_f0_range": [float(results["strike_over_f0"].min()),
                                        float(results["strike_over_f0"].max())],
               "mean_premium_bp": float(1e4 * results["premium"].mean()),
               "mean_payoff_bp": float(1e4 * results["payoff"].mean()),
               "mean_own_iv_fallback_share": float(results["own_iv_fallback_share"].mean()),
               "period_window": [str(first.date()), str(last.date())]}
    return results, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", choices=["own", "vix"], default="own")
    parser.add_argument("--labels", nargs="+",
                        default=["vix_hyst", "vix_q75", "rv21_q75", "ms_filter"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or OUT / args.vol
    out.mkdir(parents=True, exist_ok=True)
    panel = bt.load_panel()
    nifty = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv", index_col=0, parse_dates=True)
    mkt = Market(fd.load_fo(), nifty.index)
    config = {"maturity_days": TARGET_DAYS, "labels": args.labels, "vol_source": args.vol,
              "design": "monthly call nearest to 21 trading days, strike nearest F(t0), "
                        "held to expiry, hedged with same-expiry future", "periods": {}}
    for period in ("train", "test"):
        res, summary = run_period(mkt, panel, period, args.labels, args.vol)
        np.savez_compressed(out / f"per_option_{period}.npz", **res)
        config["periods"][period] = summary
        print(period, json.dumps(summary))
    (out / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
