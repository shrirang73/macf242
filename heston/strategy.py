"""Stage 5: a trading strategy built from the project -- selling hedged insurance.

Idea in one line: index options are insurance, and buyers pay a premium above
the expected payout (the volatility risk premium). A seller who delta-hedges
with futures removes the direction of the market and keeps that premium, but
takes rare large losses in crashes, like an insurer.

Strategy (uses only information at the close of each day):
  * Every trading day, sell one ATM monthly NIFTY call (the options of
    hedge.py), sized at 1/21 of the book, so about one month of options is open.
  * Hedge daily with the same-expiry future using one of
        none   : no hedge (a naked bet that the market will not rise much),
        bs     : Black-Scholes delta at the option's own implied vol,
        mv     : Heston minimum-variance delta (eq. (2) of the report).
  * Optional risk rule 'skip stress': do not open new options on days the causal
    India-VIX label says 'stress' (label from data/inputs/regime_labels.csv).
  * Costs: futures cost per unit traded and half the option bid-ask spread as a
    fraction of the premium, paid when the option is sold. Two levels:
        low  : futures 1 bp, option spread 1% of premium
        high : futures 5 bp, option spread 3% of premium
    The rule and cost levels are fixed in advance, not fitted.

Daily mark-to-market: every open option is valued each day at Black-76 with the
implied vol used for hedging that day (its own market vol when it traded), and
the futures hedge at the futures close (results/hedge_paths.csv.gz). The book's
daily P&L is the sum over open options, divided by 21 (fraction of notional).
Summed over its life, an option's daily P&L equals its final hedging result.

Outputs: results/strategy.csv (metrics), results/strategy_daily.csv (daily
components: gross P&L, futures turnover, option-spread base, by hedge and by the
entry-day regime), results/strategy_curves.csv (daily net P&L per variant).

Usage:  python -m heston.strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INPUTS, ROOT
from .stats import EVENTS, TEST_START

COSTS = {"low": (1e-4, 0.01), "high": (5e-4, 0.03)}      # (futures per unit, option half-spread)
HEDGES = {"none": None, "bs": "bs_own", "mv": "heston_mv"}
BOOK = 21
YEAR_DAYS = 252


def load():
    h = pd.read_csv(ROOT / "results" / "hedge.csv.gz", parse_dates=["date", "expiry"])
    p = pd.read_csv(ROOT / "results" / "hedge_paths.csv.gz", parse_dates=["date"])
    reg = pd.read_csv(INPUTS / "regime_labels.csv", parse_dates=["date"])[["date", "vix", "vix_hyst"]]
    h = h.merge(reg, on="date", how="left")
    return h, p.sort_values(["opt", "k"]).reset_index(drop=True)


def daily_components(h: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Book-level daily components, columns (hedge, group, comp), as fractions of notional.

    comp = gross (P&L before costs), fut (futures turnover |d phi| * F_k / F0),
    opt (premium / F0, booked on the sale day; cost = spread * opt).
    group = 'calm' or 'stress': regime label on the option's sale day."""
    F0 = h["F0"].to_numpy()[p["opt"].to_numpy()]
    stress = (h["vix_hyst"].fillna(0).to_numpy() == 1)[p["opt"].to_numpy()]
    g = p.groupby("opt")
    dV = g["V"].diff().to_numpy()                       # V_k - V_{k-1}, NaN at k = 0
    dF = g["F"].diff().to_numpy()
    first = p["k"].to_numpy() == 0
    parts = {}
    for hedge, col in HEDGES.items():
        if col is None:
            phi = np.zeros(len(p))
        else:
            phi = p[f"phi_{col}"].to_numpy()
        phi_prev = g[f"phi_{col}"].shift(1).to_numpy() if col else np.zeros(len(p))
        phi_prev = np.where(first, 0.0, phi_prev)
        gross = np.where(first, 0.0, -dV + phi_prev * dF) / F0      # P&L earned on day k
        # trade at the close of day k from phi_{k-1} to phi_k (phi_n is closed by settlement)
        phi_now = np.where(np.isnan(phi), phi_prev, phi)            # last day: no new trade
        fut = np.abs(phi_now - phi_prev) * p["F"].to_numpy() / F0 if col else np.zeros(len(p))
        opt = np.where(first, h["c0"].to_numpy()[p["opt"].to_numpy()], 0.0) / F0
        frame = pd.DataFrame({"date": p["date"], "stress": stress, "gross": gross, "fut": fut, "opt": opt})
        for grp, sel in (("calm", ~frame["stress"]), ("stress", frame["stress"])):
            s = frame[sel].groupby("date")[["gross", "fut", "opt"]].sum() / BOOK
            for comp in ("gross", "fut", "opt"):
                parts[(hedge, grp, comp)] = s[comp]
    out = pd.DataFrame(parts).sort_index().fillna(0.0)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["hedge", "group", "comp"])
    return out.sort_index(axis=1)


def net_series(comp: pd.DataFrame, hedge: str, cost: str, rule: str) -> pd.Series:
    fut, spread = COSTS[cost]
    groups = ["calm", "stress"] if rule == "always" else ["calm"]
    r = 0.0
    for grp in groups:
        c = comp[(hedge, grp)]
        r = r + c["gross"] - fut * c["fut"] - spread * c["opt"]
    return r


def metrics(r: pd.Series) -> dict:
    eq = r.cumsum()
    m = r.groupby(r.index.to_period("M")).sum()
    ann, vol = YEAR_DAYS * r.mean(), np.sqrt(YEAR_DAYS) * r.std(ddof=1)
    return dict(ann_ret_pct=100 * ann, ann_vol_pct=100 * vol, sharpe=ann / vol if vol > 0 else np.nan,
                max_dd_pct=100 * (eq - eq.cummax()).min(), worst_day_pct=100 * r.min(),
                worst_month_pct=100 * m.min(), pos_months_pct=100 * (m > 0).mean(), days=len(r))


def run(verbose: bool = True):
    h, p = load()
    comp = daily_components(h, p)
    comp.to_csv(ROOT / "results" / "strategy_daily.csv")
    rows, curves = [], {}
    cs, ce = EVENTS["COVID-19 crash"]
    for cost in COSTS:
        for hedge in HEDGES:
            for rule in ("always", "skip stress"):
                r = net_series(comp, hedge, cost, rule)
                curves[(cost, hedge, rule)] = r
                for period, sel in (("train", r.index < TEST_START), ("test", r.index >= TEST_START),
                                    ("all", np.ones(len(r), bool))):
                    rows.append(dict(cost=cost, hedge=hedge, rule=rule, period=period, **metrics(r[sel]),
                                     covid_pnl_pct=100 * r[(r.index >= cs) & (r.index <= ce)].sum()))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "strategy.csv", index=False)
    cur = pd.DataFrame({f"{c}|{hd}|{ru}": s for (c, hd, ru), s in curves.items()})
    cur.to_csv(ROOT / "results" / "strategy_curves.csv")
    if verbose:
        with pd.option_context("display.width", 220):
            print(out[out.period == "test"].drop(columns="period").round(2).to_string(index=False))
    return out, curves, h, comp


if __name__ == "__main__":
    run()
