"""NIFTY monthly futures and options from the bhavcopy extract (src/bhavcopy.py).

Conventions
-----------
* Monthly contracts: NIFTY futures are listed only for monthly expiries, so an
  option is "monthly" if its expiry is also a futures expiry. This stays correct
  when weekly options are listed and when the expiry weekday changes.
* A price counts as a trade only if ``contracts > 0``. Untraded contracts carry a
  stale close in the bhavcopy.
* Black-76 with zero discounting: the futures price F is the forward, and option
  prices are not discounted (about 0.5% of the price for one month at 6% rates;
  see notes/options_log.md).

Usage (checks): python -m src.fo_data
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from .upstream import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "raw" / "bhavcopy"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CHECKS = PROJECT_ROOT / "results" / "data_checks"
YEAR = 252


# ---------------------------------------------------------------------------
# Black-76 (zero discounting)
# ---------------------------------------------------------------------------


def black_call(f, k, tau, vol):
    sd = vol * np.sqrt(tau)
    d1 = (np.log(f / k) + 0.5 * sd**2) / sd
    return f * ndtr(d1) - k * ndtr(d1 - sd)


def black_delta(f, k, tau, vol):
    sd = vol * np.sqrt(tau)
    return ndtr((np.log(f / k) + 0.5 * sd**2) / sd)


def implied_vol(price: float, f: float, k: float, tau: float) -> float:
    """Black-76 implied vol of a call; NaN if the price violates the bounds."""
    intrinsic = max(f - k, 0.0)
    if not (np.isfinite(price) and price > intrinsic + 1e-10 and price < f and tau > 0):
        return float("nan")
    return brentq(lambda v: black_call(f, k, tau, v) - price, 1e-4, 5.0, xtol=1e-10)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_fo(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """All NIFTY rows from the daily extracts (optionally a date window)."""
    manifest = [json.loads(line) for line in (RAW / "MANIFEST.jsonl").read_text().splitlines()
                if line.strip()]
    files = sorted({m["file"] for m in manifest if "file" in m
                    and (start is None or m["date"] >= start) and (end is None or m["date"] <= end)})
    parts = []
    for rel in files:
        with gzip.open(PROJECT_ROOT / rel, "rt") as handle:
            day = pd.read_csv(handle)
        # Keep futures and monthly options only (expiry = a futures expiry that
        # day); weekly options are about half the rows and are not used.
        monthly = set(day.loc[day["instrument"] == "FUT", "expiry"])
        parts.append(day[(day["instrument"] == "FUT") | day["expiry"].isin(monthly)])
    fo = pd.concat(parts, ignore_index=True)
    fo["date"] = pd.to_datetime(fo["date"])
    fo["expiry"] = pd.to_datetime(fo["expiry"])
    return fo


def split_fo(fo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(futures, monthly calls and puts)."""
    fut = fo[fo["instrument"] == "FUT"].copy()
    monthly = set(fut["expiry"].unique())
    opt = fo[fo["instrument"].isin(["CE", "PE"]) & fo["expiry"].isin(monthly)].copy()
    return fut, opt


def trading_days_between(calendar: pd.DatetimeIndex, start, end) -> int:
    """Number of trading days in (start, end]."""
    return int(calendar.searchsorted(end, side="right") - calendar.searchsorted(start, side="right"))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_checks(fo: pd.DataFrame, nifty: pd.DataFrame, vix: pd.Series) -> dict:
    fut, opt = split_fo(fo)
    cal = nifty.index
    report: dict = {"rows": int(len(fo)), "days": int(fo["date"].nunique()),
                    "first": str(fo["date"].min().date()), "last": str(fo["date"].max().date())}
    # 1. Expiring futures settle at the index close on expiry day.
    exp_rows = fut[fut["date"] == fut["expiry"]].set_index("date")
    joined = exp_rows.join(nifty["close"].rename("index_close"), how="inner")
    rel = (joined["settle"] / joined["index_close"] - 1).abs()
    report["expiry_settle_vs_index"] = {
        "expiries": int(len(joined)), "median_abs_rel": float(rel.median()),
        "max_abs_rel": float(rel.max()),
        "worst": {str(d.date()): float(v) for d, v in rel.nlargest(5).items()}}
    # 2. Near-month futures basis F/S - 1 (cost of carry).
    near = fut.sort_values("expiry").groupby("date").first()
    basis = (near["close"] / nifty["close"].reindex(near.index) - 1).dropna()
    report["near_futures_basis_pct"] = {
        "q01": float(100 * basis.quantile(0.01)), "median": float(100 * basis.median()),
        "q99": float(100 * basis.quantile(0.99)),
        "outside_minus1_plus2": int(((basis < -0.01) | (basis > 0.02)).sum())}
    # 3. Put-call parity on traded near-ATM monthly pairs: C - P vs F - K.
    traded = opt[opt["contracts"] > 0]
    calls = traded[traded["instrument"] == "CE"].set_index(["date", "expiry", "strike"])["close"]
    puts = traded[traded["instrument"] == "PE"].set_index(["date", "expiry", "strike"])["close"]
    pairs = pd.concat([calls.rename("c"), puts.rename("p")], axis=1, join="inner").reset_index()
    fclose = fut.set_index(["date", "expiry"])["close"].rename("f")
    pairs = pairs.join(fclose, on=["date", "expiry"]).dropna()
    pairs = pairs[(pairs["strike"] / pairs["f"] - 1).abs() < 0.03]
    dev = (pairs["c"] - pairs["p"] - (pairs["f"] - pairs["strike"])) / pairs["f"]
    report["put_call_parity_near_atm"] = {
        "pairs": int(len(pairs)), "median_dev_bp": float(1e4 * dev.median()),
        "q01_bp": float(1e4 * dev.quantile(0.01)), "q99_bp": float(1e4 * dev.quantile(0.99)),
        "abs_dev_above_50bp": int((dev.abs() > 0.005).sum())}
    # 4. No-arbitrage bounds for traded calls: exp(-r tau)(F-K)+ <= C <= F.
    # The discount factor is unknown here, so we use r = 10% as a generous bound;
    # a violation below that is a stale or non-synchronous print, not discounting.
    tc = traded[traded["instrument"] == "CE"].join(fclose, on=["date", "expiry"]).dropna(subset=["f"])
    tau = np.array([trading_days_between(cal, d, e) for d, e in zip(tc["date"], tc["expiry"])]) / YEAR
    lower = np.exp(-0.10 * tau) * (tc["f"] - tc["strike"]).clip(lower=0)
    tolerance = 0.0005 * tc["f"]  # 5 bp of the forward
    below = tc["close"] < lower - tolerance
    moneyness = tc["strike"] / tc["f"] - 1
    near_atm = moneyness.abs() < 0.03
    report["call_bounds_violations"] = {
        "traded_calls": int(len(tc)),
        "below_discounted_intrinsic_minus_5bp": int(below.sum()),
        "of_which_near_atm_3pct": int((below & near_atm).sum()),
        "near_atm_traded_calls": int(near_atm.sum()),
        "violations_by_moneyness_K_over_F": {
            label: int((below & mask).sum()) for label, mask in (
                ("below_0.90", moneyness < -0.10), ("0.90_0.97", (moneyness >= -0.10) & (moneyness < -0.03)),
                ("0.97_1.03", near_atm), ("above_1.03", moneyness >= 0.03))},
        "above_forward": int((tc["close"] > tc["f"]).sum())}
    # 5. ATM implied vol of the near-month (>= 5 trading days) vs India VIX.
    rows = []
    for (day, expiry), grp in pairs.groupby(["date", "expiry"]):
        tau_days = trading_days_between(cal, day, expiry)
        if tau_days < 5:
            continue
        row = grp.iloc[(grp["strike"] - grp["f"]).abs().argmin()]
        iv = implied_vol(row["c"], row["f"], row["strike"], tau_days / YEAR)
        rows.append((day, expiry, tau_days, iv))
    atm = pd.DataFrame(rows, columns=["date", "expiry", "tau_days", "iv"])
    atm = atm.sort_values("tau_days").groupby("date").first()  # nearest monthly expiry
    both = pd.concat([atm["iv"], vix.reindex(atm.index) / 100], axis=1, keys=["iv", "vix"]).dropna()
    report["atm_iv_vs_india_vix"] = {
        "days": int(len(both)), "corr": float(both.corr().iloc[0, 1]),
        "median_ratio_iv_over_vix": float((both["iv"] / both["vix"]).median())}
    return report


def main():
    nifty = pd.read_csv(PROCESSED / "nifty50.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(PROCESSED / "indiavix.csv", index_col=0, parse_dates=True)["close"]
    fo = load_fo()
    report = run_checks(fo, nifty, vix)
    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / "fo_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
