"""Build clean daily NIFTY option chains from the verified NSE bhavcopy extract.

Source files (bundled in this project, copied unmodified from our earlier project):
    data/raw/bhavcopy/nifty/YYYY/YYYYMMDD.csv.gz

Conventions (same as the earlier project's notes/options_log.md):
* Forward F = close of the same-expiry NIFTY future (Black-76, zero discounting).
* Only monthly expiries (an option expiry that is also a futures expiry).
* A price counts only if contracts > 0 (untraded contracts carry stale closes).
* Out-of-the-money options only (puts for K < F, calls for K >= F); a put is
  converted to its call-equivalent by parity, C = P + F - K.
* tau = trading days to expiry / 252, counted on the NIFTY calendar.

Usage:  python -m heston.data
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
RAW = OUT / "raw" / "bhavcopy" / "nifty"          # one gzipped CSV per trading day
INPUTS = OUT / "inputs"                            # nifty50.csv, indiavix.csv, regime_labels.csv
YEAR = 252
MIN_PRICE = 0.5          # index points; below this the tick size dominates
MAX_LOGMONEY = 0.12      # |ln(K/F)| window
TAU_DAYS = (5, 70)       # keep expiries 5..70 trading days away


def black_call(f, k, tau, vol):
    sd = vol * np.sqrt(tau)
    d1 = (np.log(f / k) + 0.5 * sd ** 2) / sd
    return f * ndtr(d1) - k * ndtr(d1 - sd)


def black_iv(price, f, k, tau):
    """Black-76 implied vol; NaN if the price violates no-arbitrage bounds."""
    if not (price > max(f - k, 0.0) + 1e-9 and price < f):
        return np.nan
    try:
        return brentq(lambda v: black_call(f, k, tau, v) - price, 1e-3, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


def trading_calendar(dates) -> pd.DatetimeIndex:
    """NIFTY trading days, extended with weekdays after the last day in the data.

    Without the extension, an option expiring after the last known trading day
    would have its days-to-expiry counted only up to that day (too short). Beyond
    the data, NSE holidays are unknown, so weekdays are used: an approximation of
    at most a day or two, affecting only options expiring after the data end."""
    known = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    future = pd.bdate_range(known[-1] + pd.Timedelta(days=1), periods=400)
    return known.append(future)


def chain_for_day(day: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    date = pd.Timestamp(day["date"].iloc[0])
    fut = day[day["instrument"] == "FUT"].set_index("expiry")["close"]
    spot = float(day["underlying"].iloc[0])
    opt = day[day["instrument"].isin(["CE", "PE"]) & day["expiry"].isin(fut.index)
              & (day["contracts"] > 0) & (day["close"] > 0)].copy()
    opt["F"] = opt["expiry"].map(fut)
    # OTM only
    opt = opt[((opt["instrument"] == "PE") & (opt["strike"] < opt["F"]))
              | ((opt["instrument"] == "CE") & (opt["strike"] >= opt["F"]))]
    opt["call"] = np.where(opt["instrument"] == "PE",
                           opt["close"] + opt["F"] - opt["strike"], opt["close"])
    opt = opt[(opt["close"] >= MIN_PRICE)
              & (np.log(opt["strike"] / opt["F"]).abs() <= MAX_LOGMONEY)]
    exp = pd.to_datetime(opt["expiry"])
    opt["tau_days"] = (calendar.searchsorted(exp.values, side="right")
                       - calendar.searchsorted(date, side="right"))
    opt = opt[(opt["tau_days"] >= TAU_DAYS[0]) & (opt["tau_days"] <= TAU_DAYS[1])]
    opt["tau"] = opt["tau_days"] / YEAR
    opt["iv"] = [black_iv(c, f, k, t) for c, f, k, t in
                 zip(opt["call"], opt["F"], opt["strike"], opt["tau"])]
    opt = opt.dropna(subset=["iv"])
    opt["date"] = date
    opt["spot"] = spot
    return opt[["date", "expiry", "tau_days", "tau", "strike", "F", "spot", "instrument",
                "close", "call", "iv", "contracts", "oi"]]


def build(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    nifty = pd.read_csv(INPUTS / "nifty50.csv", parse_dates=["date"])
    calendar = trading_calendar(nifty["date"])
    files = sorted(RAW.glob("*/*.csv.gz"))
    parts = []
    for path in files:
        stamp = path.name[:8]
        d = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        if (start and d < start) or (end and d > end):
            continue
        with gzip.open(path, "rt") as h:
            day = pd.read_csv(h)
        if day.empty:
            continue
        parts.append(chain_for_day(day, calendar))
    chains = pd.concat(parts, ignore_index=True)
    chains["expiry"] = pd.to_datetime(chains["expiry"])
    return chains


def build_futures() -> pd.DataFrame:
    """Daily close of every NIFTY future (needed for hedge P&L to expiry)."""
    parts = []
    for path in sorted(RAW.glob("*/*.csv.gz")):
        with gzip.open(path, "rt") as h:
            day = pd.read_csv(h)
        f = day[day["instrument"] == "FUT"][["date", "expiry", "close", "contracts"]]
        parts.append(f)
    fut = pd.concat(parts, ignore_index=True)
    fut["date"] = pd.to_datetime(fut["date"])
    fut["expiry"] = pd.to_datetime(fut["expiry"])
    return fut


def main():
    OUT.mkdir(exist_ok=True)
    build_futures().to_csv(OUT / "futures.csv.gz", index=False)
    chains = build()
    chains.to_csv(OUT / "chains.csv.gz", index=False)
    n_days = chains["date"].nunique()
    print(f"{len(chains):,} option rows on {n_days} days; "
          f"{chains['date'].min().date()} .. {chains['date'].max().date()}")
    print(chains.groupby(chains["date"].dt.year).agg(
        rows=("iv", "size"), days=("date", "nunique"), med_iv=("iv", "median")).round(3))


if __name__ == "__main__":
    main()
