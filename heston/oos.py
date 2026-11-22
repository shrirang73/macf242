"""Stage 2: one-day-ahead out-of-sample pricing errors.

For every pair of consecutive trading days (t, t+1) and every option quoted on
both days (same expiry and strike), price the option at t+1 using ONLY
information available at the close of t:

  heston   : Heston parameters calibrated at t (v0 held at its t value),
             evaluated at t+1's forward and time to expiry.
  bs_own   : Black-76 with the option's own implied vol at t ("sticky strike",
             the standard market benchmark).
  bs_atm   : Black-76 with the ATM implied vol of that expiry at t (flat smile).

Errors are model implied vol minus market implied vol at t+1 (vol points).

Usage:  python -m heston.oos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calib import call_vec
from .data import OUT, ROOT, black_call, black_iv
from .pricer import HestonParams

MAX_GAP_DAYS = 5   # skip pairs separated by a long market closure


def _atm_iv(day: pd.DataFrame) -> dict:
    """ATM implied vol per expiry: the option whose strike is nearest the forward."""
    idx = (np.log(day["strike"] / day["F"]).abs()).groupby(day["expiry"]).idxmin()
    return day.loc[idx.values].set_index("expiry")["iv"].to_dict()


def run() -> pd.DataFrame:
    chains = pd.read_csv(OUT / "chains.csv.gz", parse_dates=["date", "expiry"])
    params = pd.read_csv(ROOT / "results" / "params.csv", parse_dates=["date"]).set_index("date")
    by_day = {d: g for d, g in chains.groupby("date")}
    dates = sorted(by_day)
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        if d0 not in params.index or (d1 - d0).days > MAX_GAP_DAYS:
            continue
        a, b = by_day[d0], by_day[d1]
        m = a.merge(b, on=["expiry", "strike"], suffixes=("0", "1"))
        if m.empty:
            continue
        p = HestonParams(*params.loc[d0, ["v0", "kappa", "theta", "xi", "rho"]].astype(float))
        atm = _atm_iv(a)
        K = (m["strike"] / m["F1"]).to_numpy()          # moneyness at t+1, F = 1
        T = m["tau1"].to_numpy()
        price_h = call_vec(1.0, K, T, p)
        iv_h = np.array([black_iv(c, 1.0, k, t) for c, k, t in zip(price_h, K, T)])
        iv_atm = m["expiry"].map(atm).to_numpy()
        out = pd.DataFrame({
            "date": d1, "expiry": m["expiry"], "strike": m["strike"],
            "logm": np.log(K), "tau_days": m["tau_days1"], "iv_mkt": m["iv1"],
            "err_heston": iv_h - m["iv1"].to_numpy(),
            "err_bs_own": m["iv0"].to_numpy() - m["iv1"].to_numpy(),
            "err_bs_atm": iv_atm - m["iv1"].to_numpy(),
            "F1": m["F1"], "close1": m["call1"],
        })
        rows.append(out)
    res = pd.concat(rows, ignore_index=True).dropna(subset=["err_heston"])
    return res


if __name__ == "__main__":
    res = run()
    res.to_csv(ROOT / "results" / "oos.csv.gz", index=False)
    print(len(res), "option-day errors")
    print((res[["err_heston", "err_bs_own", "err_bs_atm"]].pow(2).mean() ** 0.5 * 100).round(3))
