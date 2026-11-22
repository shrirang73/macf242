"""Stage 3: delta-hedging backtest of a short ATM monthly NIFTY call.

Each trading day t0 we sell the monthly call (expiry nearest 21 trading days,
at least 10) at the listed strike nearest the same-expiry future F(t0), and
hedge daily with that future until expiry. Hedge ratios:

  bs_own   : Black-76 delta at the option's own implied vol (carried forward if
             the contract does not trade that day; ATM IV of the expiry as fallback).
  heston   : dC/dF under the Heston parameters calibrated at the close of the day
             (stochastic variance held fixed).
  heston_mv: minimum-variance delta  dC/dF + (rho * xi / F) * dC/dv0.  With
             d<v,S> = rho*xi*v*S dt and d<S> = v*S^2 dt this is the ratio
             d<C,S>/d<S> (the GKW projection of the call onto the stock).
  none     : no hedge.

Error e = C0 - (F_E - K)^+ + sum_k phi_k (F_{k+1} - F_k), per unit of F(t0).
Costs are ignored (identical option trade across hedges; futures turnover is
reported separately). All inputs at day k are known at the close of day k.
Near expiry (< 5 trading days) the time to expiry is floored at 5 days in the
delta formulae for BOTH models; the future's P&L is always exact.

Usage:  python -m heston.hedge
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .calib import call_vec
from .data import INPUTS, OUT, ROOT, YEAR, black_call
from .pricer import HestonParams

TARGET_DAYS, MIN_DAYS, TAU_FLOOR = 21, 10, 5
LOGM_MAX = 0.01     # start strike within 1% of F


def bs_delta(m, tau, vol):
    """Black-76 call delta for moneyness m = K/F (zero discounting)."""
    sd = vol * np.sqrt(tau)
    return ndtr((-np.log(m) + 0.5 * sd ** 2) / sd)


def heston_deltas(m, tau, p: HestonParams, h=1e-3):
    """(delta, mv_delta) for arrays of moneyness m and tau, F = 1.

    C(F,K) = F c(K/F)  =>  dC/dF = c - m c'(m).  dC/dv0 by central bump."""
    c = call_vec(1.0, m, tau, p)
    cp = (call_vec(1.0, m * (1 + h), tau, p) - call_vec(1.0, m * (1 - h), tau, p)) / (2 * m * h)
    delta = c - m * cp
    hv = 0.05 * p.v0
    up = HestonParams(p.v0 + hv, p.kappa, p.theta, p.xi, p.rho)
    dn = HestonParams(p.v0 - hv, p.kappa, p.theta, p.xi, p.rho)
    dcdv = (call_vec(1.0, m, tau, up) - call_vec(1.0, m, tau, dn)) / (2 * hv)
    return delta, delta + p.rho * p.xi * dcdv


def build_positions(chains, fut, nifty):
    """One short call per start day; returns list of dicts with path data."""
    fpath = {e: g.set_index("date")["close"] for e, g in fut.groupby("expiry")}
    close = nifty.set_index("date")["close"]
    cal = pd.DatetimeIndex(sorted(chains["date"].unique()))
    by_day = {d: g for d, g in chains.groupby("date")}
    pos, dropped = [], {"no_start": 0, "short_path": 0}
    for d0 in cal:
        g = by_day[d0]
        cand = g[(g["tau_days"] >= MIN_DAYS)]
        if cand.empty:
            dropped["no_start"] += 1
            continue
        exp = cand.iloc[(cand["tau_days"] - TARGET_DAYS).abs().argsort()[:1]]["expiry"].iloc[0]
        ge = cand[cand["expiry"] == exp]
        row = ge.iloc[np.abs(np.log(ge["strike"] / ge["F"])).argmin()]
        if abs(np.log(row["strike"] / row["F"])) > LOGM_MAX:
            dropped["no_start"] += 1
            continue
        fp = fpath.get(exp)
        if fp is None or exp not in close.index:
            dropped["short_path"] += 1
            continue
        days = fp.index[(fp.index >= d0) & (fp.index <= exp)]
        # require the path to contain every trading day of the NIFTY calendar to expiry
        nd = close.index[(close.index >= d0) & (close.index <= exp)]
        if len(days) < 2 or len(days) != len(nd) or (days != nd).any():
            dropped["short_path"] += 1
            continue
        pos.append(dict(d0=d0, expiry=exp, K=float(row["strike"]), c0=float(row["call"]),
                        days=days, F=fp.loc[days].to_numpy(float), S_E=float(close.loc[exp]),
                        iv0=float(row["iv"])))
    return pos, dropped


def run():
    chains = pd.read_csv(OUT / "chains.csv.gz", parse_dates=["date", "expiry"])
    fut = pd.read_csv(OUT / "futures.csv.gz", parse_dates=["date", "expiry"])
    nifty = pd.read_csv(INPUTS / "nifty50.csv", parse_dates=["date"])
    params = pd.read_csv(ROOT / "results" / "params.csv", parse_dates=["date"]).set_index("date")
    pos, dropped = build_positions(chains, fut, nifty)

    # market IV lookup: (date, expiry, strike) -> iv ; ATM IV by (date, expiry)
    iv_tab = chains.set_index(["date", "expiry", "strike"])["iv"]
    iv_tab = iv_tab[~iv_tab.index.duplicated()]
    atm = chains.loc[np.abs(np.log(chains["strike"] / chains["F"])).groupby(
        [chains["date"], chains["expiry"]]).idxmin()].set_index(["date", "expiry"])["iv"]

    # Per position: hedge-ratio paths. Group by day so Heston is vectorised.
    for p_ in pos:
        n = len(p_["days"]) - 1                       # hedge dates k = 0..n-1
        p_["phi"] = {k: np.full(n, np.nan) for k in ("bs_own", "heston", "heston_mv")}
        p_["iv"] = np.full(n, np.nan)                 # implied vol used on each hedge date
        p_["ok"] = True
    day_jobs = {}
    for i, p_ in enumerate(pos):
        for k, d in enumerate(p_["days"][:-1]):
            day_jobs.setdefault(d, []).append((i, k))
    for d, jobs in day_jobs.items():
        if d not in params.index:
            for i, _ in jobs:
                pos[i]["ok"] = False
            continue
        pr = params.loc[d]
        hp = HestonParams(*pr[["v0", "kappa", "theta", "xi", "rho"]].astype(float))
        m = np.array([pos[i]["K"] / pos[i]["F"][k] for i, k in jobs])
        tdays = np.array([max(len(pos[i]["days"]) - 1 - k, TAU_FLOOR) for i, k in jobs])
        tau = tdays / YEAR
        dh, dmv = heston_deltas(m, tau, hp)
        for j, (i, k) in enumerate(jobs):
            p_ = pos[i]
            key = (d, p_["expiry"], p_["K"])
            v = iv_tab.get(key, np.nan)
            if not np.isfinite(v):
                v = atm.get((d, p_["expiry"]), np.nan)
            if not np.isfinite(v):                    # carry forward the last known vol
                prev = p_["phi"].get("_iv", p_["iv0"])
                v = prev
            p_["phi"]["_iv"] = v
            p_["iv"][k] = v
            p_["phi"]["bs_own"][k] = bs_delta(m[j], tau[j], v)
            p_["phi"]["heston"][k] = dh[j]
            p_["phi"]["heston_mv"][k] = dmv[j]

    rows, paths = [], []
    for p_ in pos:
        if not p_["ok"]:
            dropped["short_path"] += 1
            continue
        F, K = p_["F"], p_["K"]
        # Daily path for mark-to-market: option value V_k = Black-76 at the implied vol
        # used that day (the option's own market vol when quoted), with the actual time
        # to expiry; V_0 = the traded premium, V_n = the payoff at settlement.
        n = len(F) - 1
        tau_k = (n - np.arange(n)) / YEAR
        V = np.empty(n + 1)
        V[:n] = black_call(F[:n], K, tau_k, p_["iv"])
        V[0], V[n] = p_["c0"], max(p_["S_E"] - K, 0.0)
        opt = len(rows)
        for k in range(n + 1):
            paths.append((opt, k, p_["days"][k], F[k], V[k],
                          *(p_["phi"][h][k] if k < n else np.nan
                            for h in ("bs_own", "heston", "heston_mv")),
                          p_["iv"][k] if k < n else np.nan))
        dF = np.diff(F)
        payoff = max(p_["S_E"] - K, 0.0)             # future settles at the index close
        base = p_["c0"] - payoff
        row = dict(date=p_["d0"], expiry=p_["expiry"], K=K, F0=F[0], c0=p_["c0"],
                   n_days=len(dF), logm=np.log(K / F[0]))
        row["e_none"] = base / F[0]
        for name in ("bs_own", "heston", "heston_mv"):
            phi = p_["phi"][name]
            if not np.isfinite(phi).all():
                row[f"e_{name}"] = np.nan
                continue
            row[f"e_{name}"] = (base + (phi * dF).sum()) / F[0]
            row[f"turn_{name}"] = np.abs(np.diff(np.concatenate([[0.0], phi]))).sum()
        rows.append(row)
    res = pd.DataFrame(rows)
    path_df = pd.DataFrame(paths, columns=["opt", "k", "date", "F", "V", "phi_bs_own",
                                           "phi_heston", "phi_heston_mv", "iv"])
    print("dropped:", dropped, "| kept:", len(res), "| path rows:", len(path_df))
    return res, path_df


if __name__ == "__main__":
    res, path_df = run()
    res.to_csv(ROOT / "results" / "hedge.csv.gz", index=False)
    # one row per option per trading day (opt = row number in hedge.csv.gz)
    path_df.to_csv(ROOT / "results" / "hedge_paths.csv.gz", index=False)
    e = res[[c for c in res if c.startswith("e_")]].dropna()
    print("RMSE of hedging error (bp of F0):")
    print((e.pow(2).mean() ** 0.5 * 1e4).round(1))
    print("variance reduction vs none:")
    print((1 - e.var() / e["e_none"].var()).round(3))
