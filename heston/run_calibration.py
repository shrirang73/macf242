"""Stage 1: calibrate Heston to every day's NIFTY option chain (warm-started).

Output: results/params.csv with one row per day:
    date, v0, kappa, theta, xi, rho, rmse_iv (in-sample, vol points), n_opt, n_exp, ok

Usage:  python -m heston.run_calibration [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .calib import DEFAULT, calibrate
from .data import OUT, ROOT

MIN_OPTIONS = 8


def run(start=None, end=None) -> pd.DataFrame:
    chains = pd.read_csv(OUT / "chains.csv.gz", parse_dates=["date", "expiry"])
    if start:
        chains = chains[chains["date"] >= start]
    if end:
        chains = chains[chains["date"] <= end]
    rows, prior = [], None
    t0 = time.time()
    for n, (date, g) in enumerate(chains.groupby("date")):
        # one forward per expiry: Heston is calibrated on the forward of each expiry,
        # so express every option in terms of its own forward via moneyness K/F.
        # With r = q = 0 the model is homogeneous in F, so we price on F = 1.
        K = (g["strike"] / g["F"]).to_numpy()
        T = g["tau"].to_numpy()
        iv = g["iv"].to_numpy()
        if len(g) < MIN_OPTIONS:
            continue
        p, rmse, ok = calibrate(1.0, K, T, iv, prior=prior)
        prior = p
        rows.append(dict(date=date, v0=p.v0, kappa=p.kappa, theta=p.theta, xi=p.xi,
                         rho=p.rho, rmse_iv=rmse, n_opt=len(g),
                         n_exp=g["expiry"].nunique(), ok=ok))
        if n % 100 == 0:
            print(f"{date.date()}  rmse={rmse*100:.2f} vol pts  ({time.time()-t0:.0f}s)", flush=True)
            # checkpoint so an interrupted run keeps its progress
            pd.DataFrame(rows).to_csv(ROOT / "results" / "params_partial.csv", index=False)
    return pd.DataFrame(rows)


def _chunk(args):
    """Calibrate one date chunk and save it at once, so a crash loses at most one chunk."""
    i, bounds = args
    part_file = ROOT / "results" / "chunks" / f"chunk_{i:02d}.csv"
    if part_file.exists():                       # resume: skip finished chunks
        return pd.read_csv(part_file, parse_dates=["date"])
    res = run(*bounds)
    part_file.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(part_file, index=False)
    return res


def run_parallel(workers: int = 5) -> pd.DataFrame:
    """Split the sample into equal-size date chunks and calibrate them in parallel.

    Each chunk is warm-started within itself; its first day starts from the default
    parameters (two starts), so chunk boundaries do not leak information between
    chunks. Results are identical in kind to the sequential run."""
    from concurrent.futures import ProcessPoolExecutor
    chains = pd.read_csv(OUT / "chains.csv.gz", usecols=["date"], parse_dates=["date"])
    dates = np.sort(chains["date"].unique())
    edges = np.array_split(dates, workers * 2)
    bounds = [(str(pd.Timestamp(e[0]).date()), str(pd.Timestamp(e[-1]).date())) for e in edges]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_chunk, enumerate(bounds)))
    return pd.concat(parts, ignore_index=True).sort_values("date")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="params.csv")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    res = run_parallel(a.workers) if a.workers else run(a.start, a.end)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / a.out, index=False)
    print(res.describe().T[["mean", "50%", "min", "max"]].round(4))
