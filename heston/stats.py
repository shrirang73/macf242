"""Paired comparison tools with a moving-block bootstrap.

Daily errors are serially dependent (overlapping option lives, persistent
volatility), so confidence intervals use blocks of consecutive days, never
individual observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stress episodes inside the NIFTY option sample. Windows are fixed here, before
# looking at results; they are market events, not fitted to the data.
EVENTS = {
    "US downgrade / Euro crisis": ("2011-08-01", "2011-12-31"),
    "Taper tantrum": ("2013-05-22", "2013-09-30"),
    "Demonetisation": ("2016-11-08", "2017-01-31"),
    # same COVID window as the earlier project and the time-series experiment
    # (backtest_analysis --exclude-test 2020-01-15 2020-06-30), so one definition is used throughout
    "COVID-19 crash": ("2020-01-15", "2020-06-30"),
    "2022 rate-hike sell-off": ("2022-01-01", "2022-06-30"),
}
TEST_START = "2019-01-01"       # chronological split, as in the earlier project


def block_bootstrap(date_index, stat, n_boot=2000, block=21, seed=0):
    """Moving-block bootstrap of stat(idx) where idx are row positions.

    `date_index` is the (sorted) array of dates of the rows; rows on the same date
    stay together by resampling blocks of *unique dates*."""
    rng = np.random.default_rng(seed)
    dates = pd.Index(date_index)
    uniq = dates.unique().sort_values()
    pos_by_date = {d: np.flatnonzero(dates == d) for d in uniq} if len(uniq) < 600 else None
    if pos_by_date is None:                    # faster grouping for long samples
        order = np.argsort(dates.values, kind="stable")
        sorted_dates = dates.values[order]
        bounds = np.flatnonzero(np.r_[True, sorted_dates[1:] != sorted_dates[:-1], True])
        pos_by_date = {uniq[i]: order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))}
    groups = [pos_by_date[d] for d in uniq]
    n = len(groups)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.concatenate(groups[s:s + block]) for s in starts])
        draws[b] = stat(idx)
    return draws


def paired_ratio(df, col_a, col_b, kind="mse", **kw):
    """Relative change of loss(a) vs loss(b) in percent, with bootstrap SE and z.

    kind='mse' : mean squared value (pricing errors, centred at zero by design)
    kind='var' : variance (hedging errors, whose mean carries the risk premium)
    """
    a, b = df[col_a].to_numpy(), df[col_b].to_numpy()
    f = (lambda x: np.mean(x ** 2)) if kind == "mse" else np.var
    point = 100 * (f(a) / f(b) - 1)
    draws = block_bootstrap(df["date"].values, lambda i: 100 * (f(a[i]) / f(b[i]) - 1), **kw)
    se = draws.std(ddof=1)
    return dict(change_pct=point, se=se, z=point / se if se > 0 else np.nan,
                lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5), n=len(df))


def in_window(df, start, end, col="date"):
    return df[(df[col] >= start) & (df[col] <= end)]
