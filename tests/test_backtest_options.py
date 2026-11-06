"""Checks for the real-option backtest (src/fo_data.py, src/backtest_options.py).

Uses Jul-Dec 2010 bhavcopy extracts (the first part of the download).
Run: python -m pytest tests/test_backtest_options.py -q
"""

from __future__ import annotations

import gzip
import math

import numpy as np
import pandas as pd
import pytest

from src import backtest_options as bo
from src import backtest_real as bt
from src import fo_data as fd
from src import rebalancing as rb
from src.upstream import PROJECT_ROOT

END = "2010-12-31"


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_call_scalar(f, k, tau, vol):
    sd = vol * math.sqrt(tau)
    d1 = (math.log(f / k) + 0.5 * sd * sd) / sd
    return f * norm_cdf(d1) - k * norm_cdf(d1 - sd)


def iv_bisect(price, f, k, tau):
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if black_call_scalar(f, k, tau, mid) < price else (lo, mid)
    return 0.5 * (lo + hi)


@pytest.fixture(scope="module")
def setup():
    nifty = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv", index_col=0, parse_dates=True)
    fo = fd.load_fo(end=END)
    return nifty, fo, bo.Market(fo, nifty.index)


def test_implied_vol_round_trip():
    for f, k, tau, vol in ((100, 100, 21 / 252, 0.2), (100, 95, 10 / 252, 0.35), (100, 103, 30 / 252, 0.12)):
        price = float(fd.black_call(f, k, tau, vol))
        assert abs(fd.implied_vol(price, f, k, tau) - vol) < 1e-8
    assert math.isnan(fd.implied_vol(4.0, 100, 95, 0.1))  # below intrinsic


def test_expiring_future_settles_at_index_close(setup):
    nifty, fo, _ = setup
    fut = fo[(fo["instrument"] == "FUT") & (fo["date"] == fo["expiry"])]
    assert len(fut) >= 4
    for _, r in fut.iterrows():
        assert r["settle"] == pytest.approx(nifty.loc[r["date"], "close"], abs=0.01)


def test_one_option_by_hand_from_raw_files(setup):
    """Rebuild one option directly from the gzipped daily files and compare."""
    nifty, _, mkt = setup
    day = pd.Timestamp("2010-08-02")
    opt, why = bo.make_option(mkt, day, pd.Series(dtype=float), "own")
    assert opt is not None, why

    def rows(d):
        path = PROJECT_ROOT / "data" / "raw" / "bhavcopy" / "nifty" / f"{d:%Y}" / f"{d:%Y%m%d}.csv.gz"
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle, parse_dates=["date", "expiry"])

    expiry, dates = opt["label"], opt["dates"]  # rows are keyed by the contract label
    r0 = rows(day)
    fut0 = r0[(r0.instrument == "FUT") & (r0.expiry == expiry)]["close"].item()
    calls0 = r0[(r0.instrument == "CE") & (r0.expiry == expiry) & (r0.contracts > 0)]
    k = calls0.loc[(calls0.strike - fut0).abs().idxmin(), "strike"]
    premium = calls0.loc[calls0.strike == k, "close"].item()
    assert opt["strike"] * fut0 == pytest.approx(k)
    assert opt["premium"] * fut0 == pytest.approx(premium)
    n = len(dates) - 1
    gain, fprev = 0.0, None
    for i, d in enumerate(dates[:-1]):
        r = rows(d)
        f = r[(r.instrument == "FUT") & (r.expiry == expiry)]["close"].item()
        c = r[(r.instrument == "CE") & (r.expiry == expiry) & (r.strike == k) & (r.contracts > 0)]["close"]
        tau = (n - i) / 252
        vol = iv_bisect(c.item(), f, k, tau) if len(c) and c.item() > max(f - k, 0) else opt["vol"][i]
        sd = vol * math.sqrt(tau)
        delta = norm_cdf((math.log(f / k) + 0.5 * sd * sd) / sd)
        f_next = (rows(dates[i + 1]).query("instrument == 'FUT' and expiry == @expiry")
                  ["settle" if i + 1 == n else "close"].item())
        gain += delta * (f_next - f)
    phi, _ = rb.held_positions(
        fd.black_delta(opt["f"][None, :-1], opt["strike"], (n - np.arange(n))[None, :] / 252, opt["vol"][None, :]),
        rb.FixedFrequency(1))
    ours = rb.trading_outcome(phi, opt["f"][None, :])["gain"][0] * fut0
    assert ours == pytest.approx(gain, rel=1e-6, abs=1e-6)


def test_grouped_engine_matches_single_option(setup):
    nifty, _, mkt = setup
    panel = bt.load_panel()
    labels = {k: pd.Series(v, index=panel.index)
              for k, v in bt.label_variants(panel, ["vix_hyst"]).items()}
    opts = [o for d in pd.bdate_range("2010-08-02", "2010-09-30")
            for o, _ in [bo.make_option(mkt, d, panel["vix"], "own")] if o is not None]
    n = opts[0]["n"]
    group = [o for o in opts if o["n"] == n]
    rules = bt.build_rules(list(labels), n)
    together = bo.evaluate_group(group, labels, rules)
    for j, o in enumerate(group):
        alone = bo.evaluate_group([o], labels, rules)
        for key in alone:
            assert np.allclose(alone[key][0], together[key][j])


def test_future_data_does_not_change_past_positions(setup):
    """Perturb all bhavcopy rows after a date: positions before it are unchanged."""
    nifty, fo, mkt = setup
    cut = pd.Timestamp("2010-08-16")
    fo2 = fo.copy()
    late = fo2["date"] > cut
    fo2.loc[late, ["close", "settle"]] *= 1.07
    mkt2 = bo.Market(fo2, nifty.index)
    day = pd.Timestamp("2010-08-02")
    o1, _ = bo.make_option(mkt, day, pd.Series(dtype=float), "own")
    o2, _ = bo.make_option(mkt2, day, pd.Series(dtype=float), "own")
    k = int(np.sum(o1["dates"][:-1] <= cut))
    assert np.allclose(o1["vol"][:k], o2["vol"][:k]) and np.allclose(o1["f"][:k], o2["f"][:k])
    n = o1["n"]
    tau = (n - np.arange(n))[None, :] / 252
    t1 = fd.black_delta(o1["f"][None, :-1], o1["strike"], tau, o1["vol"][None, :])
    t2 = fd.black_delta(o2["f"][None, :-1], o2["strike"], tau, o2["vol"][None, :])
    for rule in (rb.FixedFrequency(3), rb.NoTradeBand(0.05)):
        p1, _ = rb.held_positions(t1, rule)
        p2, _ = rb.held_positions(t2, rule)
        assert np.array_equal(p1[:, :k], p2[:, :k])


def test_actual_expiry_handles_holidays():
    """Moved expiries: label 2014-04-24 settled on 2014-04-23 (holiday); stale
    labels 2014-02-27 and 2023-06-29 are not real final settlements."""
    nifty = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv", index_col=0, parse_dates=True)
    fo = pd.concat([fd.load_fo(start="2013-12-01", end="2014-05-01"),
                    fd.load_fo(start="2023-06-01", end="2023-07-05")])
    mkt = bo.Market(fo, nifty.index)
    # Moved expiry, same label: settles the day before the label date.
    assert mkt.actual_expiry[pd.Timestamp("2014-04-24")] == pd.Timestamp("2014-04-23")
    # Relabelled contracts are merged into one contract under the new label.
    assert mkt.aliases[pd.Timestamp("2023-06-29")] == pd.Timestamp("2023-06-28")
    assert mkt.aliases[pd.Timestamp("2014-02-27")] == pd.Timestamp("2014-02-26")
    assert mkt.actual_expiry[pd.Timestamp("2023-06-28")] == pd.Timestamp("2023-06-28")
    # The merged June-2023 contract has a continuous futures path up to expiry.
    days = nifty.index[(nifty.index >= "2023-06-01") & (nifty.index <= "2023-06-28")]
    assert all((d, pd.Timestamp("2023-06-28")) in mkt.fut_close for d in days)
