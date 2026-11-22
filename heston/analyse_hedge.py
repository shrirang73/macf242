"""Paired significance of hedging comparisons by period, regime and event.

Usage:  python -m heston.analyse_hedge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INPUTS, ROOT
from .stats import EVENTS, TEST_START, in_window, paired_ratio

PAIRS = [("e_heston_mv", "e_bs_own"), ("e_heston", "e_bs_own"), ("e_heston_mv", "e_heston")]


def table(df, label):
    rows = []
    for a, b in PAIRS:
        d = df.dropna(subset=[a, b])
        if len(d) < 30:
            continue
        r = paired_ratio(d, a, b, kind="var", n_boot=1000, block=21)
        rows.append({"sample": label, "a": a[2:], "b": b[2:], **r})
    return rows


def main():
    h = pd.read_csv(ROOT / "results" / "hedge.csv.gz", parse_dates=["date", "expiry"])
    reg = pd.read_csv(INPUTS / "regime_labels.csv", parse_dates=["date"])
    h = h.merge(reg[["date", "vix", "vix_hyst"]], on="date", how="left")
    rows = []
    rows += table(h, "all")
    rows += table(h[h.date < TEST_START], "train 2010-18")
    rows += table(h[h.date >= TEST_START], "test 2019-26")
    for flag, name in ((0, "calm"), (1, "stress")):
        rows += table(h[(h.vix_hyst == flag) & (h.date >= TEST_START)], f"test, {name}")
    for name, (s, e) in EVENTS.items():
        rows += table(in_window(h, s, e), name)
    covid = EVENTS["COVID-19 crash"]
    ex = h[(h.date >= TEST_START) & ~((h.date >= covid[0]) & (h.date <= covid[1]))]
    rows += table(ex, "test excl. COVID")
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))
    out.to_csv(ROOT / "results" / "hedge_paired.csv", index=False)


if __name__ == "__main__":
    main()
