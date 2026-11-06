"""Build the verified daily NIFTY 50 and India VIX series and run data checks.

Inputs (all produced by ``data_download.py`` or saved from NSE, see data/raw/):
  processed/nifty50_niftyindices.csv  official NIFTY 50 (NSE Indices Ltd)
  processed/nifty50_yahoo.csv         Yahoo ^NSEI (cross-check only)
  processed/indiavix_yahoo.csv        Yahoo ^INDIAVIX
  raw/nse/vix_monthly_aggregates.txt  NSE India VIX monthly (count, sum, min, max)
  raw/nse/vix_daily_patch_months.txt  NSE India VIX daily closes, months with gaps
  raw/nse/vix_reconciliation_evidence.txt  how each Yahoo/NSE disagreement was settled

Outputs:
  processed/nifty50.csv   official closes, flags: special_session, in_yahoo
  processed/indiavix.csv  Yahoo closes patched with NSE (fills + corrections),
                          columns: close, source, verified
  results/data_checks/report.json   every check below with its outcome

Why these choices (details in notes/data_log.md):
* The official NIFTY series has every trading day; Yahoo lacks 47 of them but agrees to
  5e-8 relative error on all common days, so Yahoo is used only as a check.
* Yahoo India VIX lacks 49 days and is wrong on 2 days. NSE's own India VIX history
  has 2 placeholder rows (O=H=L=0). Each disagreement is settled with NSE's next-day
  previous-close field (raw/nse/vix_reconciliation_evidence.txt).
* NSE India VIX starts on 19 Jul 2010, so earlier Yahoo values cannot be verified
  and are flagged ``verified = False``.

Usage: python -m src.data_prepare
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .upstream import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
RAW_NSE = DATA / "raw" / "nse"
PROCESSED = DATA / "processed"
REPORT_DIR = PROJECT_ROOT / "results" / "data_checks"
NSE_VIX_START = pd.Timestamp("2010-07-19")
# Days on which Yahoo and NSE disagree, settled by NSE's next-day prev close.
VIX_CORRECTIONS = {  # date -> correct close
    "2013-08-22": 27.68,   # NSE consistent; Yahoo 29.06 wrong
    "2018-11-07": 17.8775,  # NSE consistent; Yahoo 17.72 wrong (Muhurat session)
}
VIX_KEEP_YAHOO = ["2021-02-12", "2021-03-30"]  # NSE rows are placeholders


def _data_lines(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def load_nse_monthly() -> pd.DataFrame:
    records = _data_lines(RAW_NSE / "vix_monthly_aggregates.txt")[0].split("|")
    frame = pd.DataFrame([r.split() for r in records], columns=["month", "n", "sum", "min", "max"])
    return frame.set_index("month").astype(float)


def load_nse_daily_patch() -> pd.Series:
    rows = {}
    for block in _data_lines(RAW_NSE / "vix_daily_patch_months.txt")[0].split("|"):
        month, values = block.split("=")
        for item in values.split(","):
            day, close = item.split(":")
            rows[pd.Timestamp(f"{month}-{day}")] = float(close)
    return pd.Series(rows).sort_index()


def build_nifty(checks: dict) -> pd.DataFrame:
    official = pd.read_csv(PROCESSED / "nifty50_niftyindices.csv", index_col=0, parse_dates=True)
    yahoo = pd.read_csv(PROCESSED / "nifty50_yahoo.csv", index_col=0, parse_dates=True)
    common = official.index.intersection(yahoo.index)
    rel = (yahoo.loc[common, "close"] / official.loc[common, "close"] - 1).abs()
    span = official.index >= yahoo.index[0]
    ohlc_bad = ((official["low"] > official[["open", "close"]].min(axis=1) + 1e-9)
                | (official["high"] < official[["open", "close"]].max(axis=1) - 1e-9))
    logret = np.log(official["close"]).diff()
    gaps = official.index.to_series().diff().dt.days
    checks["nifty"] = {
        "rows": len(official), "first": str(official.index[0].date()),
        "last": str(official.index[-1].date()),
        "duplicate_dates": int(official.index.duplicated().sum()),
        "missing_close": int(official["close"].isna().sum()),
        "yahoo_common_days": len(common),
        "yahoo_max_rel_close_diff": float(rel.max()),
        "days_missing_in_yahoo": int(len(official.index[span].difference(yahoo.index))),
        "days_only_in_yahoo": int(len(yahoo.index.difference(official.index))),
        "weekend_special_sessions": [str(d.date()) for d in official.index[official.index.dayofweek >= 5]],
        "ohlc_inconsistent_rows": [str(d.date()) for d in official.index[ohlc_bad]],
        "abs_log_return_above_6pct": {str(d.date()): round(float(r), 4)
                                      for d, r in logret[logret.abs() > 0.06].items()},
        "calendar_gaps_over_5_days": {str(d.date()): int(g) for d, g in gaps[gaps > 5].items()},
    }
    out = official[["open", "high", "low", "close"]].copy()
    out["special_session"] = out.index.dayofweek >= 5
    out["in_yahoo"] = out.index.isin(yahoo.index)
    return out


def build_vix(checks: dict, nifty_dates: pd.DatetimeIndex) -> pd.DataFrame:
    yahoo = pd.read_csv(PROCESSED / "indiavix_yahoo.csv", index_col=0, parse_dates=True)["close"]
    nse_patch = load_nse_daily_patch()
    vix = pd.DataFrame({"close": yahoo, "source": "yahoo"})
    # 1. Days Yahoo lacks: take NSE's value.
    fills = nse_patch.index.difference(vix.index)
    vix = pd.concat([vix, pd.DataFrame({"close": nse_patch[fills], "source": "nse_fill"})])
    # 2. Days where Yahoo is wrong.
    for day, value in VIX_CORRECTIONS.items():
        vix.loc[pd.Timestamp(day), ["close", "source"]] = [value, "nse_correction"]
    vix = vix.sort_index()
    vix["verified"] = vix.index >= NSE_VIX_START

    # Check A: every month from NSE start must now match NSE's monthly aggregates.
    # Tolerance: Yahoo rounds to 2 dp while NSE has 4 dp, so each value may differ
    # by up to 0.005 (sum by n * 0.005). Placeholder days are excluded from NSE by
    # construction of its aggregates, so those two months are allowed their known gap.
    nse_monthly = load_nse_monthly()
    sub = vix.loc[vix.index >= NSE_VIX_START, "close"]
    grouped = sub.groupby(sub.index.strftime("%Y-%m"))
    ours = pd.DataFrame({"n": grouped.size(), "sum": grouped.sum(),
                         "min": grouped.min(), "max": grouped.max()})
    joined = nse_monthly.join(ours, rsuffix="_ours", how="outer")
    known_gap = {pd.Timestamp(d).strftime("%Y-%m"): None for d in VIX_KEEP_YAHOO}
    tol_sum = 0.0051 * joined["n"]
    ok = ((joined["n"] == joined["n_ours"])
          & ((joined["sum"] - joined["sum_ours"]).abs() <= tol_sum)
          & ((joined["min"] - joined["min_ours"]).abs() <= 0.0051)
          & ((joined["max"] - joined["max_ours"]).abs() <= 0.0051))
    failed = joined.index[~ok]
    unexplained = [m for m in failed if m not in known_gap]

    # Check B: daily values in patch months agree with NSE except the known placeholders.
    common = nse_patch.index.intersection(vix.index)
    diff = (vix.loc[common, "close"] - nse_patch[common]).abs()
    daily_bad = [str(d.date()) for d in diff[diff > 0.0051].index]

    nifty_after = nifty_dates[nifty_dates >= NSE_VIX_START]
    checks["vix"] = {
        "rows": len(vix), "first": str(vix.index[0].date()), "last": str(vix.index[-1].date()),
        "rows_verified": int(vix["verified"].sum()),
        "source_counts": vix["source"].value_counts().to_dict(),
        "duplicate_dates": int(vix.index.duplicated().sum()),
        "range": [float(vix["close"].min()), float(vix["close"].max())],
        "months_checked_vs_nse": int(len(joined)),
        "months_matching_nse": int(ok.sum()),
        "months_not_matching": [str(m) for m in failed],
        "months_not_matching_unexplained": unexplained,
        "patch_month_days_differing_from_nse": daily_bad,
        "expected_differences_nse_placeholders": VIX_KEEP_YAHOO,
        "nifty_days_without_vix_since_2010_07_19": [
            str(d.date()) for d in nifty_after.difference(vix.index)],
        "vix_days_without_nifty": [str(d.date()) for d in vix.index.difference(nifty_dates)],
    }
    return vix


def main():
    checks: dict = {}
    nifty = build_nifty(checks)
    vix = build_vix(checks, nifty.index)
    nifty.to_csv(PROCESSED / "nifty50.csv")
    vix.to_csv(PROCESSED / "indiavix.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=1)[:5000])


if __name__ == "__main__":
    main()
