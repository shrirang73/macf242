"""Download daily NIFTY 50 and India VIX history (free sources).

Sources
-------
* Yahoo Finance chart API (primary; full history in one request):
    ^NSEI      NIFTY 50 index, from 17 Sep 2007
    ^INDIAVIX  India VIX, from 3 Mar 2008
* NSE Indices Ltd (niftyindices.com), the official NIFTY index provider
  (secondary; used only to cross-check the Yahoo NIFTY series). The site
  serves at most one year per request.
* NSE's own India VIX history (nseindia.com/api/historicalOR/vixhistory) blocks
  scripted requests, so India VIX was cross-checked on a sample fetched in a
  browser; see ``data/raw/nse_vix_sample.json`` and ``data_checks.py``.

Every raw response is saved unmodified under ``data/raw/`` and listed in
``data/raw/MANIFEST.json`` with its URL, retrieval time (UTC) and SHA-256, so
each processed number can be traced to a raw file. Parsed series are written
to ``data/processed/``.

Conventions
-----------
* Dates are exchange-local (Asia/Kolkata) trading dates.
* Bars dated on or after the retrieval date (IST) are dropped: today's bar may be
  intraday and is not final.
* Rows with a missing close are dropped and counted in the manifest.

Usage: python -m src.data_download
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .upstream import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
USER_AGENT = "Mozilla/5.0 (academic course project; MAC F244)"
YAHOO = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
         "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
NIFTYINDICES = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"


def _get(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT,
                                                              **(headers or {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


class Manifest:
    """Records every raw file written in this run."""

    def __init__(self):
        self.entries: list[dict] = []

    def save_raw(self, name: str, payload: bytes, url: str, note: str = "") -> Path:
        path = RAW / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.entries.append({
            "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "url": url,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "note": note,
        })
        return path

    def write(self, extra: dict):
        path = RAW / "MANIFEST.json"
        path.write_text(json.dumps({"files": self.entries, **extra}, indent=2))


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------


def parse_yahoo(payload: dict, retrieval_date_ist: dt.date) -> tuple[pd.DataFrame, dict]:
    """Chart-API JSON -> daily OHLC frame indexed by IST trading date."""
    if payload["chart"]["error"]:
        raise RuntimeError(payload["chart"]["error"])
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    if meta["exchangeTimezoneName"] != "Asia/Kolkata":
        raise ValueError(f"unexpected exchange timezone {meta['exchangeTimezoneName']}")
    quote = result["indicators"]["quote"][0]
    stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata")
    frame = pd.DataFrame({
        "date": stamps.date,
        "bar_time_ist": stamps.strftime("%H:%M"),
        "open": quote["open"], "high": quote["high"], "low": quote["low"],
        "close": quote["close"], "volume": quote["volume"],
    })
    info = {"rows_raw": len(frame), "symbol": meta["symbol"]}
    info["rows_missing_close"] = int(frame["close"].isna().sum())
    frame = frame.dropna(subset=["close"])
    unfinal = frame["date"] >= retrieval_date_ist
    info["rows_dropped_not_final"] = int(unfinal.sum())
    frame = frame[~unfinal]
    info["duplicate_dates"] = int(frame["date"].duplicated().sum())
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    info["first"], info["last"] = str(frame.index[0].date()), str(frame.index[-1].date())
    info["rows"] = len(frame)
    return frame, info


def download_yahoo(symbol: str, manifest: Manifest, now_utc: dt.datetime):
    p1 = int(dt.datetime(2007, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    p2 = int(now_utc.timestamp())
    url = YAHOO.format(symbol=symbol.replace("^", "%5E"), p1=p1, p2=p2)
    raw = _get(url)
    tag = symbol.strip("^").lower()
    manifest.save_raw(f"yahoo/{tag}_{now_utc:%Y%m%d}.json", raw, url, "Yahoo chart API, daily")
    return parse_yahoo(json.loads(raw), now_utc.astimezone(IST).date())


# ---------------------------------------------------------------------------
# niftyindices.com (official NIFTY provider), one year per request
# ---------------------------------------------------------------------------


def parse_niftyindices(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = pd.DataFrame({
        "date": pd.to_datetime(frame["HistoricalDate"], format="%d %b %Y"),
        "open": pd.to_numeric(frame["OPEN"], errors="coerce"),
        "high": pd.to_numeric(frame["HIGH"], errors="coerce"),
        "low": pd.to_numeric(frame["LOW"], errors="coerce"),
        "close": pd.to_numeric(frame["CLOSE"], errors="coerce"),
    })
    return frame.set_index("date").sort_index()


def download_niftyindices(manifest: Manifest, start: dt.date, end: dt.date,
                          pause: float = 1.0) -> pd.DataFrame:
    parts = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=364), end)
        fmt = lambda d: d.strftime("%d-%b-%Y")
        cinfo = (f"{{'name':'NIFTY 50','startDate':'{fmt(chunk_start)}',"
                 f"'endDate':'{fmt(chunk_end)}','indexName':'NIFTY 50'}}")
        body = json.dumps({"cinfo": cinfo}).encode()
        raw = _get(NIFTYINDICES, data=body, headers={
            "Content-Type": "application/json; charset=utf-8",
            "Referer": "https://www.niftyindices.com/reports/historical-data"})
        manifest.save_raw(f"niftyindices/nifty50_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json",
                          raw, NIFTYINDICES, f"POST cinfo={cinfo}")
        payload = json.loads(raw)
        rows = json.loads(payload["d"]) if isinstance(payload, dict) and "d" in payload else payload
        if rows:
            parts.append(parse_niftyindices(rows))
        chunk_start = chunk_end + dt.timedelta(days=1)
        time.sleep(pause)  # be polite to the server
    frame = pd.concat(parts)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def main():
    now_utc = dt.datetime.now(dt.timezone.utc)
    manifest = Manifest()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    infos = {}
    for symbol, name in (("^NSEI", "nifty50_yahoo"), ("^INDIAVIX", "indiavix_yahoo")):
        frame, info = download_yahoo(symbol, manifest, now_utc)
        frame.to_csv(PROCESSED / f"{name}.csv")
        infos[name] = info
        print(name, info)
    retrieval_ist = now_utc.astimezone(IST).date()
    nifty_official = download_niftyindices(
        manifest, dt.date(2007, 9, 1), retrieval_ist - dt.timedelta(days=1))
    nifty_official.to_csv(PROCESSED / "nifty50_niftyindices.csv")
    infos["nifty50_niftyindices"] = {"rows": len(nifty_official),
                                     "first": str(nifty_official.index[0].date()),
                                     "last": str(nifty_official.index[-1].date())}
    print("nifty50_niftyindices", infos["nifty50_niftyindices"])
    manifest.write({"retrieval_utc": now_utc.isoformat(timespec="seconds"),
                    "processed": infos})


if __name__ == "__main__":
    main()
