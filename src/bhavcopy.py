"""Download NSE F&O bhavcopies and keep the NIFTY index futures and options rows.

Source (public NSE archives; plain HTTPS GET with a browser user agent):
  until 2024-07-05  content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
  from  2024-07-08  content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip   (UDiFF)
on https://nsearchives.nseindia.com. If one format returns 404, the other is tried.

For each trading day (taken from data/processed/nifty50.csv), the zip is downloaded
into memory, the NIFTY rows (SYMBOL/TckrSymb == "NIFTY", index futures and
options only) are normalised to one schema and written to
    data/raw/bhavcopy/nifty/YYYY/YYYYMMDD.csv.gz
and one line is appended to data/raw/bhavcopy/MANIFEST.jsonl with the URL,
HTTP status, SHA-256 of the original zip and row counts. The full zips (about
3 GB) are not kept; their hashes identify them. The script is resumable:
days already in the manifest are skipped.

Normalised columns
  date, instrument (FUT | CE | PE), expiry, strike (0 for FUT), open, high, low,
  close, settle, contracts (traded), oi, underlying (UDiFF only, else empty)

Untraded options keep a stale "close" in the bhavcopy; downstream code must use
``contracts > 0`` to decide whether a price is a real trade.

Usage: python -m src.bhavcopy [--start 2010-07-19] [--end 2026-10-21] [--pause 0.8]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .upstream import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "raw" / "bhavcopy"
BASE = "https://nsearchives.nseindia.com"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
UDIFF_START = dt.date(2024, 7, 8)
COLUMNS = ["date", "instrument", "expiry", "strike", "open", "high", "low", "close",
           "settle", "contracts", "oi", "underlying"]


def url_old(day: dt.date) -> str:
    mon = day.strftime("%b").upper()
    return (f"{BASE}/content/historical/DERIVATIVES/{day:%Y}/{mon}/"
            f"fo{day:%d}{mon}{day:%Y}bhav.csv.zip")


def url_udiff(day: dt.date) -> str:
    return f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def _expiry(text: str) -> str:
    """Old-format expiry: '29-Jul-2010', but some 2012 files use '31-May-12'."""
    text = text.strip()
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised expiry date {text!r}")


def _num(text: str) -> float:
    text = (text or "").strip()
    return float(text) if text not in ("", "-") else float("nan")


def parse_rows(payload: bytes, day: dt.date) -> list[dict]:
    """Extract NIFTY index futures/options rows from one bhavcopy zip."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise ValueError(f"expected one CSV in the zip, got {archive.namelist()}")
    text = archive.read(names[0]).decode("utf-8", "replace")
    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = [f.strip() for f in reader.fieldnames]
    rows = []
    if "SYMBOL" in reader.fieldnames:  # pre-July-2024 format
        for r in reader:
            # (r.get(...) or "") skips footer lines such as "*Updated" in some 2012 files.
            if ((r.get("SYMBOL") or "").strip() != "NIFTY"
                    or (r.get("INSTRUMENT") or "").strip() not in ("FUTIDX", "OPTIDX")):
                continue
            inst = "FUT" if r["INSTRUMENT"].strip() == "FUTIDX" else r["OPTION_TYP"].strip()
            rows.append({
                "date": day.isoformat(), "instrument": inst,
                "expiry": _expiry(r["EXPIRY_DT"]),
                "strike": _num(r["STRIKE_PR"]), "open": _num(r["OPEN"]), "high": _num(r["HIGH"]),
                "low": _num(r["LOW"]), "close": _num(r["CLOSE"]), "settle": _num(r["SETTLE_PR"]),
                "contracts": _num(r["CONTRACTS"]), "oi": _num(r["OPEN_INT"]), "underlying": float("nan"),
            })
    elif "TckrSymb" in reader.fieldnames:  # UDiFF
        for r in reader:
            if ((r.get("TckrSymb") or "").strip() != "NIFTY"
                    or (r.get("FinInstrmTp") or "").strip() not in ("IDF", "IDO")):
                continue
            inst = "FUT" if r["FinInstrmTp"].strip() == "IDF" else r["OptnTp"].strip()
            rows.append({
                "date": day.isoformat(), "instrument": inst, "expiry": r["XpryDt"].strip(),
                "strike": _num(r["StrkPric"]) if inst != "FUT" else 0.0,
                "open": _num(r["OpnPric"]), "high": _num(r["HghPric"]), "low": _num(r["LwPric"]),
                "close": _num(r["ClsPric"]), "settle": _num(r["SttlmPric"]),
                "contracts": _num(r["TtlTradgVol"]), "oi": _num(r["OpnIntrst"]),
                "underlying": _num(r["UndrlygPric"]),
            })
    else:
        raise ValueError(f"unknown bhavcopy header: {reader.fieldnames[:6]}")
    bad = [r for r in rows if r["instrument"] not in ("FUT", "CE", "PE")]
    if bad:
        raise ValueError(f"unexpected instrument codes, e.g. {bad[0]}")
    return rows


def done_days() -> set[str]:
    manifest = OUT / "MANIFEST.jsonl"
    if not manifest.exists():
        return set()
    # Days that ended in an error are retried on the next run.
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return {r["date"] for r in records if "file" in r or r.get("missing")}


def process_day(day: dt.date) -> dict:
    urls = [url_udiff(day), url_old(day)] if day >= UDIFF_START else [url_old(day), url_udiff(day)]
    record = {"date": day.isoformat(), "attempts": []}
    for url in urls:
        status, payload = fetch(url)
        record["attempts"].append({"url": url, "status": status})
        if status == 200 and payload:
            rows = parse_rows(payload, day)
            path = OUT / "nifty" / f"{day:%Y}" / f"{day:%Y%m%d}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            record.update({
                "url": url, "zip_sha256": hashlib.sha256(payload).hexdigest(),
                "zip_bytes": len(payload), "nifty_rows": len(rows),
                "futures": sum(r["instrument"] == "FUT" for r in rows),
                "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            })
            return record
    record["missing"] = True
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-07-19")
    parser.add_argument("--end", default="2026-10-21")
    parser.add_argument("--pause", type=float, default=0.8, help="seconds between days")
    args = parser.parse_args(argv)
    nifty = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv", index_col=0,
                        parse_dates=True)
    days = [d.date() for d in nifty.index
            if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    skip = done_days()
    todo = [d for d in days if d.isoformat() not in skip]
    print(f"{len(days)} trading days, {len(skip)} already done, {len(todo)} to fetch", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (OUT / "MANIFEST.jsonl").open("a") as manifest:
        for i, day in enumerate(todo, 1):
            for attempt in range(3):  # retry transient network errors
                try:
                    record = process_day(day)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                    if attempt == 2:
                        record = {"date": day.isoformat(), "error": repr(error)}
                    time.sleep(5 * (attempt + 1))
                except (ValueError, KeyError, zipfile.BadZipFile) as error:
                    # Unexpected file content: log it and move on (retried next run).
                    record = {"date": day.isoformat(), "error": f"parse: {error!r}"}
                    break
            record["retrieved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                print(f"{i}/{len(todo)} {day} ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
