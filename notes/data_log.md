# Data log: NIFTY 50 and India VIX (daily)

Everything here can be reproduced with:

```
python -m src.data_download   # raw files + MANIFEST (Yahoo, niftyindices)
python -m src.data_prepare    # verified series + results/data_checks/report.json
python -m src.regimes         # regime labels + results/regimes/
python -m pytest tests -q     # includes tests/test_regimes.py (no look-ahead)
```

The NSE India VIX cross-check (`data/raw/nse/`) was fetched through a browser,
because nseindia.com rejects scripted requests (HTTP 403). It is saved as text
files and is not re-downloaded by the scripts.

## 1. Sources

| Series | Source | Coverage | Role |
|---|---|---|---|
| NIFTY 50 close (OHLC) | NSE Indices Ltd, niftyindices.com historical data (`/BackPage/getHistoricaldatatabletoString`, one year per request) | 2007-09-03 .. 2026-10-21, 4,740 days | **Primary** (official index provider) |
| NIFTY 50 | Yahoo Finance chart API, `^NSEI` | 2007-09-17 .. 2026-10-19 | Cross-check only |
| India VIX | Yahoo Finance chart API, `^INDIAVIX` | 2008-03-03 .. 2026-10-19 | Base series |
| India VIX | NSE, `nseindia.com/api/historicalOR/vixhistory` | 2010-07-19 .. 2026-10-21 (4,032 days) | Verification and patches |

Raw responses, URLs, retrieval times and SHA-256 hashes are in
`data/raw/MANIFEST.json`. The browser-fetched NSE files carry their provenance in
their headers.

## 2. What the checks found (`results/data_checks/report.json`)

**NIFTY 50**
- Yahoo matches the official closes on all 4,683 common days, with a maximum
  relative difference of 4.8e-8 (float rounding). Yahoo is therefore a copy of the
  official series, but it **lacks 47 trading days**, including the latest two.
- The official series includes 20 weekend special sessions: Budget days, Diwali
  Muhurat sessions and the 2024 disaster-recovery test days. They are kept and
  flagged `special_session`.
- There are no duplicates, missing closes or OHLC inconsistencies. The only
  calendar gap above 5 days is 1-7 Oct 2014, which consists of holidays.
- There are 21 daily moves above 6% in absolute log return: 14 in Jan 2008 -
  Jul 2009 (largest +16.3% on 2009-05-18 and -13.0% on 2008-10-24), plus
  2015-08-24, five in Mar-Apr 2020 (largest -13.9% on 2020-03-23) and 2024-06-04.
  These match known market events, and they were left as they are. Yahoo
  confirms 20 of the 21; it lacks 2008-10-28, which is probably the Diwali
  Muhurat session (not verified).

**India VIX**
- Yahoo lacks 49 days (42 of which NSE has) and rounds to 2 decimals (NSE gives
  4). The rounding is harmless (at most 0.005).
- Four days had real disagreements. Each was settled with NSE's next-day
  "previous close" field (`data/raw/nse/vix_reconciliation_evidence.txt`):
  - 2013-08-22 and 2018-11-07: **Yahoo was wrong** and has been replaced by NSE.
  - 2021-02-12 and 2021-03-30: **NSE's rows are placeholders** (O=H=L=0, close
    carried forward), so Yahoo's values were kept.
- After patching, all 196 months from July 2010 match NSE's monthly count, sum,
  min and max, apart from the two placeholder months. The verified series has
  exactly NSE's 4,032 days.
- **2008-03-03 .. 2010-07-16 is Yahoo-only and unverified** (NSE's API has
  nothing earlier). It is flagged `verified = False`. The main analysis starts on
  2010-07-19.
- 2024-03-02 (a NIFTY test session) has no VIX. The previous close is carried
  forward in the regime panel (causal).

## 3. Regime labels (`results/regimes/`)

Split: train = 2010-07-20 .. 2018-12-31 (2,097 days), test = 2019-01-01 ..
2026-10-21 (1,935 days). Every threshold and parameter comes from the train
period. Each label dated t uses data up to the close of t.

| Method | Rule | Train calibration |
|---|---|---|
| `vix_q75` | VIX > train q75 | 20.16 |
| `vix_hyst` | enter above q75, leave below q60 | 20.16 / 17.88 |
| `rv21_q75` | 21-day trailing realised vol > train q75 | 17.4% annualised |
| `ms_filter` | 2-state Markov-switching, filtered P(stress) > 0.5 | daily sigma 0.71% calm vs 1.23% stress; expected durations 87 / 64 days; all 8 random starts reach the same likelihood |

**Out-of-sample validation (test period).** For every method, a stress label at
t is followed by higher realised volatility over (t, t+5]: 1.7-2.0 times the calm
level. The moving-block bootstrap 95% CIs (20-day blocks) all exclude 1
(`summary.json`, `fwd_rv5_stress_over_calm`). The same holds over 21 days
(1.5-1.75x). The methods agree on 81-94% of test days.

**Behaviour.**
- `vix_hyst` is the most stable rule: 13 test-period stress spells with a median
  of 16 days. Spells include Apr-May 2019, COVID (2020-02-28 .. 2021-05-27),
  most of Feb-Jul 2022, 13 May - 5 Jun 2024 and 9 Mar - 16 Apr 2026 (full
  list in `summary.json`).
- `ms_filter` flickers (median spell 3-6.5 days) because its filtered
  probability often sits near 0.5. Use it as a robustness check, not as the main
  label.
- `vix_q75` without hysteresis also flickers (median spell 3 days in test).

**Sensitivity (`results/regimes_robustness/`).** Out-of-sample ratio of forward
5-day realised vol, stress vs calm, with block-bootstrap 95% CI:

| Variant | VIX q75 | vix_q75 | vix_hyst | rv21_q75 | ms_filter |
|---|---|---|---|---|---|
| main (train to 2018) | 20.16 | 2.00 [1.57, 2.58] | 1.94 [1.58, 2.44] | 1.79 [1.37, 2.41] | 1.74 [1.40, 2.19] |
| train to 2017 | 20.77 | 2.07 [1.59, 2.73] | 1.94 [1.57, 2.46] | 1.80 [1.37, 2.38] | 1.71 [1.40, 2.16] |
| train to 2019 | 19.81 | 2.03 [1.61, 2.66] | 2.01 [1.65, 2.54] | 1.85 [1.40, 2.47] | 1.79 [1.43, 2.28] |
| start 2008 (unverified VIX) | 24.91 | 2.65 [1.71, 4.01] | 2.25 [1.52, 3.35] | 1.86 [1.32, 2.64] | 2.28 [1.45, 3.34] |

- The conclusions do not depend on the split date.
- Including 2008 puts the global financial crisis into training, which raises
  the thresholds, so stress is rarer and the ratios are larger but noisier.

**Tests (`tests/test_regimes.py`, 8 pass).**
- Truncation invariance: the label at t is the same when the data stops at t.
- Scrambling the data after t leaves every label up to t unchanged.
- Calibration is unaffected by test-period data.
- The Hamilton filter equals a brute-force sum over all 2^T state paths (to 1e-12).
- The MLE recovers known parameters on simulated data.

## 4. Limitations (state these in the report)

- Daily closes only. NIFTY and VIX closes are both end-of-day, but intraday
  timing is not modelled.
- India VIX is a forward-looking implied measure. Its use as a regime signal is
  causal, because only the close at t is used at t.
- Before 2010-07-19, VIX cannot be verified against NSE.
- The NSE pages and APIs are undocumented and may change; the
  `data/raw/` snapshot is what the results use. Check the NSE and niftyindices
  terms of use before redistributing the data. We use it for academic purposes
  and cite the sources.
