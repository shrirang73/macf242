# Traded-option backtest log (NSE F&O bhavcopy, plan option b)

Reproduce:

```
python -m src.bhavcopy            # ~1.5 h, resumable; NIFTY rows only -> data/raw/bhavcopy/
python -m src.fo_data             # data checks -> results/data_checks/fo_report.json
python -m src.backtest_options --vol own
python -m src.backtest_options --vol vix
python -m src.backtest_analysis results/backtest_options/own
python -m src.backtest_analysis results/backtest_options/own --exclude-test 2020-01-15 2020-06-30 --tag excl_covid
python -m src.backtest_analysis results/backtest_options/own --block-mult 3 --tag block3x
python -m src.backtest_analysis results/backtest_options/vix
python -m pytest tests/test_backtest_options.py -q      # 6 checks
```

## 1. Data

- **Source.** NSE archives at nsearchives.nseindia.com:
  - old format `content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip` until 2024-07-05;
  - UDiFF `content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip` from 2024-07-08.

  Downloaded with plain GETs at about 1 request/s.
- **What is kept.** Only the NIFTY index futures and options rows, normalised to
  one schema (`src/bhavcopy.py`). Each day's original zip is identified by its
  SHA-256 in `data/raw/bhavcopy/MANIFEST.jsonl`; the zips themselves (about 3 GB)
  are not kept.
- **Coverage.** 4,031 of 4,033 NIFTY trading days in 2010-07-19 .. 2026-10-21.
  NSE returns 404 in both formats for 2013-10-09 and 2021-03-30. On 2021-03-30
  NSE's India VIX row is also a placeholder. All 3 monthly futures are present on
  every downloaded day.
- **Format quirks handled.**
  - Some 2012 files use 2-digit expiry years (`31-May-12`) and end with an
    `*Updated` footer line.
  - Untraded contracts carry a stale close, so a price counts only if
    `contracts > 0`.
  - In UDiFF the option settlement price differs from the close, so we use the
    close of traded options.
- **Monthly contracts** = options whose expiry is also a futures expiry. This is
  robust to weekly options and to the later change of expiry weekday.
- **Moved and relabelled expiries** (`Market` in `src/backtest_options.py`, tested):
  - The label 2014-04-24 expired and settled on 2014-04-23 (holiday).
  - The labels 2014-02-27 and 2023-06-29 were replaced by 2014-02-26 and
    2023-06-28 during the contracts' lives. We merge them into one contract:
    label L becomes L2 when L2 first appears on the trading day after L's last
    appearance and lies within 7 days of L.
  - Rule for the real expiry: the label if it is a trading day; otherwise the
    last appearance, provided the future settled there at the index close.

## 2. Data checks (`results/data_checks/fo_report.json`)

| Check | Result |
|---|---|
| Expiring future settles at the NIFTY close on expiry day | 194/194 expiries, relative difference exactly 0 |
| Near-month basis F/S - 1 | 1-99%: -0.23% .. +0.75%; none outside [-1%, +2%] |
| Put-call parity C - P vs F - K, near ATM (within 3%), traded pairs | 136,118 pairs; median +0.3 bp; 1-99%: +-28 bp; 0.6% beyond 50 bp |
| Traded calls below exp(-0.1 tau)(F-K)+ minus 5 bp | 32,462 of 510,990, almost all deep ITM (K/F < 0.97); near ATM 401 of 144,996 (0.28%) |
| ATM implied vol (near month) vs India VIX | correlation 0.956; ATM IV is about 0.90 x VIX (VIX also prices the skew) |

## 3. Backtest design

- At each close, sell the monthly call whose expiry is closest to 21 trading days
  away (at least 10), at the listed strike nearest the same-expiry futures close
  F(t0). The call must have traded at t0 and its price must lie inside the bounds.
- Hedge with the same-expiry future at the Black-76 delta. Hedge P&L is
  sum phi_k (F_{k+1} - F_k). At expiry the future settles at the index close, and
  the call pays (F_E - K)^+ at the same price.
- Delta vol:
  - `own`: the option's implied vol from its close that day. If the option did
    not trade, use the ATM IV of the same expiry, else the previous value. The
    fallback is needed on 6.2% (train) and 2.6% (test) of option-days.
  - `vix`: India VIX/100.
- P&L per unit of F(t0). Costs apply to futures turnover only (the option trade
  is the same for every rule). Margin financing and discounting are ignored.
- Rules, labels, placebos, criterion J and the block bootstrap are as in the
  synthetic backtest (`notes/backtest_log.md`).
- **Sample.** Train has 2,067 options (2010-07-20 .. 2018-12-12 starts) and test
  has 1,898 (2019-01-01 .. 2026-09-15). Option lives are 10-34 trading days
  (median 21), and strikes lie within +-1% of F(t0).
- **Exclusions.**
  - Paths crossing a missing bhavcopy day: 17 train, 12 test.
  - Expiry after the period end: 12 train.
  - No expiry at least 10 days away near the end of the data: 24 test.
  - Start day with no bhavcopy: 1 in each period.

## 4. Results (test period, `results/backtest_options/`)

- **Level.** Daily hedging gives J-RMSE = 57.7 bp of notional against 64.0 bp in
  the synthetic backtest. The cost-free mean P&L is +18 bp per option, and the
  mean premium is 179 bp.
- **Own IV vs VIX delta** (paired, block bootstrap):
  - daily rebalancing at 0 bp: J -4.3% (z -1.5);
  - the other combinations tested (every 1/2/5 days at 0/5/20 bp): between
    -3.5% and +1.0% (|z| <= 1.2).

  Not significant.
- **Train-to-test.** The best fixed rule on train is "every day" at every kappa.
  The train-picked regime rule ("daily in stress, every 2 days in calm") is
  worse on test:

| Variant | kappa 5 bp | kappa 20 bp | equal turnover: beat fixed | median z | beat inverted placebo |
|---|---|---|---|---|---|
| Own IV | +35.0% (z 2.1) | +15.1% (z 1.4) | 48/48 | -1.0 | 48/48 |
| Own IV, excl. COVID window | +51.8% (z 2.0) | +18.8% (z 1.3) | 17/48 | +0.4 | 41/48 |
| Own IV, block 3x | +35.0% (z 2.0) | +15.1% (z 1.3) | 48/48 | -0.9 | 48/48 |
| VIX delta | +33.7% (z 2.1) | +16.3% (z 1.5) | 47/48 | -1.0 | 46/48 |

- **Bands** (train-picked) are about equal to daily rebalancing: +0.5% to +0.7% J
  at 0-5 bp and -0.1% at 20 bp. Without COVID they are -4.2% at 20 bp (z -1.7),
  which is not significant.
- **Reading.** The same as the synthetic backtest, a bit sharper:
  - Regime-aware rules beat fixed rules at equal turnover only because of the
    COVID crash.
  - Real labels beat inverted or stale labels, so trading less in stress hurts.
  - At NIFTY cost levels, daily rebalancing wins.

## 5. Limitations

- Option closes and futures closes are not synchronous (option close is close
  to the last traded price). This adds noise to the implied vol and the start
  premium. It mostly affects the level of J, not the paired comparisons between
  rules.
- No bid-ask spread on the option sale (it is common to all rules). Futures
  costs are proportional and do not depend on the regime.
- Black-76 with zero discounting; margin financing ignored.
- One option per start day: lives overlap, so the effective sample is about 90
  months.
