# Real-data backtest log (synthetic ATM NIFTY calls)

Reproduce (about 1 min in total):

```
python -m src.backtest_real                    # 21-day options -> results/backtest_real/maturity_21/
python -m src.backtest_real --maturity 63      # 3-month options
python -m src.backtest_analysis results/backtest_real/maturity_21
python -m src.backtest_analysis results/backtest_real/maturity_21 --exclude-test 2020-01-15 2020-06-30 --tag excl_covid
python -m src.backtest_analysis results/backtest_real/maturity_21 --block-mult 3 --tag block3x
python -m src.backtest_analysis results/backtest_real/maturity_63
python -m pytest tests -q                      # tests/test_backtest_real.py: 8 checks
```

## 1. Design

- **Data.** The verified NIFTY 50 and India VIX series and the train-calibrated
  regime labels (`notes/data_log.md`). Train covers 2010-07-20 .. 2018-12-31 and
  test covers 2019-01-01 .. 2026-10-21.
- **Options.** One ATM European call is sold at every close, with 21 trading
  days to expiry; a 63-day variant is a robustness check. The whole life of each
  option lies inside one period: 2,076 train and 1,914 test options at 21 days.
- **Premium and hedge.** Black-Scholes at implied vol = India VIX/100: VIX at the
  start for the premium, VIX at each rebalancing close for the delta. Rates and
  dividends are set to zero, and the index itself is the hedge (see limitations).
  Prices are divided by S(t0), so P&L is per unit of notional, in bp.
- **Rules** (reusing `src/rebalancing.py`):
  - fixed every m in {1,2,3,4,5,7,10,14,21} days;
  - regime-aware: every m_stress days in stress, m_calm in calm, with 12
    (m_calm, m_stress) pairs and 4 label methods;
  - no-trade bands with h in {0.02, 0.05, 0.1, 0.2}.

  Decisions at the close of t use only data up to t; this is tested.
- **Placebos.** The same regime rules driven by inverted labels (`flip:`) or by
  labels from 252 trading days earlier (`lag252:`, causal but stale).
- **Costs.** kappa in {0, 1, 2, 5, 10, 20} bp of traded notional, including the
  initial trade and the final liquidation.
- **Criterion: J = E[(e - m0)^2] = Var(e) + (E[cost])^2.** m0 is the rule's own
  mean P&L at zero cost. On the test period the BS-at-VIX seller earns about
  +30 bp per option on average (about 15% of the premium). That mean is a risk
  premium, not hedging quality, so the raw E[e^2] was misleading: it *fell* as
  costs rose, because costs pulled the mean toward zero. J equals the
  simulation's E[e^2] when the premium is fair. The raw E[e^2] is kept in
  `rules_test.csv`.
- **Inference.** Options overlap (one per day, 21-day life), so every standard
  error is a moving-block bootstrap over start dates (block = 2 x maturity; 3 x
  as a check), applied to paired per-option differences.

## 2. Checks (`tests/test_backtest_real.py`, 8 pass)

- BS delta equals the finite-difference derivative of the BS price.
- One option's P&L matches an independent scalar loop to 1e-12.
- Option windows never cross the train/test boundary.
- Positions up to step k do not change when prices, vols or labels after k change.
- The lag placebo is causal (label from t-252), and the flip placebo is 1 - label.
- The block bootstrap equals the i.i.d. formula on i.i.d. data (within 10%) and
  inflates by about 3x for AR(1) with phi = 0.8, matching the theoretical
  sqrt((1+phi)/(1-phi)) = 3.
- End to end in a Black-Scholes world (GBM, VIX = true vol): the hedged seller
  breaks even on average, and halving the rebalancing frequency roughly doubles
  the error variance.

## 3. Results (test period; J-RMSE in bp of notional; mean premium 197 bp)

**(a) At realistic NIFTY cost levels, daily rebalancing is best.** For every
kappa from 0 to 20 bp, the best fixed rule on TRAIN is "every day". The best
regime rule on train (main label `vix_hyst`) is "daily in stress, every 2
days in calm". On TEST it is **worse** than daily fixed rebalancing:

| kappa | J change, regime vs daily (z) | J change, band vs daily (z) |
|---|---|---|
| 0 bp | +20.4% (+2.9) | +0.6% (+1.2) |
| 5 bp | +19.5% (+2.9) | +0.6% (+1.0) |
| 20 bp | +7.9% (+1.6) | -1.6% (-0.8) |

The same holds for 63-day options (+8.3%, z +3.6 at 0 bp) and when the COVID
window is excluded (+30.0%, z +2.7 at 5 bp). Costs are small relative to the
hedging error: J-RMSE of daily hedging is 64.0 bp at 0 bp cost and 79.6 bp at
20 bp.

**(b) At equal turnover, the apparent gain from regime-aware rules is driven by
COVID 2020.** Each regime rule is compared with the equal-turnover mixture of
its two neighbouring fixed rules (paired, no selection). The table counts the 48
real-label rules (4 labels x 12 pairs) at kappa = 5 bp:

| Variant | real rule beats fixed | median z | real beats inverted placebo | real beats lag-252 placebo |
|---|---|---|---|---|
| 21-day, full test | 42/48 | -1.1 | 46/48 | 46/48 |
| 21-day, excl. options starting 2020-01-15 .. 2020-06-30 | **20/48** | **+0.2** | 34/48 | 33/48 |
| 21-day, bootstrap block 3x | 42/48 | -1.0 | 46/48 | 46/48 |
| 63-day options | **19/48** | **+0.3** | 35/48 | 40/48 |

(Median of the 48 z values; for an even count, the mean of the two middle values.)

Individual z values are small: in the full test, only 1 of 48 has z < -2 (min
-2.2). The point estimates for the low-turnover rules are large (up to about
-39% J), but they come from one episode.

**(c) What the labels do show.** Rules driven by real labels beat the same
rules driven by inverted or stale labels in most comparisons in every variant
(33-46 of 48). Trading *less* in stress is clearly harmful: in the full test,
for rules with m_calm >= 5 the inverted placebo adds +31% to +262% to J relative
to equal-turnover fixed hedging. For rules that trade every 2-3 days in calm the
placebo penalty vanishes. Trading *more* in stress does not reliably beat a
fixed schedule.

**(d) Bands.** Bands picked on train are close to daily fixed rebalancing on test:
- At 0-5 bp: +0.6% to +1.0% J, i.e. slightly worse.
- At 20 bp: -1.6% to -4.7% J across the variants, i.e. slightly better.

The largest |z| is 1.7, so none of these differences is significant. At equal
turnover, the widest band (h = 0.2) beats fixed rules for 21-day options:
-40% to -45% J (z -2.3), and still -37% (z -4.1) without COVID. This is the
most robust "smart rule" result in the backtest, unlike the regime rules. It is
weaker for 63-day options (-9%, z -0.9). It applies only at low turnover,
where J is well above the daily-rebalancing level (79 vs 65 bp at 5 bp).

## 4. Interpretation (for the report)

- The real data agree with the simulation (`notes/robustness_log.md`, F3):
  adapting the timing to the regime helps mainly when turnover must be limited.
  At the cost-optimal point, where trading is frequent, the rules converge. On
  NIFTY with costs up to 20 bp, the cost-optimal point is daily rebalancing.
- The real data add one caution: the low-turnover gains are **episodic**
  (COVID). A single stress episode can dominate an 8-year backtest, so report
  the result with and without it.
- This is not evidence that regime-aware hedging is useless. It is evidence
  that, with proportional costs this small, there is little to gain. Fixed
  per-trade costs, wider spreads in stress, or a turnover budget could change
  this; they are not modelled.

## 5. Limitations

- Synthetic options: premium and delta use BS at India VIX for every strike and
  maturity (no smile, no term structure). Real NIFTY options (plan option b)
  would fix this.
- Zero rates and dividends; the hedge is the index, not futures (no basis or
  roll). This affects the mean P&L more than the hedging-error variance that we
  rank by.
- Daily closes only. NIFTY and VIX closes are taken as synchronous, and there
  is no intraday rebalancing.
- Proportional costs only, constant across regimes. In reality spreads widen in
  stress, which would penalise trading in stress.
- The test period is about 7.8 years with one dominant crisis (2020), so
  effective sample sizes are small.
- The 48 regime rules are not independent, so the counts in 3(b) are
  descriptive, not a formal test.

## 6. Implied vs realised volatility (gamma P&L check)

Why: a delta hedge run at implied vol sigma_i earns on average
(1/2) Gamma S^2 (sigma_i^2 - sigma_r^2) dt. The seller's positive mean P&L
(+30 bp synthetic, +18 bp traded) is explained only if implied exceeded
realised. The check is computed in `src/report_assets.py`
(`implied_vs_realised`) on the test period 2019-2026:

| Comparison | implied > realised | mean implied | mean realised | variance ratio, excl. COVID | variance ratio, COVID window |
|---|---|---|---|---|---|
| India VIX at t vs realised vol over t+1..t+21 | 79% of days | 17.1% | 14.7% | 1.36 | 0.73 |
| Traded option IV at sale vs realised vol over its life | 73% of options | 15.7% | 14.5% | 1.20 | 0.68 |

- The variance ratio is mean(sigma_i^2) / mean(sigma_r^2). The COVID window is
  2020-01-15 .. 2020-06-30, the same as in the backtests (by start date for
  options, by date for VIX).
- Reading: implied usually exceeded realised, so gamma P&L explains the
  seller's average gain. In the COVID crash realised volatility was higher
  (ratio < 1). Over all traded options, mean implied and realised variance are
  about equal (0.0302 vs 0.0305), so the premium shows up in typical periods
  rather than in the overall average of variances.
