# Model comparison from saved results (no new runs)

All hedging numbers: NIFTY 1-month ATM calls, test period 2019-2026, rules picked on 2010-2018 per hedge, bp of notional. Main setting: daily rebalancing, 5 bp cost. Bootstrap: paired moving-block bootstrap, block 42 option-days, 1000 resamples; 95% intervals; **bold** = interval excludes 0 (difference is real).

# Part A - Hedging

## A1. Summary table (daily, 5 bp)

### Test period with COVID (1914 options)

| Hedge | RMSE | Variance reduction vs no hedge | VaR95 | VaR99 | ES95 | ES99 | Worst | Mean of 5 worst | Worst option starts |
|---|---|---|---|---|---|---|---|---|---|
| Black-Scholes | 64.8 | 94.2% | 75.0 | 195.9 | 170.7 | 313.1 | 511.8 | 442.6 | 2020-03-11 |
| Heston delta (fixed) | 64.9 | 94.2% | 83.4 | 196.9 | 167.0 | 293.6 | 486.0 | 415.2 | 2020-03-11 |
| Heston min-var (fixed) | 62.2 | 94.7% | 78.0 | 197.3 | 165.2 | 278.2 | 449.5 | 382.9 | 2020-03-09 |
| Heston delta, 12-month | 65.0 | 94.2% | 86.9 | 194.6 | 168.3 | 287.1 | 472.1 | 401.4 | 2020-03-09 |
| Heston min-var, 12-month | 62.3 | 94.7% | 84.7 | 195.4 | 166.6 | 275.1 | 443.9 | 378.7 | 2020-03-09 |
| Heston delta, 6-month | 65.4 | 94.1% | 85.6 | 195.9 | 169.2 | 293.1 | 492.3 | 421.3 | 2020-03-11 |
| Heston min-var, 6-month | 62.3 | 94.7% | 85.6 | 195.7 | 166.7 | 279.8 | 459.5 | 389.9 | 2020-03-09 |
| Heston delta, expanding | 65.0 | 94.2% | 82.1 | 196.9 | 166.4 | 293.1 | 485.1 | 414.4 | 2020-03-11 |
| Heston min-var, expanding | 62.1 | 94.7% | 74.9 | 197.1 | 164.7 | 278.3 | 449.6 | 383.1 | 2020-03-09 |
| alpha=1 delta | 67.4 | 93.8% | 82.4 | 205.9 | 171.7 | 321.2 | 536.6 | 462.7 | 2020-03-11 |
| alpha=1 min-var | 62.8 | 94.6% | 73.9 | 205.3 | 168.6 | 293.9 | 479.5 | 406.6 | 2020-03-11 |
| Rough delta | 69.3 | 93.4% | 87.0 | 218.5 | 173.0 | 325.1 | 548.1 | 471.0 | 2020-03-11 |
| Rough min-var | 63.1 | 94.5% | 74.8 | 216.2 | 171.1 | 293.2 | 471.8 | 402.3 | 2020-03-11 |

### Test period without COVID (1801 options)

| Hedge | RMSE | Variance reduction vs no hedge | VaR95 | VaR99 | ES95 | ES99 | Worst | Mean of 5 worst | Worst option starts |
|---|---|---|---|---|---|---|---|---|---|
| Black-Scholes | 52.3 | 95.9% | 64.2 | 174.9 | 133.9 | 213.5 | 295.9 | 275.2 | 2024-05-07 |
| Heston delta (fixed) | 53.6 | 95.6% | 66.8 | 168.2 | 135.2 | 210.1 | 298.0 | 273.9 | 2019-08-21 |
| Heston min-var (fixed) | 52.8 | 95.8% | 64.4 | 183.0 | 139.4 | 219.6 | 300.3 | 278.5 | 2024-05-07 |
| Heston delta, 12-month | 54.3 | 95.5% | 68.4 | 174.8 | 139.4 | 215.9 | 298.0 | 278.1 | 2019-08-21 |
| Heston min-var, 12-month | 53.2 | 95.7% | 68.5 | 179.6 | 142.0 | 218.7 | 298.9 | 278.1 | 2024-05-07 |
| Heston delta, 6-month | 54.5 | 95.5% | 69.9 | 178.1 | 141.3 | 217.3 | 298.7 | 278.8 | 2019-08-21 |
| Heston min-var, 6-month | 53.3 | 95.7% | 67.4 | 180.5 | 142.4 | 219.5 | 299.4 | 278.8 | 2024-05-07 |
| Heston delta, expanding | 53.6 | 95.7% | 65.8 | 166.5 | 134.5 | 209.4 | 298.0 | 273.5 | 2019-08-21 |
| Heston min-var, expanding | 52.7 | 95.8% | 63.9 | 182.5 | 138.6 | 219.6 | 300.5 | 278.5 | 2024-05-07 |
| alpha=1 delta | 54.2 | 95.6% | 64.8 | 164.6 | 134.0 | 211.6 | 306.3 | 274.6 | 2019-08-21 |
| alpha=1 min-var | 52.9 | 95.8% | 64.2 | 188.6 | 140.8 | 225.6 | 305.1 | 283.2 | 2019-08-21 |
| Rough delta | 56.1 | 95.2% | 73.6 | 168.0 | 137.0 | 216.7 | 333.3 | 274.2 | 2019-08-21 |
| Rough min-var | 54.3 | 95.5% | 66.9 | 196.7 | 147.0 | 235.9 | 326.5 | 294.4 | 2019-08-21 |

## A2. Are the differences real? (paired block bootstrap, daily, 5 bp)

Difference = hedge minus reference, bp. Negative = hedge better (lower error / smaller tail loss).

### Reference: Black-Scholes, test with COVID

| Hedge | RMSE difference [95% CI] | ES95 difference [95% CI] | ES99 difference [95% CI] |
|---|---|---|---|
| Heston delta (fixed) | +0.1 [-2.1, +2.1] | -3.7 [-14.8, +4.7] | -19.5 [-34.6, +0.8] |
| Heston min-var (fixed) | -2.6 [-7.3, +1.2] | -5.6 [-26.9, +7.7] | -34.9 [-70.2, +7.1] |
| Heston delta, 12-month | +0.1 [-2.9, +3.0] | -2.4 [-18.1, +8.8] | -26.0 [-49.1, +4.7] |
| Heston min-var, 12-month | -2.6 [-7.3, +1.6] | -4.2 [-27.0, +11.2] | -38.0 [-74.9, +7.7] |
| Heston delta, 6-month | +0.6 [-2.3, +3.4] | -1.5 [-17.7, +10.7] | -20.0 [-34.9, +7.1] |
| Heston min-var, 6-month | -2.5 [-7.0, +1.5] | -4.0 [-27.0, +11.3] | -33.3 [-63.2, +8.7] |
| Heston delta, expanding | +0.1 [-2.1, +2.2] | -4.4 [-15.6, +4.5] | -20.0 [-35.4, +0.7] |
| Heston min-var, expanding | -2.7 [-7.4, +1.1] | -6.0 [-26.7, +7.4] | -34.8 [-69.9, +7.3] |
| alpha=1 delta | **+2.5 [+0.9, +4.0]** | +0.9 [-5.8, +8.8] | +8.1 [-16.0, +19.0] |
| alpha=1 min-var | -2.0 [-5.9, +1.6] | -2.1 [-19.7, +12.2] | -19.2 [-44.9, +15.1] |
| Rough delta | **+4.5 [+2.1, +6.8]** | +2.2 [-8.7, +15.6] | +12.0 [-23.0, +26.5] |
| Rough min-var | -1.7 [-7.5, +3.4] | +0.4 [-25.2, +21.4] | -19.9 [-56.9, +30.8] |

### Reference: Black-Scholes, test without COVID

| Hedge | RMSE difference [95% CI] | ES95 difference [95% CI] | ES99 difference [95% CI] |
|---|---|---|---|
| Heston delta (fixed) | **+1.3 [+0.4, +2.4]** | +1.3 [-4.8, +6.2] | -3.4 [-8.8, +3.0] |
| Heston min-var (fixed) | +0.5 [-0.9, +1.8] | +5.5 [-0.7, +9.5] | **+6.1 [+1.4, +8.4]** |
| Heston delta, 12-month | **+2.1 [+1.0, +3.1]** | +5.5 [-1.5, +10.2] | +2.4 [-5.0, +8.3] |
| Heston min-var, 12-month | +0.9 [-0.5, +2.4] | **+8.2 [+0.5, +12.8]** | +5.2 [-0.2, +9.8] |
| Heston delta, 6-month | **+2.3 [+1.0, +3.4]** | +7.4 [-0.6, +12.1] | +3.8 [-2.6, +10.8] |
| Heston min-var, 6-month | +1.0 [-0.4, +2.3] | **+8.6 [+2.2, +13.5]** | **+6.0 [+1.2, +11.0]** |
| Heston delta, expanding | **+1.3 [+0.2, +2.5]** | +0.6 [-5.6, +6.4] | -4.0 [-10.2, +2.6] |
| Heston min-var, expanding | +0.4 [-1.2, +1.8] | +4.8 [-2.1, +9.0] | **+6.1 [+0.9, +8.6]** |
| alpha=1 delta | **+1.9 [+0.1, +3.7]** | +0.2 [-8.7, +8.6] | -1.8 [-17.7, +8.1] |
| alpha=1 min-var | +0.6 [-1.7, +2.6] | +6.9 [-2.0, +15.7] | **+12.1 [+1.5, +17.5]** |
| Rough delta | **+3.8 [+0.9, +6.8]** | +3.1 [-12.5, +17.1] | +3.2 [-26.7, +22.4] |
| Rough min-var | +2.0 [-1.3, +4.9] | **+13.2 [+0.1, +27.0]** | **+22.5 [+6.0, +32.6]** |

### Reference: Heston min-var (fixed), test with COVID

| Hedge | RMSE difference [95% CI] | ES95 difference [95% CI] | ES99 difference [95% CI] |
|---|---|---|---|
| Black-Scholes | +2.6 [-1.2, +7.3] | +5.6 [-7.7, +26.9] | +34.9 [-7.1, +70.2] |
| Heston delta (fixed) | +2.7 [-0.2, +5.9] | +1.9 [-9.8, +15.9] | +15.4 [-14.8, +37.6] |
| Heston delta, 12-month | **+2.8 [+0.5, +4.9]** | +3.2 [-6.2, +12.5] | +8.9 [-8.5, +24.3] |
| Heston min-var, 12-month | +0.1 [-0.6, +0.9] | +1.4 [-2.6, +5.3] | -3.1 [-7.3, +2.8] |
| Heston delta, 6-month | **+3.3 [+0.7, +5.6]** | +4.0 [-5.1, +12.8] | +14.9 [-6.9, +42.5] |
| Heston min-var, 6-month | +0.1 [-0.8, +1.1] | +1.6 [-2.4, +5.5] | +1.6 [-3.8, +11.7] |
| Heston delta, expanding | +2.8 [-0.4, +6.0] | +1.2 [-10.5, +14.8] | +14.9 [-16.4, +36.9] |
| Heston min-var, expanding | -0.1 [-0.3, +0.1] | -0.4 [-1.6, +1.2] | +0.1 [-0.5, +0.7] |
| alpha=1 delta | +5.2 [-0.2, +10.8] | +6.5 [-12.1, +32.5] | +43.0 [-22.4, +86.2] |
| alpha=1 min-var | +0.6 [-0.8, +2.0] | +3.4 [-1.9, +12.2] | **+15.7 [+0.4, +28.9]** |
| Rough delta | **+7.1 [+1.1, +13.0]** | +7.8 [-13.1, +36.2] | +46.9 [-29.8, +91.5] |
| Rough min-var | +0.9 [-0.9, +2.9] | +6.0 [-2.3, +17.7] | **+15.0 [+2.3, +25.9]** |

### Reference: Heston min-var (fixed), test without COVID

| Hedge | RMSE difference [95% CI] | ES95 difference [95% CI] | ES99 difference [95% CI] |
|---|---|---|---|
| Black-Scholes | -0.5 [-1.8, +0.9] | -5.5 [-9.5, +0.7] | **-6.1 [-8.4, -1.4]** |
| Heston delta (fixed) | +0.8 [-1.3, +3.2] | -4.2 [-12.4, +6.7] | -9.5 [-15.8, +1.0] |
| Heston delta, 12-month | +1.5 [-0.4, +3.6] | +0.0 [-8.6, +8.3] | -3.7 [-10.8, +5.3] |
| Heston min-var, 12-month | +0.4 [-0.3, +1.1] | +2.7 [-2.1, +5.9] | -0.9 [-3.8, +4.9] |
| Heston delta, 6-month | +1.7 [-0.2, +3.7] | +1.9 [-7.6, +8.9] | -2.3 [-8.3, +7.3] |
| Heston min-var, 6-month | +0.5 [-0.5, +1.4] | +3.1 [-1.9, +6.2] | -0.1 [-3.5, +5.5] |
| Heston delta, expanding | +0.7 [-1.5, +3.4] | -4.9 [-13.3, +6.8] | -10.2 [-17.4, +0.5] |
| Heston min-var, expanding | -0.2 [-0.3, +0.0] | -0.7 [-1.9, +1.0] | -0.0 [-1.1, +1.2] |
| alpha=1 delta | +1.3 [-1.5, +4.5] | -5.3 [-16.7, +9.6] | -7.9 [-24.8, +3.6] |
| alpha=1 min-var | +0.0 [-1.0, +1.2] | +1.4 [-2.5, +8.6] | +6.0 [-1.3, +11.1] |
| Rough delta | +3.3 [-0.8, +7.5] | -2.4 [-19.8, +18.2] | -2.9 [-33.4, +18.1] |
| Rough min-var | +1.5 [-0.6, +3.7] | +7.7 [-0.2, +19.8] | **+16.4 [+2.1, +28.1]** |

## A3. Win rates (daily, 5 bp)

Share of options where the hedge's cost-adjusted error is smaller in size than the reference's. 50% = no edge.

| Hedge | vs Black-Scholes (with COVID) | vs Black-Scholes (without COVID) | vs Heston min-var (with COVID) | vs Heston min-var (without COVID) |
|---|---|---|---|---|
| Black-Scholes | - | - | 47% | 50% |
| Heston delta (fixed) | 38% | 38% | 43% | 45% |
| Heston min-var (fixed) | 53% | 50% | - | - |
| Heston delta, 12-month | 38% | 38% | 40% | 42% |
| Heston min-var, 12-month | 50% | 47% | 46% | 45% |
| Heston delta, 6-month | 39% | 39% | 41% | 42% |
| Heston min-var, 6-month | 51% | 49% | 45% | 44% |
| Heston delta, expanding | 39% | 40% | 43% | 45% |
| Heston min-var, expanding | 52% | 50% | 48% | 49% |
| alpha=1 delta | 37% | 39% | 41% | 43% |
| alpha=1 min-var | 52% | 50% | 53% | 52% |
| Rough delta | 36% | 37% | 38% | 40% |
| Rough min-var | 51% | 49% | 49% | 47% |

## A4. Rank in each year (RMSE, daily, 5 bp; 1 = best of 13)

| Hedge | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Years better than BS | Average rank |
|---|---|---|---|---|---|---|---|---|---|---|
| Black-Scholes | 2 | 11 | 1 | 7 | 3 | 4 | 3 | 7 | - | 4.8 |
| Heston delta (fixed) | 4 | 9 | 3 | 8 | 8 | 3 | 5 | 10 | 2 | 6.2 |
| Heston min-var (fixed) | 9 | 4 | 6 | 4 | 11 | 7 | 11 | 4 | 3 | 7.0 |
| Heston delta, 12-month | 1 | 7 | 9 | 9 | 5 | 12 | 6 | 9 | 2 | 7.2 |
| Heston min-var, 12-month | 6 | 2 | 12 | 5 | 9 | 9 | 12 | 5 | 3 | 7.5 |
| Heston delta, 6-month | 3 | 8 | 5 | 11 | 4 | 13 | 7 | 8 | 1 | 7.4 |
| Heston min-var, 6-month | 7 | 3 | 10 | 6 | 1 | 10 | 10 | 6 | 4 | 6.6 |
| Heston delta, expanding | 5 | 10 | 2 | 10 | 7 | 1 | 4 | 11 | 2 | 6.2 |
| Heston min-var, expanding | 8 | 5 | 7 | 3 | 12 | 6 | 9 | 3 | 3 | 6.6 |
| alpha=1 delta | 10 | 12 | 4 | 12 | 2 | 2 | 2 | 12 | 3 | 7.0 |
| alpha=1 min-var | 11 | 6 | 8 | 2 | 10 | 8 | 8 | 1 | 3 | 6.8 |
| Rough delta | 12 | 13 | 13 | 13 | 6 | 5 | 1 | 13 | 1 | 9.5 |
| Rough min-var | 13 | 1 | 11 | 1 | 13 | 11 | 13 | 2 | 3 | 8.1 |

Options per year: 2019: 245, 2020: 252, 2021: 248, 2022: 248, 2023: 246, 2024: 249, 2025: 249, 2026: 177 (2026 to 18 Sep).

## A5. Rank stability across all 18 settings (3 costs x 3 strategies x with/without COVID)

| Hedge | Avg rank RMSE | Best-worst rank RMSE | Avg rank ES99 | Best-worst rank ES99 | Settings beating BS on RMSE | Settings beating BS on ES99 |
|---|---|---|---|---|---|---|
| Heston min-var, expanding | 1.8 | 1-3 | 6.4 | 2-11 | 9/18 | 9/18 |
| Heston min-var (fixed) | 2.7 | 1-4 | 6.2 | 2-11 | 9/18 | 9/18 |
| Heston min-var, 12-month | 3.9 | 2-6 | 4.6 | 1-10 | 9/18 | 9/18 |
| Black-Scholes | 3.9 | 1-7 | 8.1 | 4-11 | - | - |
| alpha=1 min-var | 5.1 | 2-9 | 10.2 | 5-12 | 9/18 | 9/18 |
| Heston min-var, 6-month | 5.2 | 4-8 | 6.8 | 4-11 | 9/18 | 9/18 |
| Heston delta (fixed) | 7.4 | 4-9 | 5.3 | 2-10 | 2/18 | 18/18 |
| Heston delta, expanding | 7.9 | 5-10 | 4.0 | 1-9 | 0/18 | 18/18 |
| Rough min-var | 8.4 | 5-12 | 10.6 | 6-13 | 6/18 | 9/18 |
| Heston delta, 12-month | 9.5 | 8-11 | 5.1 | 4-6 | 0/18 | 12/18 |
| alpha=1 delta | 10.9 | 9-12 | 7.4 | 1-12 | 0/18 | 9/18 |
| Heston delta, 6-month | 11.2 | 10-12 | 7.3 | 6-10 | 0/18 | 12/18 |
| Rough delta | 13.0 | 13-13 | 9.1 | 4-13 | 0/18 | 4/18 |

## A6. What drives the tail: start months of the worst 1% of options (daily, 5 bp, with COVID)

- **Black-Scholes** (20 options): 2019-08 x5, 2020-02 x3, 2020-03 x9, 2024-05 x3
- **Heston min-var (fixed)** (20 options): 2019-08 x7, 2020-02 x1, 2020-03 x8, 2024-05 x4
- **Rough min-var** (20 options): 2019-08 x8, 2020-02 x1, 2020-03 x8, 2024-05 x3
- **Rough delta** (20 options): 2019-08 x6, 2020-02 x2, 2020-03 x9, 2024-05 x3

- **Black-Scholes, without COVID** (19 options): 2019-08 x9, 2019-09 x1, 2024-05 x9
- **Heston min-var (fixed), without COVID** (19 options): 2019-08 x9, 2019-09 x1, 2024-05 x9

## A7. Stress windows (daily, 5 bp; RMSE / worst loss, bp)

| Hedge | COVID window (Jan-Jun 2020) (n=113) | VIX stress, excl. COVID (n=332) | Calm (VIX below 75th pct) (n=1501) | Fall 2010-11 (-28%), in-sample (n=278) | Fall 2015-16 (-23%), in-sample (n=245) | COVID fall Jan-Mar 2020 (-38%) (n=49) |
|---|---|---|---|---|---|---|
| Black-Scholes | 166 / 512 | 58 / 186 | 53 / 296 | 50 / 95 | 66 / 176 | 210 / 512 |
| Heston delta (fixed) | 160 / 486 | 60 / 176 | 54 / 298 | 51 / 102 | 69 / 176 | 198 / 486 |
| Heston min-var (fixed) | 144 / 449 | 59 / 193 | 53 / 300 | 45 / 103 | 60 / 168 | 175 / 449 |
| Heston delta, 12-month | 156 / 472 | 63 / 190 | 54 / 298 | 51 / 102 | 69 / 176 | 190 / 472 |
| Heston min-var, 12-month | 143 / 444 | 61 / 193 | 53 / 299 | 45 / 103 | 60 / 168 | 174 / 444 |
| Heston delta, 6-month | 158 / 492 | 64 / 193 | 54 / 299 | 51 / 102 | 69 / 176 | 192 / 492 |
| Heston min-var, 6-month | 143 / 459 | 61 / 195 | 52 / 299 | 45 / 103 | 60 / 168 | 176 / 459 |
| Heston delta, expanding | 160 / 485 | 60 / 174 | 54 / 298 | 51 / 102 | 69 / 176 | 197 / 485 |
| Heston min-var, expanding | 144 / 450 | 59 / 193 | 53 / 300 | 45 / 103 | 60 / 168 | 175 / 450 |
| alpha=1 delta | 173 / 537 | 61 / 165 | 55 / 306 | 54 / 99 | 72 / 186 | 219 / 537 |
| alpha=1 min-var | 149 / 480 | 59 / 194 | 53 / 305 | 44 / 101 | 57 / 164 | 183 / 480 |
| Rough delta | 176 / 548 | 64 / 181 | 56 / 333 | 56 / 111 | 75 / 198 | 221 / 548 |
| Rough min-var | 142 / 472 | 60 / 196 | 54 / 326 | 43 / 119 | 53 / 144 | 172 / 472 |

In the 2010-2018 falls the monthly-updated hedges use the fixed 2010-2018 parameters (they were only re-estimated in the test period), so they equal the fixed versions there.

# Part B - Pricing (traded NIFTY calls, ~3 months to expiry, test 2019-2026)

Error = model minus market. vol pts = implied-volatility points; bp = basis points of the futures price. Each model uses only that day's India VIX (and its own parameters); none is fitted to option prices.

## B1. Error breakdown: ATM call (the paper's option), one per day (1819 options, 1819 days)

| Model | Bias vol | MAE vol | RMSE vol | P95 abs vol | P99 abs vol | Overpriced | Bias bp | RMSE bp | P99 abs bp |
|---|---|---|---|---|---|---|---|---|---|
| Rough Heston | +0.51 | 1.36 | 2.25 | 4.81 | 8.33 | 62% | +9.3 | 40.2 | 152.1 |
| alpha=1 control | +0.84 | 1.46 | 2.33 | 4.78 | 8.16 | 73% | +15.2 | 41.7 | 151.9 |
| Black-Scholes | +1.45 | 1.81 | 2.57 | 5.06 | 8.48 | 88% | +26.1 | 46.2 | 157.4 |
| Heston, expanding | +2.20 | 2.69 | 3.30 | 4.54 | 8.71 | 91% | +39.8 | 59.0 | 159.0 |
| Heston, fixed 2010-18 | +2.11 | 2.88 | 3.56 | 4.97 | 10.55 | 85% | +38.1 | 63.7 | 172.1 |
| Heston, 6-month | +1.89 | 2.77 | 3.73 | 7.66 | 11.57 | 82% | +34.4 | 67.4 | 221.9 |
| Heston, 12-month | +1.91 | 2.89 | 3.80 | 6.42 | 10.05 | 84% | +34.6 | 67.7 | 171.2 |

## B1. Error breakdown: All traded calls, strikes 0.90-1.10 x futures (46299 options, 1819 days)

| Model | Bias vol | MAE vol | RMSE vol | P95 abs vol | P99 abs vol | Overpriced | Bias bp | RMSE bp | P99 abs bp |
|---|---|---|---|---|---|---|---|---|---|
| Rough Heston | +0.44 | 1.36 | 1.95 | 4.06 | 6.49 | 60% | +6.5 | 28.3 | 96.4 |
| alpha=1 control | +0.73 | 1.40 | 2.00 | 4.13 | 6.48 | 68% | +10.9 | 29.5 | 98.9 |
| Black-Scholes | +1.33 | 1.80 | 2.38 | 4.72 | 7.07 | 80% | +20.1 | 35.9 | 110.1 |
| Heston, 6-month | +1.39 | 2.22 | 2.92 | 5.79 | 9.18 | 76% | +21.2 | 45.3 | 150.0 |
| Heston, 12-month | +1.60 | 2.31 | 2.96 | 5.85 | 7.86 | 80% | +24.2 | 45.6 | 127.0 |
| Heston, expanding | +2.49 | 2.69 | 3.02 | 4.55 | 6.54 | 94% | +37.2 | 46.1 | 92.1 |
| Heston, fixed 2010-18 | +2.64 | 2.93 | 3.33 | 4.98 | 6.79 | 92% | +39.5 | 51.2 | 98.9 |

## B2. Largest single mispricings (all strikes, vol pts)

| Model | Largest overpricing (date, K/F, market IV) | Largest underpricing (date, K/F, market IV) |
|---|---|---|
| Black-Scholes | +30.0 (2020-03-25, 1.01, 47.7%) | -30.3 (2020-04-07, 1.04, 82.1%) |
| Heston, fixed 2010-18 | +12.8 (2026-04-22, 0.96, 6.0%) | -44.9 (2020-04-07, 1.04, 82.1%) |
| Heston, 12-month | +14.5 (2026-04-02, 0.92, 10.3%) | -47.3 (2020-04-07, 1.04, 82.1%) |
| Heston, 6-month | +15.5 (2020-07-21, 0.96, 16.3%) | -42.9 (2020-04-07, 1.04, 82.1%) |
| Heston, expanding | +14.8 (2020-04-09, 0.96, 33.6%) | -35.1 (2020-03-23, 1.05, 88.3%) |
| alpha=1 control | +29.5 (2020-03-25, 1.01, 47.7%) | -31.0 (2020-04-07, 1.04, 82.1%) |
| Rough Heston | +29.2 (2020-03-25, 1.01, 47.7%) | -31.3 (2020-04-07, 1.04, 82.1%) |

## B3. Error by strike (mean error / mean absolute error, vol pts)

| K/F | Options | Market IV % | Black-Scholes | Heston, fixed 2010-18 | Heston, 12-month | Heston, 6-month | Heston, expanding | alpha=1 control | Rough Heston |
|---|---|---|---|---|---|---|---|---|---|
| 0.90-0.95 | 4646 | 16.0 | -0.02 / 1.62 | +2.35 / 3.17 | +1.06 / 2.59 | +0.55 / 2.47 | +2.55 / 3.08 | +1.56 / 2.13 | +1.81 / 2.31 |
| 0.95-0.99 | 12492 | 14.6 | +0.56 / 1.22 | +2.66 / 2.98 | +1.43 / 2.25 | +1.08 / 2.08 | +2.55 / 2.75 | +0.76 / 1.26 | +0.64 / 1.22 |
| 0.99-1.01 | 8609 | 14.0 | +1.17 / 1.44 | +2.79 / 3.00 | +1.75 / 2.30 | +1.54 / 2.17 | +2.57 / 2.71 | +0.54 / 1.13 | +0.20 / 1.07 |
| 1.01-1.05 | 12954 | 13.7 | +1.93 / 2.11 | +2.87 / 3.04 | +1.92 / 2.39 | +1.77 / 2.28 | +2.61 / 2.72 | +0.59 / 1.32 | +0.12 / 1.26 |
| 1.05-1.10 | 7598 | 14.5 | +2.56 / 2.77 | +2.20 / 2.45 | +1.49 / 2.15 | +1.59 / 2.22 | +2.06 / 2.26 | +0.61 / 1.65 | +0.11 / 1.52 |

## B4. Are the pricing differences real? (block bootstrap over days, block 63 trading days)

Difference in RMSE (vol pts), model minus reference; negative = model prices better.

### ATM

| Model | vs Black-Scholes [95% CI] | vs Rough Heston [95% CI] |
|---|---|---|
| Black-Scholes | - | **+0.33 [+0.24, +0.44]** |
| Heston, fixed 2010-18 | **+0.99 [+0.47, +1.53]** | **+1.31 [+0.81, +1.82]** |
| Heston, 12-month | **+1.22 [+0.59, +1.82]** | **+1.55 [+0.94, +2.16]** |
| Heston, 6-month | **+1.16 [+0.40, +2.05]** | **+1.48 [+0.71, +2.41]** |
| Heston, expanding | **+0.73 [+0.29, +1.20]** | **+1.06 [+0.60, +1.51]** |
| alpha=1 control | **-0.25 [-0.32, -0.20]** | **+0.08 [+0.04, +0.12]** |
| Rough Heston | **-0.33 [-0.44, -0.24]** | - |

### All strikes

| Model | vs Black-Scholes [95% CI] | vs Rough Heston [95% CI] |
|---|---|---|
| Black-Scholes | - | **+0.43 [+0.25, +0.61]** |
| Heston, fixed 2010-18 | **+0.95 [+0.36, +1.54]** | **+1.38 [+0.85, +1.85]** |
| Heston, 12-month | **+0.57 [+0.08, +1.15]** | **+1.01 [+0.54, +1.55]** |
| Heston, 6-month | **+0.54 [+0.01, +1.12]** | **+0.97 [+0.45, +1.54]** |
| Heston, expanding | **+0.64 [+0.17, +1.11]** | **+1.07 [+0.66, +1.44]** |
| alpha=1 control | **-0.39 [-0.51, -0.25]** | +0.04 [-0.02, +0.10] |
| Rough Heston | **-0.43 [-0.61, -0.25]** | - |

## B5. Win rates and yearly ranks (ATM call)

| Model | Days beating BS | Days beating Rough Heston | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Avg rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rough Heston | 77% | - | 1 | 1 | 1 | 1 | 2 | 4 | 1 | 1 | 1.5 |
| alpha=1 control | 81% | 33% | 2 | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 2.0 |
| Black-Scholes | - | 23% | 3 | 3 | 3 | 5 | 3 | 5 | 3 | 3 | 3.5 |
| Heston, 12-month | 29% | 22% | 5 | 6 | 7 | 6 | 5 | 1 | 5 | 4 | 4.9 |
| Heston, 6-month | 34% | 25% | 4 | 7 | 6 | 7 | 4 | 2 | 4 | 5 | 4.9 |
| Heston, expanding | 31% | 22% | 7 | 4 | 5 | 4 | 6 | 6 | 6 | 6 | 5.5 |
| Heston, fixed 2010-18 | 31% | 24% | 6 | 5 | 4 | 3 | 7 | 7 | 7 | 7 | 5.8 |

Yearly ranks: 1 = lowest ATM RMSE of the 7 models in that year.

