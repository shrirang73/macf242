# Hedging error / ES99 for every hedge (NIFTY, test 2019-2026)

Each cell: RMSE (overall hedging error) / ES99 (average loss in the worst 1% of options), bp of notional.
Rules picked on 2010-2018 for each hedge: daily = fixed/every=1; regime = regime[vix_hyst]/calm=2,stress=1; band = band/h=0.02 (h=0.05 at 20 bp).

## Whole test period 2019-2026 (with COVID)

### Cost 0 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 64.0 / 302.2 | 70.2 / 349.8 | 64.2 / 298.7 |
| Heston delta (fixed) | 64.1 / 282.1 | 70.3 / 332.3 | 64.4 / 278.2 |
| Heston min-var (fixed) | 61.1 / 265.8 | 67.6 / 323.0 | 61.4 / 262.4 |
| Heston delta, 12-month | 64.1 / 275.1 | 70.3 / 326.5 | 64.3 / 270.9 |
| Heston min-var, 12-month | 61.1 / 262.8 | 67.5 / 319.3 | 61.3 / 258.7 |
| Heston delta, 6-month | 64.5 / 280.8 | 70.7 / 334.3 | 64.8 / 277.1 |
| Heston min-var, 6-month | 61.1 / 267.1 | 67.6 / 325.3 | 61.4 / 263.8 |
| Heston delta, expanding | 64.1 / 281.5 | 70.3 / 331.9 | 64.4 / 277.6 |
| Heston min-var, expanding | 61.0 / 266.0 | 67.5 / 322.9 | 61.2 / 262.6 |
| alpha=1 delta | 66.5 / 309.9 | 72.7 / 359.6 | 66.8 / 307.3 |
| alpha=1 min-var | 61.5 / 281.7 | 68.2 / 342.0 | 61.7 / 277.2 |
| Rough delta | 68.3 / 312.9 | 74.7 / 366.0 | 68.6 / 310.0 |
| Rough min-var | 61.5 / 280.4 | 68.4 / 345.2 | 61.6 / 275.7 |

### Cost 5 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 64.8 / 313.1 | 70.9 / 361.5 | 65.0 / 309.5 |
| Heston delta (fixed) | 64.9 / 293.6 | 70.9 / 343.8 | 65.2 / 289.7 |
| Heston min-var (fixed) | 62.2 / 278.2 | 68.5 / 335.2 | 62.4 / 274.4 |
| Heston delta, 12-month | 65.0 / 287.1 | 71.0 / 338.7 | 65.2 / 283.0 |
| Heston min-var, 12-month | 62.3 / 275.1 | 68.5 / 331.6 | 62.4 / 270.7 |
| Heston delta, 6-month | 65.4 / 293.1 | 71.5 / 346.4 | 65.7 / 289.5 |
| Heston min-var, 6-month | 62.3 / 279.8 | 68.6 / 337.6 | 62.5 / 276.3 |
| Heston delta, expanding | 65.0 / 293.1 | 70.9 / 343.4 | 65.2 / 289.2 |
| Heston min-var, expanding | 62.1 / 278.3 | 68.4 / 335.2 | 62.3 / 274.5 |
| alpha=1 delta | 67.4 / 321.2 | 73.4 / 370.9 | 67.6 / 318.5 |
| alpha=1 min-var | 62.8 / 293.9 | 69.3 / 354.1 | 62.9 / 289.2 |
| Rough delta | 69.3 / 325.1 | 75.4 / 377.5 | 69.6 / 321.8 |
| Rough min-var | 63.1 / 293.2 | 69.8 / 357.5 | 63.2 / 288.3 |

### Cost 20 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 79.6 / 347.0 | 82.7 / 396.8 | 79.0 / 335.6 |
| Heston delta (fixed) | 79.6 / 330.2 | 82.6 / 380.1 | 79.4 / 329.0 |
| Heston min-var (fixed) | 77.6 / 316.2 | 80.9 / 372.6 | 77.3 / 316.9 |
| Heston delta, 12-month | 79.9 / 324.6 | 82.9 / 375.0 | 79.8 / 326.9 |
| Heston min-var, 12-month | 77.9 / 313.7 | 81.2 / 369.3 | 77.5 / 312.0 |
| Heston delta, 6-month | 80.3 / 331.4 | 83.4 / 382.7 | 80.3 / 335.0 |
| Heston min-var, 6-month | 78.1 / 318.9 | 81.4 / 374.5 | 77.9 / 321.4 |
| Heston delta, expanding | 79.7 / 329.7 | 82.7 / 379.7 | 79.5 / 328.4 |
| Heston min-var, expanding | 77.6 / 316.3 | 81.0 / 372.5 | 77.3 / 317.1 |
| alpha=1 delta | 82.2 / 356.4 | 85.1 / 406.6 | 81.5 / 347.5 |
| alpha=1 min-var | 78.8 / 331.6 | 82.2 / 391.0 | 78.0 / 326.3 |
| Rough delta | 84.8 / 361.7 | 87.4 / 413.4 | 83.9 / 353.2 |
| Rough min-var | 79.9 / 332.3 | 83.3 / 395.6 | 79.4 / 327.5 |

## Test period without the COVID window (options starting 2020-01-15..2020-06-30 removed)

### Cost 0 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 50.7 / 196.3 | 58.3 / 276.8 | 50.9 / 196.8 |
| Heston delta (fixed) | 52.2 / 192.9 | 59.5 / 268.4 | 52.4 / 193.6 |
| Heston min-var (fixed) | 51.1 / 202.5 | 58.7 / 281.9 | 51.4 / 203.2 |
| Heston delta, 12-month | 52.9 / 198.5 | 60.2 / 272.7 | 53.1 / 199.3 |
| Heston min-var, 12-month | 51.4 / 201.4 | 58.9 / 280.0 | 51.7 / 202.0 |
| Heston delta, 6-month | 53.0 / 199.8 | 60.3 / 274.1 | 53.2 / 200.8 |
| Heston min-var, 6-month | 51.5 / 202.1 | 59.1 / 280.5 | 51.7 / 203.0 |
| Heston delta, expanding | 52.2 / 192.2 | 59.5 / 267.9 | 52.4 / 192.9 |
| Heston min-var, expanding | 50.9 / 202.4 | 58.6 / 282.2 | 51.2 / 203.2 |
| alpha=1 delta | 52.7 / 194.1 | 60.2 / 270.2 | 52.9 / 194.1 |
| alpha=1 min-var | 50.9 / 208.3 | 58.8 / 292.5 | 51.1 / 208.4 |
| Rough delta | 54.5 / 199.3 | 62.1 / 273.5 | 54.6 / 199.0 |
| Rough min-var | 52.0 / 218.5 | 60.1 / 302.8 | 52.2 / 218.1 |

### Cost 5 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 52.3 / 213.5 | 59.6 / 292.4 | 52.4 / 213.8 |
| Heston delta (fixed) | 53.6 / 210.1 | 60.7 / 283.9 | 53.8 / 210.7 |
| Heston min-var (fixed) | 52.8 / 219.6 | 60.2 / 297.4 | 53.0 / 220.2 |
| Heston delta, 12-month | 54.3 / 215.9 | 61.4 / 288.3 | 54.5 / 216.6 |
| Heston min-var, 12-month | 53.2 / 218.7 | 60.5 / 295.6 | 53.4 / 219.2 |
| Heston delta, 6-month | 54.5 / 217.3 | 61.6 / 289.8 | 54.7 / 218.1 |
| Heston min-var, 6-month | 53.3 / 219.5 | 60.6 / 296.2 | 53.5 / 220.3 |
| Heston delta, expanding | 53.6 / 209.4 | 60.6 / 283.5 | 53.8 / 210.0 |
| Heston min-var, expanding | 52.7 / 219.6 | 60.0 / 297.7 | 52.9 / 220.2 |
| alpha=1 delta | 54.2 / 211.6 | 61.4 / 285.7 | 54.4 / 211.5 |
| alpha=1 min-var | 52.9 / 225.6 | 60.5 / 308.1 | 53.0 / 225.5 |
| Rough delta | 56.1 / 216.7 | 63.3 / 288.3 | 56.3 / 216.3 |
| Rough min-var | 54.3 / 235.9 | 62.0 / 318.5 | 54.4 / 235.3 |

### Cost 20 bp

| Hedge | Daily | Regime-aware | Band |
|---|---|---|---|
| Black-Scholes | 70.8 / 265.1 | 74.2 / 339.2 | 69.9 / 270.3 |
| Heston delta (fixed) | 71.5 / 261.9 | 74.9 / 330.6 | 70.6 / 266.8 |
| Heston min-var (fixed) | 71.2 / 271.0 | 74.9 / 344.0 | 70.3 / 277.1 |
| Heston delta, 12-month | 72.4 / 268.1 | 75.7 / 335.2 | 71.5 / 275.1 |
| Heston min-var, 12-month | 71.8 / 270.7 | 75.3 / 342.5 | 70.9 / 277.3 |
| Heston delta, 6-month | 72.6 / 269.7 | 76.0 / 337.0 | 71.8 / 276.9 |
| Heston min-var, 6-month | 71.9 / 271.7 | 75.5 / 343.2 | 71.1 / 278.8 |
| Heston delta, expanding | 71.6 / 261.2 | 74.9 / 330.2 | 70.7 / 266.0 |
| Heston min-var, expanding | 71.2 / 271.0 | 74.8 / 344.2 | 70.3 / 277.1 |
| alpha=1 delta | 72.8 / 264.3 | 75.9 / 332.6 | 71.7 / 265.3 |
| alpha=1 min-var | 72.1 / 277.5 | 75.8 / 354.9 | 71.1 / 281.2 |
| Rough delta | 75.3 / 268.9 | 78.1 / 334.7 | 74.0 / 268.1 |
| Rough min-var | 74.0 / 288.4 | 77.7 / 365.6 | 73.1 / 289.7 |
