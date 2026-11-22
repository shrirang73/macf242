# Option pricing: all models, traded NIFTY calls, test period 2019-2026

Error = model implied vol - market implied vol (vol points). Shape RMSE = after removing each day's average error. vs BS = change in mean squared error vs Black-Scholes; z from a moving-block bootstrap (|z|>2 ~ significant). Ranked by RMSE.

## ATM call, ~3 months (the paper's option) (1819 options, 1819 days)

| Rank | Model | Bias (vol pts) | RMSE (vol pts) | Shape RMSE | vs BS (sq. error) | z |
|---|---|---|---|---|---|---|
| 1 | Rough Heston | +0.51 | 2.25 | 0.00 | -24% | -5.43 |
| 2 | alpha=1 control | +0.84 | 2.33 | 0.00 | -18% | -6.58 |
| 3 | Black-Scholes | +1.45 | 2.57 | 0.00 | +0% | +0.00 |
| 4 | Heston, expanding | +2.20 | 3.30 | 0.00 | +65% | +2.69 |
| 5 | Heston, fixed 2010-18 | +2.11 | 3.56 | 0.00 | +91% | +3.14 |
| 6 | Heston, 6-month | +1.89 | 3.73 | 0.00 | +110% | +2.29 |
| 7 | Heston, 12-month | +1.91 | 3.80 | 0.00 | +117% | +2.94 |

## All traded calls, strikes 0.90-1.10 x futures (46299 options, 1819 days)

| Rank | Model | Bias (vol pts) | RMSE (vol pts) | Shape RMSE | vs BS (sq. error) | z |
|---|---|---|---|---|---|---|
| 1 | Rough Heston | +0.44 | 1.95 | 1.22 | -33% | -4.71 |
| 2 | alpha=1 control | +0.73 | 2.00 | 1.15 | -30% | -5.81 |
| 3 | Black-Scholes | +1.33 | 2.38 | 1.32 | +0% | +0.00 |
| 4 | Heston, 6-month | +1.39 | 2.92 | 1.16 | +50% | +1.80 |
| 5 | Heston, 12-month | +1.60 | 2.96 | 1.12 | +54% | +2.09 |
| 6 | Heston, expanding | +2.49 | 3.02 | 1.09 | +61% | +3.31 |
| 7 | Heston, fixed 2010-18 | +2.64 | 3.33 | 1.10 | +95% | +3.93 |

## Mean error by strike (vol pts)

| K/F | Market IV % | Black-Scholes | Heston, fixed 2010-18 | Heston, 12-month | Heston, 6-month | Heston, expanding | alpha=1 control | Rough Heston |
|---|---|---|---|---|---|---|---|---|
| 0.90-0.95 | 16.0 | -0.02 | +2.35 | +1.06 | +0.55 | +2.55 | +1.56 | +1.81 |
| 0.95-0.99 | 14.6 | +0.56 | +2.66 | +1.43 | +1.08 | +2.55 | +0.76 | +0.64 |
| 0.99-1.01 | 14.0 | +1.17 | +2.79 | +1.75 | +1.54 | +2.57 | +0.54 | +0.20 |
| 1.01-1.05 | 13.7 | +1.93 | +2.87 | +1.92 | +1.77 | +2.61 | +0.59 | +0.12 |
| 1.05-1.10 | 14.5 | +2.56 | +2.20 | +1.49 | +1.59 | +2.06 | +0.61 | +0.11 |
