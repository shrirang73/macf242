# Heston and rough Heston option pricing and hedging on NIFTY (MAC F244)

Final report: `report/final_report.pdf` (source `report/final_report.tex`, amsart course template).
An earlier single-experiment version is `report/report.pdf`.

## Contents
| Folder | What |
|---|---|
| `heston/` | Python code (Experiment B: daily Heston calibration, one-day-ahead pricing, hedging traded calls) |
| `tests/` | 16 tests (`python -m pytest tests -q`) |
| `data/raw/bhavcopy/` | NSE F&O bhavcopy, NIFTY rows, 2010-07-19 .. 2026-10-21, one gzipped CSV per day (+ `MANIFEST.jsonl`). The daily files (504 MB) are not on GitHub; `MANIFEST.jsonl` lists them, and they are in the submission zip. The cleaned chains in `data/chains.csv.gz` are enough to rerun everything from calibration on. |
| `data/inputs/` | NIFTY 50 closes, India VIX, causal regime labels (from our earlier project) |
| `data/chains.csv.gz`, `data/futures.csv.gz` | cleaned option chains and futures closes |
| `results/` | calibrated parameters (`params.csv`), pricing errors (`oos.csv.gz`), hedging errors (`hedge.csv.gz`, `hedge_paired.csv`) |
| `report/` | reports, `numbers.tex` (generated macros), `tables/`, `figures/` (PDF), `figures_png/` |
| `demo/` | interactive demo `hedged_desk.html` (open in any browser with internet, for the chart library): run a hedged option-selling desk on the real data with your own capital, leverage, hedge and costs; built by `python -m heston.demo_data` from `template.html` |
| `team_code/` | the team's time-series / rough Heston code (Experiment A), unmodified zip, and `verification/`: outputs of a full re-run of that code (19 Nov 2026, 69 tests pass) used to check every Experiment A number in the report |

## What the code does
| Module | Stage |
|---|---|
| `heston/pricer.py`, `heston/calib.py` | Heston Fourier pricer (checked against Monte Carlo and an independent implementation); daily calibration |
| `heston/data.py` | option chains from the raw bhavcopy |
| `heston/oos.py` | one-day-ahead pricing test (Heston vs Black-Scholes benchmarks) |
| `heston/hedge.py`, `heston/analyse_hedge.py` | delta-hedging backtest of traded calls; paired block-bootstrap tests by period and event |
| `heston/simulate.py` | simulated Heston world: how much the minimum-variance hedge gains when the model is true |
| `heston/strategy.py` | trading demo: selling hedged ATM calls (volatility risk premium) with costs and a VIX risk rule |
| `heston/report_assets.py` | every number, table and figure used by the reports |

## Reproduce everything (Python 3.13+, numpy, scipy, pandas, matplotlib; about 45 minutes)
`./run_all.sh` runs all the steps below in order (set `PY=` to your Python).
```
python -m pytest tests -q
python -m heston.data                       # chains + futures from data/raw (a few minutes)
python -m heston.run_calibration --workers 5   # daily calibration (~5-10 min)
python -m heston.oos                        # one-day-ahead pricing errors
python -m heston.hedge                      # hedging backtest
python -m heston.analyse_hedge              # paired block-bootstrap tables
python -m heston.simulate                   # simulated Heston world (~20 min)
python -m heston.report_assets              # strategy backtest + numbers.tex, tables, figures
python -m heston.demo_data                  # interactive demo page (demo/hedged_desk.html)
cd report && pdflatex final_report && pdflatex final_report
```
All Experiment B numbers in the report come from `report/numbers.tex` (generated).
Experiment A numbers are transcribed from the team's outputs into
`report/teammate_numbers.tex`, `report/tables/ts_pricing.tex` and `report/tables/ts_hedging.tex`;
to regenerate them run the team's code (see `team_code/`, its README).

## Data sources
NSE India F&O bhavcopy archives (nsearchives.nseindia.com); NIFTY 50 history (niftyindices.com);
India VIX (NSE, Yahoo Finance series checked against NSE).
