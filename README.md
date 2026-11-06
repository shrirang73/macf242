# Variance-optimal hedging under rough volatility and jumps

Course project for MAC F244 (Stochastic Calculus and Applications to Finance),
BITS Pilani Goa, Sem I 2026-27. Work in progress; final submission 15 Nov 2026.

Black-Scholes theory hedges an option perfectly by holding its delta, rebalanced
continuously and for free. Real hedgers trade **discretely**, pay **costs**, and
face markets that switch between **calm and stress**. We ask how often to
rebalance, whether to trade more in stress, and how much a better hedge ratio
is worth. We test this in two settings:

- **In simulation**, under the rough Hawkes-Heston model (rough volatility,
  clustered crash jumps), using the variance-optimal hedge from Y. Wang and
  X. Wang, *Variance-Optimal Hedging in the Rough Hawkes-Heston Model*
  ([arXiv:2609.08541](https://arxiv.org/abs/2609.08541)), computed with their
  published code.
- **On Indian market data**: NIFTY 50, India VIX and traded NIFTY options.

The mathematics used is Ito's formula, Black-Scholes delta hedging, the
discrete-hedging error, gamma P&L, and the minimum-variance hedge in incomplete
markets.

**Start here:** [`report/pdf/report.pdf`](report/pdf/report.pdf) (draft report)
and [`report/pdf/viva_note.pdf`](report/pdf/viva_note.pdf) (plain-language
explanation of the maths and results). Detailed logs of every check are in
[`notes/`](notes/).

## What is reused and what is ours

| Part | Source |
|---|---|
| Model simulation and the variance-optimal hedge (Fourier and Riccati-Volterra methods) | **The authors' official code**, used unmodified as a git submodule ([gagawjbytw/hedging_rough_hawkes_heston](https://github.com/gagawjbytw/hedging_rough_hawkes_heston), commit `c6a6557`). We did not re-implement it. |
| Replication of their numbers | Ours: rerun and checked (`results/replication/`, `notes/robustness_log.md`) |
| Discrete rebalancing with costs, rebalancing rules, equal-turnover comparisons | Ours (`src/rebalancing.py`, `src/experiment_discrete.py`, `src/analysis.py`) |
| Data pipeline: NIFTY 50, India VIX (verified against NSE), NSE F&O bhavcopy | Ours (`src/data_download.py`, `src/data_prepare.py`, `src/bhavcopy.py`, `src/fo_data.py`) |
| Regime labels (VIX, realised volatility, Markov-switching filter) | Ours (`src/regimes.py`) |
| Backtests with synthetic and traded NIFTY options | Ours (`src/backtest_real.py`, `src/backtest_options.py`, `src/backtest_analysis.py`) |
| Report and viva note, with every number generated from results | Ours (`report/`, `src/report_assets.py`) |

## Main findings so far

- **Replication.** Variance reduction against no hedge is 0.828 for the GKW
  hedge and 0.732 for the Black-Scholes delta. The official stored values are
  0.824 and 0.726; the difference is Monte Carlo noise. These numbers come from
  the official code, not from the arXiv paper itself.
- **Simulation (50,000 paths).** The GKW advantage over the BS delta grows as
  rebalancing becomes coarser:

  | Rebalancing | GKW | BS delta |
  |---|---|---|
  | every step | 0.828 | 0.732 |
  | 16 dates | 0.738 | 0.541 |
  | static | 0.374 | 0.108 |

  "Smart" rules (no-trade bands, trading more when variance is high) beat
  fixed schedules at equal expected cost by 6-31% in MSE, but only when
  turnover must be low. At the cost-optimal point the gain is small: -1.2% for
  GKW (z = -0.5).
- **Regimes.** The labels are calibrated on 2010-2018 and tested on 2019-2026.
  Out of sample, a stress day is followed by 1.7-2.0 times the realised
  volatility of a calm day.
- **NIFTY backtests** (synthetic options at India VIX, and traded NIFTY options
  hedged with futures):
  - Daily rebalancing is best for costs up to 20 bp.
  - The regime-aware rule picked on 2010-2018 is worse on 2019-2026 (traded
    options: +35% in error, z = 2.1).
  - Its apparent advantage at equal turnover comes from the COVID crash.
    Without that window it disappears (17 of 48 rules beat fixed schedules).
  - The seller's average profit is gamma P&L. Implied volatility exceeded the
    volatility that followed on 79% of days (India VIX) and for 73% of traded
    options, except in the COVID crash.

Limitations are listed in the report (Section 8) and in the logs in `notes/`.

## Repository layout

```
src/            our code (see docstrings at the top of each file)
tests/          41 checks: python -m pytest tests -q
notes/          detailed logs: every check, number and caveat
results/        result tables (CSV/JSON) and figures
report/         LaTeX report and viva note; report/pdf/ has the built PDFs
proposal/       submitted project proposal (LaTeX)
data/           verified daily NIFTY 50 / India VIX and raw provenance files
external/       the authors' code (git submodule, unmodified)
```

## Setup

```bash
git clone --recurse-submodules <this repository's URL>
cd <repository folder>
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

Tested with Python 3.14.7 on Windows (`requirements.txt` pins the versions we
used). If you cloned without `--recurse-submodules`, run
`git submodule update --init`. For reproducible timing and results, set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

## Reproducing the results

```bash
python -m src.experiment_discrete             # simulation, about 13 min
python -m src.analysis results/discrete_rebalancing --holdout results/robustness/seed_1234567
python -m src.data_download                   # NIFTY 50 and India VIX, about 1 min
python -m src.data_prepare && python -m src.regimes
python -m src.backtest_real && python -m src.backtest_analysis results/backtest_real/maturity_21
python -m src.bhavcopy                        # NSE F&O archive, about 1.5 h (resumable)
python -m src.backtest_options --vol own && python -m src.backtest_analysis results/backtest_options/own
python -m src.report_assets                   # regenerate report numbers, tables and figures
```

The large per-path arrays (`*.npz`) are not in the repository; the commands
above recreate them. The robustness runs (other seeds, grids, maturities) are
listed in the logs in `notes/`.

## Data

- **Included.** The verified daily series `data/processed/nifty50.csv` and
  `data/processed/indiavix.csv`, the small raw files they are built from, and
  provenance manifests (URLs, retrieval times, SHA-256).
- **Not included.** The NSE F&O bhavcopy extract (103 MB). Rebuild it with
  `python -m src.bhavcopy`; `data/raw/bhavcopy/MANIFEST.jsonl` lists every
  source file and its hash.
- **Sources.**
  - NIFTY 50: NSE Indices Ltd (niftyindices.com).
  - India VIX: Yahoo Finance (`^INDIAVIX`), checked and patched against NSE
    (nseindia.com).
  - Options and futures: NSE F&O bhavcopy archives (nsearchives.nseindia.com).

  These data are used for academic purposes only; see the providers' terms of
  use. Details and checks: `notes/data_log.md` and `notes/options_log.md`.

## Report

Every number in the report is generated from the result files by
`python -m src.report_assets`; none is typed by hand. To build:

```bash
cd report
pdflatex report && pdflatex report
```

The PDFs in `report/pdf/` are public builds that list group members by name
only. To rebuild them:

```bash
cd report
pdflatex -output-directory=pdf -jobname=report "\def\PublicBuild{}\input{report.tex}"
pdflatex -output-directory=pdf -jobname=viva_note "\def\PublicBuild{}\input{viva_note.tex}"
```

Run each command twice so cross-references resolve.

## Team

Dhyan Laad, Natasha Nalwaya, Shrirang Bhale, Tanmai Chekuri, Shivam Mahto.
