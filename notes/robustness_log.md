# Verification and robustness log

Every entry records what was checked, how, and the outcome. Numbers are copied
from files under `results/`. "Not verified" is stated explicitly where it applies.

Environment: Windows 11, Python 3.14.7, numpy 2.5.4, scipy 1.18.1, single-threaded
BLAS (`OPENBLAS_NUM_THREADS=1`). Upstream repo at commit `c6a6557` (23 Sep 2026), unmodified
(`git status` clean after every run; upstream output files restored with `git checkout`).

## A. The paper and the official code

| Item | Detail | Source |
|---|---|---|
| Paper | Wang & Wang, arXiv:2609.08541 v1, 8 Sep 2026 | arXiv |
| Model parameters (alpha, rho, b, c, Lambda, beta, V0) | (0.527, -0.731, -1.812, 0.115, 0.276, 0.049, 0.0079) | Paper Sec. 5.1 p. 21; the `Model` dataclass agrees |
| Future work named in the paper | Historical-measure hedging, discrete rebalancing, static option positions | Paper Sec. 6, p. 24 |
| Source of the 0.82 / 0.73 variance-reduction table | Official code output (terminal-loss scripts added to the repository on 23 Sep 2026, commit `c6a6557`). The arXiv v1 numerics report convergence diagnostics only (Tables 1-2). Cite these numbers as official-code output. | Repository history; paper Sec. 5 |
| Paper Tables 1-2 vs the repository's stored outputs | All digits match, including the grid-check values (3.867e-7, 4.171e-7, 5.446e-7, 6.327e-7) and the stock mean 1.00078 (se 7.86e-4) | `numerics/output/original_market_stability/summary.json` |

## B. Replication

1. `test_original_market_stability.py`: passed.
2. `hedging_performance_experiment.py` (50k paths, 512 steps, seed 20260907):
   variance reduction rerun / stored = GKW R=80 unshifted 0.8280 / 0.8244,
   BS delta 0.7324 / 0.7265 (`results/replication/` vs the stored files in the submodule, `external/hedging_rough_hawkes_heston/numerics/output/`).
3. `original_market_stability.py` rerun (167 s): deterministic outputs (Riccati
   errors, kernel L1 errors) identical to the paper; capitals equal to ~1e-16.
   Monte Carlo outputs differ by about one standard error (e.g. 1e6 d^2 at
   eps/T = 1/4: 2.30 vs 2.447 +- 0.173).
4. **Why the Monte Carlo numbers differ with the same seed.** In both scripts every
   deterministic quantity matches, while path statistics (jump counts 188 vs 189,
   payoff mean shifted by 0.45 SE) differ. The random stream is therefore not
   the one the authors used. A plausible but **unverified** cause: NumPy does not
   guarantee identical `Generator` streams across versions. The differences are
   consistent with Monte Carlo noise.

## C. Our code (`src/`) against independent references (`tests/test_discrete.py`)

- Rebalancing every step reproduces the upstream GKW gain to < 1e-12 (unshifted and
  shifted kernels) and the upstream BS gain to < 1e-12.
- End to end, 2,000 paths: our every-step, zero-cost variance reductions equal the
  upstream script's output to 15 significant digits (GKW 0.855523046707795,
  BS 0.833213370532653; `results/discrete_rebalancing_smoke/`).
- Rule implementations equal brute-force loops; degenerate cases (threshold
  +inf / -inf, band 0) reduce to the expected fixed rules; positions up to t_k do
  not change when data after t_k is altered (predictability).
- Static hedge gain equals theta_0 (S_T - S_0); costs equal a hand calculation.
- Black-Scholes/GBM check: the discrete-hedging error variance matches the
  leading-order formula Var(e) ~ sum_k 0.5 E[Gamma^2 S^4] sigma^4 h^2 within 12%
  for >= 32 dates, with a log-log slope in h close to 1.

## D. Numerical accuracy of the GKW holdings

- GKW holdings are **not** confined to [0, 1]. About 1.2% of (path, time) values
  in the first 75% of the life are below -0.01. This fraction is stable when R
  goes from 160 to 320 (N = 2048), so it is a property of the hedge (negative
  vega/correlation correction for out-of-the-money calls) rather than only
  truncation. The interpretation is plausible but **not proven**.
- At the upstream cutoff R = 80, holdings carry truncation error growing toward
  maturity. The maximum gap to R = 320 (N = 2048, 200 paths) is 0.004 at t = 0,
  0.04 at T/2 and 0.2-0.35 in the last 10% of the life. The capital converges
  much faster.
- Quadrature (4 vs 12 Gauss points per panel) changes holdings by < 2e-6 until
  k/N ~ 0.9 and by up to 0.04 at the last step, so the cutoff is the main error.
- The explicit Riccati scheme diverges for R >= 160 at N = 128 and for R = 320 at
  N = 512. The upstream choice R <= 80 is therefore constrained by stability.
- Effect on costs (300 paths): rebalancing turnover changes only 1.5% between
  R = 80 and R = 160. The full experiment carries a `GKW_R160` (order 8)
  variant to check this at scale: its variance reductions differ from R = 80 by
  at most ~0.01 at every frequency (section F). **Conclusions do not depend on the
  truncation.**

## E. Variance proxy for the regime-aware rule

- The upstream simulation advances integrated variance A with a running maximum
  (Richard-Tan-Yang scheme). A is flat on **62%** of fine steps at N = 512 (our
  pilot: 0.620; upstream stored diagnostic: 0.618). A one-step spot-variance
  proxy is therefore zero most of the time, and its median is exactly 0.
- Proxy used: trailing realised variance over T/16 (32 steps at N = 512), known at t_k.
  It is zero on 4.4% of points. Thresholds are calibrated on an independent
  2,000-path pilot sample (median 0.00335, 75th percentile 0.01216).
- **Idealisation:** the proxy uses the model's own integrated variance. A real
  hedger would have to estimate it from prices.
- **Caveat for interpretation:** in this calibration and scheme, variance is
  intermittent (often exactly zero). That favours rules which skip trading in
  quiet periods, so the size of the gains below may be specific to this
  model/scheme.

## F. Discrete-rebalancing results (`results/discrete_rebalancing/`, 50k paths, seed 20260907)

Setup: N = 512 fine steps, T = 0.25, ATM call, upstream parameters. Hedges are
GKW (R = 80, upstream), GKW_R160 (accuracy check) and BS delta (implied vol
0.0977). Rules: 18 fixed frequencies (every m in {1,2,3,4,6,...,384,512} steps),
8 variance-triggered and 6 no-trade bands. Costs kappa in {0, 1, 5, 10} bp of
traded notional, including the initial purchase and terminal liquidation. Every
rule and hedge uses the same paths (common random numbers). Runtime 13 min.

Built-in consistency check: the "every step, kappa = 0" rows equal the upstream
script's numbers on the same paths (VR 0.828 GKW, 0.732 BS).

1. **Rebalancing frequency (kappa = 0).** Variance reduction falls as
   rebalancing becomes coarser. GKW: 0.828 (every step), 0.806 (64 dates),
   0.738 (16), 0.583 (4), 0.374 (static). BS: 0.732, 0.670, 0.541, 0.333,
   0.108. The GKW advantage over BS **grows** as rebalancing gets coarser
   (+0.10 every step, +0.25 at 4 dates). The excess variance over every-step
   hedging grows roughly linearly in the rebalancing interval: fitted log-log
   slopes are 0.93 (BS) and 1.19 (GKW) in the main run, but 0.75-1.42 across
   runs, so the exponent is **not precisely estimated**.
2. **GKW vs BS with costs.** At each hedge's own best rule, GKW has lower MSE
   at every kappa (paired z = -5.1, -5.3, -5.6, -6.2 for kappa = 0, 1, 5,
   10 bp).
3. **Smart rules vs fixed frequency at equal expected cost.** Each band or
   variance rule is compared with the randomised mixture of the two
   neighbouring fixed rules that has the same expected traded notional (a valid
   strategy; paired, no selection; `analysis.matched_turnover`).
   - Low turnover (bands 0.1-0.2; variance rules with calm interval >= 32
     steps): at kappa = 10 bp the rule beats fixed frequency at equal expected
     cost by **6-31% in MSE** in the main run (z -2.5 to -13.1). On the
     independent holdout (seed 1234567, 20k paths) the gains are 4-35% with z
     -1.0 to -6.8, **except one rule with no gain** (BS, calm = 64, median
     threshold: +0.2%, z = 0.0).
   - High turnover (bands <= 0.02 shares): no gain. The difference is within
     +-2% and |z| <= 1.4 in both runs.
   - **At the cost-optimal point the gain is small.** At kappa = 10 bp, best
     rule vs best fixed rule (best-in-family, so mildly optimistic):
     - GKW: band 0.05 vs every 6 steps, -1.2% MSE (z = -0.5; holdout z = -1.9).
       **Not significant.**
     - BS: band 0.2 vs every 4 steps, -5.9% MSE (z = -1.7; holdout z = -3.8).
       **Borderline.**

     At 5 bp: GKW -0.8% (z -0.4), BS -3.7% (z -1.4). With costs of a few bp
     the optimum trades often, where all rules nearly coincide. Adapting the
     timing matters more for the BS delta than for the GKW hedge.
   - Plain-language summary: adapting the trading times helps a lot when the
     hedger must trade rarely (high costs or a turnover budget). It barely
     matters when frequent trading is affordable.
4. **Robustness.**
   - Holdout seed: section F3.
   - GKW_R160 vs GKW R = 80: same conclusions, variance reductions within ~0.01.
   - Coarser simulation grid (N = 256, 20k paths): at the same rebalancing
     interval, variance reductions are 0.04-0.05 lower than at N = 512 (e.g. 64
     dates: 0.756 vs 0.806). That is about 1.3-1.7 unpaired SE, similar to the
     seed-to-seed spread (every-step GKW 0.828 main vs 0.793 holdout). A grid
     effect cannot be separated from noise at this sample size.
   - Unpaired standard errors of variance reductions are large (GKW about 0.012
     at 50k paths, BS about 0.03) because jumps give heavy tails. Use paired
     comparisons for any claim.

## G. Known limitations (state these in the report)

- The hedge is the continuous-time GKW strategy, held piecewise constant. It is
  not the optimal discrete-time hedge, which would also adjust for the
  rebalancing interval.
- Everything is under the risk-neutral measure Q, as in the paper (no drift and
  no historical-measure effects).
- Single calibration, maturity (3 months) and strike (ATM).
- Costs are proportional only. There are no fixed fees and no market impact.
- The variance proxy is idealised (section E). The size of the gains may depend
  on the intermittency of the simulation scheme.
