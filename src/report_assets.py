"""Generate the report's numbers, tables and figures from the result files.

Every number quoted in report/report.tex comes from here, so the text cannot
drift from the results. Rerun after any experiment:

    python -m src.report_assets && (cd report && pdflatex report && pdflatex report)

Writes
  report/numbers.tex          \\newcommand macros, one per quoted number
  report/tables/*.tex         booktabs tables (\\input in the report)
  report/figures/*.pdf        copies of result figures (report/ is self-contained)

Sections are skipped (with a printed note) when their results do not exist yet.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .upstream import PROJECT_ROOT, UPSTREAM_NUMERICS

R = PROJECT_ROOT / "results"
REPORT = PROJECT_ROOT / "report"
# The authors' stored outputs, read from the unmodified submodule (commit c6a6557).
UPSTREAM_OUTPUT = UPSTREAM_NUMERICS / "output"
MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names must be letters only: {name}")
    MACROS[name] = str(value)


def f(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def pct(x: float, digits: int = 1, sign: bool = True) -> str:
    r"""Percentage; signed values get a true minus sign (\ensuremath works in text and math)."""
    if sign:
        return rf"\ensuremath{{{x:+.{digits}f}}}\%"
    return rf"{x:.{digits}f}\%"


def zf(x: float, digits: int = 1) -> str:
    """Signed statistic (z values) with a true minus sign."""
    return rf"\ensuremath{{{x:.{digits}f}}}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def write_table(name: str, body: str) -> None:
    (REPORT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT / "tables" / f"{name}.tex").write_text(body)


def copy_figure(src: Path, name: str) -> None:
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, REPORT / "figures" / name)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# 1. Replication
# ---------------------------------------------------------------------------


def replication():
    stored = {r["strategy"]: r for r in read_csv(UPSTREAM_OUTPUT / "hedging_performance_experiment" / "terminal_mse.csv")}
    rerun = {r["strategy"]: r for r in read_csv(R / "replication" / "hedging_performance_experiment" / "terminal_mse.csv")}
    # Only the two hedges we build on; the approximation rows (kernel shift and
    # Fourier cutoff) are in results/replication/ and notes/robustness_log.md.
    rows = [("Variance-optimal (GKW) hedge", "Unshifted-kernel hedge, R=80"),
            ("Black--Scholes delta (implied vol)", "Black--Scholes delta (implied-vol calibrated)")]
    lines = []
    for label, key in rows:
        a, b = stored[key], rerun[key]
        lines.append(f"{label} & {f(float(a['variance_reduction_vs_no_hedge']), 3)} & "
                     f"{f(float(b['variance_reduction_vs_no_hedge']), 3)} \\\\")
    write_table("replication", "\\begin{tabular}{lcc}\n\\toprule\n"
                "Hedge & Official output & Our rerun \\\\\n\\midrule\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    macro("RepGKW", f(float(rerun["Unshifted-kernel hedge, R=80"]["variance_reduction_vs_no_hedge"]), 3))
    macro("RepBS", f(float(rerun["Black--Scholes delta (implied-vol calibrated)"]["variance_reduction_vs_no_hedge"]), 3))
    macro("RepGKWStored", f(float(stored["Unshifted-kernel hedge, R=80"]["variance_reduction_vs_no_hedge"]), 3))
    macro("RepBSStored", f(float(stored["Black--Scholes delta (implied-vol calibrated)"]["variance_reduction_vs_no_hedge"]), 3))


# ---------------------------------------------------------------------------
# 2. Simulation: discrete rebalancing
# ---------------------------------------------------------------------------


def simulation():
    from . import analysis as an
    d = R / "discrete_rebalancing"
    a = json.loads((d / "analysis.json").read_text())
    cfg = a["config"]
    macro("SimPaths", thousands(cfg["paths"]))
    macro("SimSteps", cfg["steps"])
    freq = a["frequency_kappa0"]
    show = (1, 8, 32, 128, 512)
    lines = []
    for every in show:
        g = next(r for r in freq["GKW"]["rows"] if r["every"] == every)
        b = next(r for r in freq["BS"]["rows"] if r["every"] == every)
        dates = math.ceil(cfg["steps"] / every)
        lines.append(f"{dates} & {f(g['variance_reduction'], 3)} ({f(g['variance_reduction_se'], 3)}) & "
                     f"{f(b['variance_reduction'], 3)} ({f(b['variance_reduction_se'], 3)}) \\\\")
    write_table("sim_frequency", "\\begin{tabular}{rcc}\n\\toprule\n"
                "Rebalancing dates & GKW hedge & BS delta \\\\\n\\midrule\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    zs = [a["best_rules"][k]["GKW_minus_BS_mse_at_best_rules"] for k in ("0.0", "0.0001", "0.0005", "0.001")]
    zvals = [z["value"] / z["se"] for z in zs]
    macro("SimGKWvsBSzMin", zf(max(zvals)))  # least negative
    macro("SimGKWvsBSzMax", zf(min(zvals)))
    # Equal-turnover comparisons at kappa = 10 bp, main run and holdout.
    def low_turnover_changes(directory: Path):
        data, config, an.COMBOS = an.load(directory)
        out = []
        for hedge in ("GKW", "BS"):
            for (h, rule) in an.COMBOS:
                if h != hedge or an.family(rule) == "fixed":
                    continue
                fam = an.family(rule)
                low = (fam == "band" and float(rule.split("=")[1]) >= 0.1) or \
                      (fam == "variance" and int(rule.split("calm=")[1].split(",")[0]) >= 32)
                if not low:
                    continue
                m = an.matched_turnover(data, config, hedge, rule, 1e-3)
                e = an.pnl(data, config, hedge, rule, 1e-3)
                mse = float(np.mean(e**2))
                out.append(100 * m["mse_minus_matched_fixed"] / (mse - m["mse_minus_matched_fixed"]))
        return out
    main = low_turnover_changes(d)
    hold = low_turnover_changes(R / "robustness" / "seed_1234567")
    macro("SimLowGainMin", pct(-max(main), 0, sign=False))
    macro("SimLowGainMax", pct(-min(main), 0, sign=False))
    macro("SimLowHoldoutAgree", f"{sum(x < 0 for x in hold)}/{len(hold)}")
    best = a["best_rules"]["0.001"]
    for hedge, tag in (("GKW", "GKW"), ("BS", "BS")):
        fam = min(("band", "variance"), key=lambda k: best[hedge][k]["mse"])
        v = best[hedge][fam]
        mse_fixed = v["mse"] - v["mse_minus_best_fixed"]
        macro(f"SimOpt{tag}Change", pct(100 * v["mse_minus_best_fixed"] / mse_fixed))
        macro(f"SimOpt{tag}z", zf(v["mse_minus_best_fixed"] / v["se"]))
    copy_figure(d / "fig_frequency.pdf", "sim_frequency.pdf")
    copy_figure(d / "fig_frontier.pdf", "sim_frontier.pdf")


# ---------------------------------------------------------------------------
# 3. Data and regimes
# ---------------------------------------------------------------------------


def data_and_regimes():
    rep = json.loads((R / "data_checks" / "report.json").read_text())
    macro("DataNiftyDays", thousands(rep["nifty"]["rows"]))
    macro("DataYahooMissing", rep["nifty"]["days_missing_in_yahoo"])
    macro("DataVixVerified", thousands(rep["vix"]["rows_verified"]))
    macro("DataVixMonths", rep["vix"]["months_checked_vs_nse"])
    s = json.loads((R / "regimes" / "summary.json").read_text())
    p = s["params"]
    macro("RegVixQ", f(p["vix_q75"], 2))
    macro("RegVixQlo", f(p["vix_q60"], 2))
    macro("RegRvQ", f(100 * p["rv21_q75"], 1))
    ms = p["markov_switching"]
    macro("RegMSsigCalm", f(ms["sigma_pct"][0], 2))
    macro("RegMSsigStress", f(ms["sigma_pct"][1], 2))
    variants = [("Main (train to 2018)", R / "regimes"),
                ("Train to 2017", R / "regimes_robustness" / "split_2017"),
                ("Train to 2019", R / "regimes_robustness" / "split_2019"),
                ("Start 2008 (unverified VIX)", R / "regimes_robustness" / "start_2008_unverified")]
    lines = []
    for label, path in variants:
        t = json.loads((path / "summary.json").read_text())["diagnostics"]["test"]
        cells = [f"{f(t[m]['fwd_rv5_stress_over_calm'][0], 2)} [{f(t[m]['fwd_rv5_stress_over_calm'][1], 2)}, "
                 f"{f(t[m]['fwd_rv5_stress_over_calm'][2], 2)}]" for m in ("vix_hyst", "rv21_q75")]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    # The Markov-switching label is a robustness check; it is quoted in the text only.
    t_main = json.loads((R / "regimes" / "summary.json").read_text())["diagnostics"]["test"]
    ms = t_main["ms_filter"]["fwd_rv5_stress_over_calm"]
    macro("RegMSRatio", f(ms[0], 2))
    macro("RegMSRatioLo", f(ms[1], 2))
    write_table("regimes", "\\begin{tabular}{lcc}\n\\toprule\n"
                "Variant & VIX (hysteresis) & 21-day realised vol \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    copy_figure(R / "regimes" / "fig_regimes.pdf", "regimes.pdf")


# ---------------------------------------------------------------------------
# 4. Backtests (synthetic options; later: real options)
# ---------------------------------------------------------------------------


def backtest_table(name: str, variants: list[tuple[str, Path]], prefix: str):
    lines = []
    for label, path in variants:
        a = json.loads(path.read_text())
        k5, k20 = a["kappas"]["5.0"], a["kappas"]["20.0"]
        r5 = k5["train_to_test"]["regime_vs_fixed_on_test"]
        r20 = k20["train_to_test"]["regime_vs_fixed_on_test"]
        sb = k5["placebo_scoreboard"]
        zs = [m["z"] for m in k5["matched_turnover_test"]
              if m["rule"].startswith("regime[") and ":" not in m["rule"].split("[")[1].split("]")[0]]
        lines.append(f"{label} & {pct(r5['mse_change_pct'])} ({zf(r5['z'])}) & "
                     f"{pct(r20['mse_change_pct'])} ({zf(r20['z'])}) & "
                     f"{sb['real_rules_beating_fixed']}/{sb['real_rules']} & {zf(float(np.median(zs)))} & "
                     f"{sb['real_beats_placebo']['flip']} \\\\")
    write_table(name, "\\begin{tabular}{lccccc}\n\\toprule\n"
                " & \\multicolumn{2}{c}{Picked on train vs daily} & \\multicolumn{3}{c}{Equal turnover, 5\\,bp} \\\\\n"
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
                "Variant & 5\\,bp & 20\\,bp & beat fixed & med.\\ $z$ & beat placebo \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    main = json.loads(variants[0][1].read_text())
    macro(f"{prefix}Options", thousands(main["options"]["test"]))
    macro(f"{prefix}Premium", f(main["mean_premium_bp"]["test"], 0))


def backtest_synthetic():
    d = R / "backtest_real"
    backtest_table("backtest_synthetic", [
        ("21-day, full test", d / "maturity_21" / "analysis.json"),
        ("21-day, excl.\\ COVID window", d / "maturity_21" / "analysis_excl_covid.json"),
        ("63-day options", d / "maturity_63" / "analysis.json")], "Syn")
    rows = {r["rule"]: r for r in read_csv(d / "maturity_21" / "rules_test.csv") if r["kappa_bp"] == "0.0"}
    macro("SynMeanPnl", f(float(rows["fixed/every=1"]["mean_pnl_bp"]), 0))
    copy_figure(d / "maturity_21" / "fig_frontier_test.pdf", "synthetic_frontier.pdf")


def backtest_options():
    d = R / "backtest_options"
    paths = [("Own implied vol", d / "own" / "analysis.json"),
             ("Own IV, excl.\\ COVID window", d / "own" / "analysis_excl_covid.json"),
             ("India VIX delta", d / "vix" / "analysis.json")]
    paths = [(lab, p) for lab, p in paths if p.exists()]
    if not paths:
        print("skip: real-option backtest results not found")
        return
    backtest_table("backtest_options", paths, "Opt")
    fig = d / "own" / "fig_frontier_test.pdf"
    if fig.exists():
        copy_figure(fig, "options_frontier.pdf")
    # Data checks on the option archive.
    rep = json.loads((R / "data_checks" / "fo_report.json").read_text())
    macro("FoDays", thousands(rep["days"]))
    macro("FoExpiries", rep["expiry_settle_vs_index"]["expiries"])
    pcp = rep["put_call_parity_near_atm"]
    macro("FoParityPairs", thousands(pcp["pairs"]))
    macro("FoParityBand", f(max(abs(pcp["q01_bp"]), abs(pcp["q99_bp"])), 0))
    macro("FoIVcorr", f(rep["atm_iv_vs_india_vix"]["corr"], 2))
    bounds = rep["call_bounds_violations"]
    macro("FoAtmViolPct", f(100 * bounds["of_which_near_atm_3pct"] / bounds["near_atm_traded_calls"], 2))
    # Daily-hedging level, cost-free mean, and own-IV vs VIX delta (paired).
    from . import backtest_analysis as ba
    from .metrics import block_bootstrap_se
    own, vix = ba.load(d / "own", "test"), ba.load(d / "vix", "test")
    if not (own["start_dates"] == vix["start_dates"]).all():
        raise ValueError("own-IV and VIX runs must contain the same options")
    rows = {r["rule"]: r for r in read_csv(d / "own" / "rules_test.csv") if r["kappa_bp"] == "0.0"}
    macro("OptDailyJ", f(float(rows["fixed/every=1"]["rmse_adj_bp"]), 1))
    macro("OptMeanPnl", f(float(rows["fixed/every=1"]["mean_pnl_bp"]), 0))
    syn = {r["rule"]: r for r in read_csv(R / "backtest_real" / "maturity_21" / "rules_test.csv")
           if r["kappa_bp"] == "0.0"}
    macro("SynDailyJ", f(float(syn["fixed/every=1"]["rmse_adj_bp"]), 1))
    a = ba.adjusted(own, "fixed/every=1", 0.0) ** 2
    b = ba.adjusted(vix, "fixed/every=1", 0.0) ** 2
    diff = a - b
    macro("OptOwnVsVix", pct(100 * diff.mean() / b.mean()))
    macro("OptOwnVsVixz", zf(diff.mean() / block_bootstrap_se(diff, 42)))


def model_parameters():
    """Parameter table straight from the upstream Model dataclass."""
    from .upstream import base
    m = base.Model()
    rows = [(r"$\alpha$", m.alpha, "roughness of the kernel $K(t)=t^{\\alpha-1}/\\Gamma(\\alpha)$"),
            (r"$\rho$", m.rho, "correlation between price and variance shocks (leverage)"),
            (r"$b$", m.b, "mean reversion of the variance (negative = pulls back)"),
            (r"$c$", m.c, "vol-of-vol: size of Brownian shocks to the variance"),
            (r"$\Lambda$", m.jump_leverage, "a jump of mark $z$ moves the log-price by $-\\Lambda z$"),
            (r"$\beta$", m.beta0, "slope of the initial curve $g_0(t)=V_0+\\beta t^\\alpha/\\Gamma(\\alpha+1)$"),
            (r"$V_0$", m.v0, "initial variance"),
            (r"$\kappa$, $m$", f"{m.jump_intensity:g}, {m.jump_mean:g}",
             "jump marks: $\\nu(dz)=\\kappa e^{-z/m}dz/m$"),
            (r"$S_0$", m.s0, "initial price (prices are relative to $S_0$)")]
    lines = [f"{sym} & ${val}$ & {desc} \\\\" for sym, val, desc in rows]  # $..$: true minus signs
    write_table("model_params", "\\begin{tabular}{llp{8.2cm}}\n\\toprule\n"
                "Parameter & Value & Meaning \\\\\n\\midrule\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    macro("ParAlpha", m.alpha)
    macro("ParHurst", f(m.alpha - 0.5, 3))
    macro("ParVzero", m.v0)
    macro("ParVolZero", f(100 * math.sqrt(m.v0), 1))
    stab = json.loads((UPSTREAM_OUTPUT / "original_market_stability" / "summary.json").read_text())
    macro("SimFlatSteps", f(100 * stab["fine_grid"]["running_max_adjustment_fraction"], 0))


def implied_vs_realised():
    """Gamma-P&L check: did implied volatility exceed the volatility that followed?

    A delta hedge run at implied vol sigma_i earns (1/2) Gamma S^2 (sigma_i^2 -
    sigma_r^2) dt, where sigma_r is realised vol. The seller's average gain is
    therefore explained only if implied exceeded realised. Two comparisons on the
    test period (2019-2026):
      * India VIX at t vs realised vol over the next 21 trading days (t+1..t+21);
      * each traded option's implied vol at its start vs NIFTY's realised vol
        over that option's life.
    Variance ratios mean(sigma_i^2)/mean(sigma_r^2) are reported with and without
    the COVID window used in the backtests (option starts 2020-01-15..2020-06-30).
    """
    import pandas as pd
    from . import regimes as rg
    covid = ("2020-01-15", "2020-06-30")
    lab = pd.read_csv(R / "regimes" / "regime_labels.csv", index_col=0, parse_dates=True)
    fwd = rg.forward_rv(lab["logret"], 21)
    d = pd.DataFrame({"iv": lab["vix"] / 100, "rv": fwd})[lab["split"] == "test"].dropna()
    in_covid = (d.index >= covid[0]) & (d.index <= covid[1])
    macro("IvrVixMean", f(100 * d["iv"].mean(), 1))
    macro("IvrVixRvMean", f(100 * d["rv"].mean(), 1))
    macro("IvrVixAbove", f(100 * (d["iv"] > d["rv"]).mean(), 0))
    ratio = lambda x: float((x["iv"] ** 2).mean() / (x["rv"] ** 2).mean())
    macro("IvrVixVarRatioEx", f(ratio(d[~in_covid]), 2))
    macro("IvrVixVarRatioCovid", f(ratio(d[in_covid]), 2))
    # Traded options: implied vol at the start vs realised vol over the life.
    o = np.load(R / "backtest_options" / "own" / "per_option_test.npz")
    logret = np.log(pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nifty50.csv",
                                index_col=0, parse_dates=True)["close"]).diff()
    rows = []
    for start, n, iv in zip(o["start_dates"], o["n_days"], o["iv0"]):
        i = logret.index.get_loc(pd.Timestamp(str(start)))
        life = logret.iloc[i + 1:i + 1 + int(n)]  # returns strictly after the sale
        rows.append((str(start), iv, math.sqrt(252 * float(np.mean(life**2)))))
    t = pd.DataFrame(rows, columns=["start", "iv", "rv"])
    t_covid = (t["start"] >= covid[0]) & (t["start"] <= covid[1])
    macro("IvrOptIvMean", f(100 * t["iv"].mean(), 1))
    macro("IvrOptRvMean", f(100 * t["rv"].mean(), 1))
    macro("IvrOptAbove", f(100 * (t["iv"] > t["rv"]).mean(), 0))
    macro("IvrOptVarRatioEx", f(ratio(t[~t_covid]), 2))
    macro("IvrOptVarRatioCovid", f(ratio(t[t_covid]), 2))


def main():
    for step in (model_parameters, replication, simulation, data_and_regimes,
                 backtest_synthetic, backtest_options, implied_vs_realised):
        try:
            step()
        except FileNotFoundError as error:
            print(f"skip {step.__name__}: {error}")
    lines = ["% Generated by src/report_assets.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(MACROS)} macros, tables: {sorted(p.name for p in (REPORT / 'tables').glob('*.tex'))}")


if __name__ == "__main__":
    main()
