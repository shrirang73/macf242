"""Paired analysis of the discrete-rebalancing experiment.

Reads ``per_path.npz`` written by ``experiment_discrete.py`` and produces:

  analysis.json        key numbers, each with a paired Monte Carlo SE
  fig_frequency.pdf/png   variance reduction vs number of rebalancing dates (kappa = 0)
  fig_frontier.pdf/png    P&L std vs expected cost for every rule (kappa > 0)

Paired SEs: all hedges and rules were evaluated on the same paths, so for a
difference of two statistics we take the SE of the pathwise difference. This
is much smaller than combining two unpaired SEs, and it is the honest SE for
"is rule A better than rule B on this model".

Usage: python -m src.analysis results/discrete_rebalancing [--holdout DIR]
``--holdout`` points at an independent run (different seed): the rule that is
best in the main run is re-evaluated there, which guards against selection bias.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (light surface, print): slot 1 blue, slot 2 orange; ink and
# hairlines from the same reference. Identity is also carried by marker shape.
COLORS = {"GKW": "#2a78d6", "BS": "#eb6834", "GKW_R160": "#2a78d6"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MARKERS = {"fixed": "o", "variance": "^", "band": "s"}
NAMES = {"GKW": "Variance-optimal (GKW, R=80)", "BS": "Black–Scholes delta",
         "GKW_R160": "GKW, R=160 (accuracy check)"}


def load(directory: Path):
    data = np.load(directory / "per_path.npz")
    config = json.loads((directory / "config.json").read_text())
    combos = sorted({tuple(k.split("|")[1:]) for k in data.files if k.startswith("gain|")})
    return data, config, combos


def pnl(data, config, hedge, rule, kappa):
    """Short-call P&L per path: x + gain - kappa*notional - payoff."""
    capital = config["gkw_r160_capital"] if hedge == "GKW_R160" else config["gkw_capital"]
    tag = f"{hedge}|{rule}"
    return (capital + data[f"gain|{tag}"] - kappa * data[f"notional|{tag}"].astype(float)
            - data["payoff"])


def mean_se(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def centered_sq(e):
    return (e - e.mean()) ** 2


def family(rule: str) -> str:
    return rule.split("/")[0]


def every(rule: str) -> int:
    return int(rule.split("=")[1])


def frequency_table(data, config, hedges):
    """kappa = 0: variance reduction and excess variance over every-step hedging."""
    steps = config["steps"]
    base_sq = centered_sq(data["payoff"])
    out = {}
    for h in hedges:
        fixed = sorted((r for (hh, r) in COMBOS if hh == h and family(r) == "fixed"), key=every)
        ref = centered_sq(pnl(data, config, h, "fixed/every=1", 0.0))
        rows = []
        for r in fixed:
            sq = centered_sq(pnl(data, config, h, r, 0.0))
            ratio = sq.mean() / base_sq.mean()
            infl = (sq - ratio * base_sq) / base_sq.mean()
            excess, excess_se = mean_se(sq - ref)
            rows.append({"every": every(r), "dates": math.ceil(steps / every(r)),
                         "variance_reduction": 1 - ratio,
                         "variance_reduction_se": float(np.std(infl, ddof=1) / math.sqrt(len(sq))),
                         "excess_variance_vs_every_step": excess,
                         "excess_variance_se": excess_se})
        # Rate: Var(m) - Var(1) ~ C * (m h)^p, fitted on 2 <= m <= steps/8.
        pts = [(row["every"], row["excess_variance_vs_every_step"]) for row in rows
               if 2 <= row["every"] <= steps // 8 and row["excess_variance_vs_every_step"] > 0]
        slope = (float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
                 if len(pts) >= 3 else None)
        out[h] = {"rows": rows, "loglog_slope_excess_variance": slope}
    return out


def paired(a, b):
    """Mean and SE of a - b (pathwise)."""
    return mean_se(a - b)


def best_rules(data, config, hedges, kappas):
    """Per kappa: best rule in each family by E[e^2] (capital = GKW Fourier capital)."""
    result = {}
    for kappa in kappas:
        per = {}
        for h in hedges:
            fam_best = {}
            for (hh, r) in COMBOS:
                if hh != h:
                    continue
                e = pnl(data, config, h, r, kappa)
                mse = float(np.mean(e**2))
                f = family(r)
                if f not in fam_best or mse < fam_best[f][1]:
                    fam_best[f] = (r, mse)
            # Paired comparison of each family's best with the best fixed rule.
            e_fixed = pnl(data, config, h, fam_best["fixed"][0], kappa)
            entry = {}
            for f, (r, mse) in fam_best.items():
                e = pnl(data, config, h, r, kappa)
                d, d_se = paired(e**2, e_fixed**2)
                entry[f] = {"rule": r, "rmse": math.sqrt(mse), "mse": mse,
                            "mse_minus_best_fixed": d, "se": d_se,
                            "mean_cost": float(kappa * data[f"notional|{h}|{r}"].mean()),
                            "mean_trades": float(data[f"trades|{h}|{r}"].mean())}
            per[h] = entry
        # GKW vs BS, each at its own best rule overall.
        best = {h: min(per[h].values(), key=lambda v: v["mse"])["rule"] for h in hedges}
        if "GKW" in hedges and "BS" in hedges:
            eg = pnl(data, config, "GKW", best["GKW"], kappa)
            eb = pnl(data, config, "BS", best["BS"], kappa)
            d, d_se = paired(eg**2, eb**2)
            per["GKW_minus_BS_mse_at_best_rules"] = {"value": d, "se": d_se,
                                                     "rules": best}
        result[str(kappa)] = per
    return result


def canonical(rule: str, config: dict) -> str:
    """Rule name without run-specific numbers: variance thresholds are
    recalibrated per run, so 'thr=<value>' is replaced by its quantile level."""
    if family(rule) != "variance":
        return rule
    head, thr = rule.rsplit(",thr=", 1)
    levels = config["variance_thresholds"]
    q = min(levels, key=lambda key: abs(levels[key] - float(thr)))
    return f"{head},q={q}"


def matched_turnover(data, config, hedge, rule, kappa):
    """Compare ``rule`` with fixed-frequency hedging at the SAME expected turnover.

    Take the two fixed rules a, b whose mean traded notional brackets that of
    ``rule``, and the randomised strategy "use a with probability p, else b",
    with p chosen so that expected notionals (hence expected costs) are equal.
    Its E[e^2] is p E[e_a^2] + (1-p) E[e_b^2], so the pathwise difference
    e_rule^2 - (p e_a^2 + (1-p) e_b^2) gives an unbiased paired estimate.
    Negative = ``rule`` beats fixed-frequency hedging at equal cost.
    """
    n_rule = float(data[f"notional|{hedge}|{rule}"].mean())
    fixed = sorted(((float(data[f"notional|{hedge}|{r}"].mean()), r) for (h, r) in COMBOS
                    if h == hedge and family(r) == "fixed"))
    lower = [x for x in fixed if x[0] <= n_rule]
    upper = [x for x in fixed if x[0] > n_rule]
    if not lower or not upper:
        return None
    (n_a, a), (n_b, b) = lower[-1], upper[0]
    p = (n_b - n_rule) / (n_b - n_a)  # weight on the lower-turnover rule a
    e = pnl(data, config, hedge, rule, kappa)
    ea, eb = pnl(data, config, hedge, a, kappa), pnl(data, config, hedge, b, kappa)
    d, d_se = mean_se(e**2 - (p * ea**2 + (1 - p) * eb**2))
    return {"rule": rule, "mixture": f"{p:.2f}*{a} + {1 - p:.2f}*{b}",
            "mean_notional": n_rule, "mse_minus_matched_fixed": d, "se": d_se,
            "z": d / d_se if d_se > 0 else 0.0}


def matched_table(data, config, hedges, kappas):
    out = {}
    for kappa in kappas:
        out[str(kappa)] = {}
        for h in hedges:
            rows = []
            for (hh, r) in COMBOS:
                if hh == h and family(r) != "fixed":
                    m = matched_turnover(data, config, h, r, kappa)
                    if m is not None:
                        m["canonical"] = canonical(r, config)
                        rows.append(m)
            out[str(kappa)][h] = rows
    return out


def holdout_check(best, holdout_dir: Path, main_config: dict):
    """Re-evaluate the main run's best rules (and the matched-turnover
    comparisons) on an independent run with a different seed."""
    data, config, combos = load(holdout_dir)
    global COMBOS
    saved, COMBOS = COMBOS, combos
    by_canonical = {(h, canonical(r, config)): r for (h, r) in combos}
    out = {"best_rules": {}, "matched_turnover": None}
    try:
        for kappa, per in best.items():
            k = float(kappa)
            out["best_rules"][kappa] = {}
            for h in ("GKW", "BS"):
                if h not in per:
                    continue
                fixed_rule = per[h]["fixed"]["rule"]
                e_fixed = pnl(data, config, h, fixed_rule, k)
                for f, v in per[h].items():
                    rule = by_canonical.get((h, canonical(v["rule"], main_config)))
                    if rule is None:
                        out["best_rules"][kappa][f"{h}/{f}"] = {"rule": v["rule"], "missing": True}
                        continue
                    e = pnl(data, config, h, rule, k)
                    d, d_se = paired(e**2, e_fixed**2)
                    out["best_rules"][kappa][f"{h}/{f}"] = {
                        "rule": rule, "rmse": math.sqrt(float(np.mean(e**2))),
                        "mse_minus_main_best_fixed": d, "se": d_se}
        out["matched_turnover"] = matched_table(data, config, ("GKW", "BS"), [0.0, 1e-3])
    finally:
        COMBOS = saved
    return out


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def plot_frequency(freq, out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for h in ("GKW", "BS"):
        rows = freq[h]["rows"]
        # N/m (fine steps per rebalance interval); distinct for every m, unlike
        # the integer count of dates (m = 256 and m = 384 both give 2 dates).
        x = np.array([STEPS_FOR_PLOT / r["every"] for r in rows])
        y = np.array([r["variance_reduction"] for r in rows])
        se = np.array([r["variance_reduction_se"] for r in rows])
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=COLORS[h], alpha=0.15, linewidth=0)
        ax.plot(x, y, "-o", color=COLORS[h], lw=1.5, ms=4.5, label=NAMES[h])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rebalancing frequency N/m (rebalances per option life, log scale)", color=INK2)
    ax.set_ylabel("variance reduction vs no hedge", color=INK2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_frequency.{ext}", dpi=200)
    plt.close(fig)


def plot_frontier(data, config, kappas, out: Path):
    kappas = [k for k in kappas if k > 0]
    capital = config["gkw_capital"]
    fig, axes = plt.subplots(1, len(kappas), figsize=(3.3 * len(kappas), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kappa in zip(axes, kappas):
        for h in ("GKW", "BS"):
            for f in ("fixed", "variance", "band"):
                pts = []
                for (hh, r) in COMBOS:
                    if hh == h and family(r) == f:
                        e = pnl(data, config, h, r, kappa)
                        cost = kappa * data[f"notional|{h}|{r}"].mean()
                        pts.append((cost / capital, e.std() / capital))
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, MARKERS[f], color=COLORS[h], ms=5, mfc="none" if f != "fixed" else COLORS[h],
                        mew=1.2, linestyle="-" if f == "fixed" else "none", lw=1.0,
                        label=f"{h}, {f}")
        ax.set_title(f"cost rate κ = {kappa * 1e4:g} bp", color=INK, fontsize=10)
        ax.set_xlabel("expected cost / option price", color=INK2)
        style(ax)
    axes[0].set_ylabel("std of hedging P&L / option price", color=INK2)
    axes[-1].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_frontier.{ext}", dpi=200)
    plt.close(fig)


COMBOS: list[tuple[str, str]] = []
STEPS_FOR_PLOT = 512


def main(argv=None):
    global COMBOS
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--holdout", type=Path, default=None)
    args = parser.parse_args(argv)
    global STEPS_FOR_PLOT
    data, config, COMBOS = load(args.directory)
    STEPS_FOR_PLOT = config["steps"]
    hedges = sorted({h for h, _ in COMBOS})
    kappas = config["kappas"]
    freq = frequency_table(data, config, hedges)
    best = best_rules(data, config, hedges, kappas)
    report = {"config": {k: config[k] for k in ("paths", "steps", "seed", "gkw_capital",
                                                "variance_thresholds", "proxy_window_steps")},
              "frequency_kappa0": freq, "best_rules": best,
              "matched_turnover": matched_table(data, config, hedges, [0.0, 1e-3])}
    if args.holdout is not None:
        report["holdout"] = {"directory": str(args.holdout),
                             "results": holdout_check(best, args.holdout, config)}
    (args.directory / "analysis.json").write_text(json.dumps(report, indent=2))
    plot_frequency(freq, args.directory)
    plot_frontier(data, config, kappas, args.directory)
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
