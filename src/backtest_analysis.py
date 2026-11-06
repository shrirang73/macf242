"""Analysis of the real-data backtest (output of backtest_real.py).

For a short ATM call hedged with held positions phi, the P&L per unit notional is
    e = premium + sum_k phi_k (S_{k+1} - S_k) - kappa * traded notional - payoff.
Criterion. On real data the premium (BS at India VIX) is not the fair price, so
the cost-free P&L has a non-zero mean m0 (about +30 bp per option on the test
period). That mean reflects risk premia (implied above realised volatility,
index drift), not hedging quality. We therefore rank rules by

    J = E[(e - m0)^2] = Var(e) + (E[cost])^2,

where m0 is the rule's own mean P&L at zero cost. J reduces to the simulation's
E[e^2] when the premium is fair (m0 = 0). It is reported as an RMSE in basis
points of notional. The raw E[e^2] is kept in the table for reference.

Inference. Options start on consecutive days and overlap, so per-option errors
are serially dependent. Every standard error is a moving-block bootstrap over
start dates (block = 2 x maturity; see metrics.block_bootstrap_se), applied to
paired per-option differences.

Comparisons
-----------
1. Equal turnover (no selection): each regime or band rule against the
   randomised mixture of the two neighbouring fixed-frequency rules with the
   same expected traded notional (same expected cost), as in the simulation
   (analysis.matched_turnover).
2. Train-to-test protocol: for each cost level, the best fixed, best regime
   (main label) and best band rule are chosen on TRAIN by E[e^2], then compared
   on TEST.
3. Placebos: the same regime rules driven by inverted labels or by labels that
   are 252 days stale, at equal turnover.

Usage: python -m src.backtest_analysis results/backtest_real/maturity_21 [--main-label vix_hyst]
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

from .metrics import block_bootstrap_se

KAPPAS_BP = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
BP = 1e4


def load(directory: Path, period: str, exclude: tuple[str, str] | None = None) -> dict:
    """Per-option arrays; optionally drop options STARTING in [exclude[0], exclude[1]]."""
    data = np.load(directory / f"per_option_{period}.npz", allow_pickle=False)
    out = {k: data[k] for k in data.files}
    if exclude is not None:
        keep = ~((out["start_dates"] >= exclude[0]) & (out["start_dates"] <= exclude[1]))
        out = {k: v[keep] for k, v in out.items()}
    return out


def placebo_scoreboard(matched: list[dict], labels: list[str]) -> dict:
    """For every (label, calm/stress pair): is the real-label rule better (lower J
    change vs its equal-turnover fixed benchmark) than each placebo?"""
    by_rule = {m["rule"]: m for m in matched}
    real_beats = {"flip": 0, "lag252": 0}
    total = {"flip": 0, "lag252": 0}
    real_negative = n_real = 0
    for rule, m in by_rule.items():
        lab = label_of(rule)
        if lab not in labels:
            continue
        n_real += 1
        real_negative += m["mse_change_pct"] < 0
        tail = rule.split("/", 1)[1]
        for kind in ("flip", "lag252"):
            other = by_rule.get(f"regime[{kind}:{lab}]/{tail}")
            if other is not None:
                total[kind] += 1
                real_beats[kind] += m["mse_change_pct"] < other["mse_change_pct"]
    return {"real_rules": n_real, "real_rules_beating_fixed": int(real_negative),
            "real_beats_placebo": {k: f"{real_beats[k]}/{total[k]}" for k in total}}


def rules_in(data: dict) -> list[str]:
    return [k.split("|", 1)[1] for k in data if k.startswith("gain|")]


def family(rule: str) -> str:
    return rule.split("/")[0].split("[")[0]  # fixed | regime | band


def label_of(rule: str) -> str | None:
    return rule[rule.index("[") + 1: rule.index("]")] if "[" in rule else None


def pnl(data: dict, rule: str, kappa: float) -> np.ndarray:
    return (data["premium"] + data[f"gain|{rule}"] - kappa * data[f"notional|{rule}"]
            - data["payoff"])


def adjusted(data: dict, rule: str, kappa: float) -> np.ndarray:
    """P&L minus the rule's cost-free mean m0, so that mean(adjusted^2) = J."""
    return pnl(data, rule, kappa) - pnl(data, rule, 0.0).mean()


def rule_table(data: dict, kappa: float, block: int) -> list[dict]:
    rows = []
    for rule in rules_in(data):
        e = pnl(data, rule, kappa)
        sq = adjusted(data, rule, kappa) ** 2
        rows.append({
            "rule": rule, "kappa_bp": kappa * BP,
            "rmse_adj_bp": math.sqrt(sq.mean()) * BP,
            "j_se_bp2": block_bootstrap_se(sq, block) * BP**2,
            "rmse_raw_bp": math.sqrt(np.mean(e**2)) * BP,
            "mean_pnl_bp": e.mean() * BP, "std_pnl_bp": e.std() * BP,
            "mean_cost_bp": kappa * data[f"notional|{rule}"].mean() * BP,
            "mean_trades": float(data[f"trades|{rule}"].mean()),
            "mean_notional": float(data[f"notional|{rule}"].mean()),
        })
    return rows


def matched_turnover(data: dict, rule: str, kappa: float, block: int) -> dict | None:
    """Rule vs the equal-turnover mixture of its two neighbouring fixed rules."""
    n_rule = data[f"notional|{rule}"].mean()
    fixed = sorted((data[f"notional|{r}"].mean(), r) for r in rules_in(data) if family(r) == "fixed")
    lower = [x for x in fixed if x[0] <= n_rule]
    upper = [x for x in fixed if x[0] > n_rule]
    if not lower or not upper:
        return None
    (n_a, a), (n_b, b) = lower[-1], upper[0]
    p = (n_b - n_rule) / (n_b - n_a)
    e, ea, eb = (adjusted(data, r, kappa) for r in (rule, a, b))
    bench = p * ea**2 + (1 - p) * eb**2
    d = e**2 - bench
    se = block_bootstrap_se(d, block)
    return {"rule": rule, "mixture": f"{p:.2f}*{a} + {1 - p:.2f}*{b}",
            "mse_change_pct": 100 * d.mean() / bench.mean(),
            "diff_bp2": d.mean() * BP**2, "se_bp2": se * BP**2,
            "z": d.mean() / se if se > 0 else 0.0}


def best_rule(data: dict, kappa: float, predicate) -> str:
    candidates = [r for r in rules_in(data) if predicate(r)]
    return min(candidates, key=lambda r: float(np.mean(adjusted(data, r, kappa) ** 2)))


def train_to_test(train: dict, test: dict, kappa: float, main_label: str, block: int) -> dict:
    picks = {
        "fixed": best_rule(train, kappa, lambda r: family(r) == "fixed"),
        "regime": best_rule(train, kappa, lambda r: family(r) == "regime" and label_of(r) == main_label),
        "band": best_rule(train, kappa, lambda r: family(r) == "band"),
    }
    e_fixed = adjusted(test, picks["fixed"], kappa)
    out = {"picked_on_train": picks}
    for fam in ("regime", "band"):
        e = adjusted(test, picks[fam], kappa)
        d = e**2 - e_fixed**2
        se = block_bootstrap_se(d, block)
        out[f"{fam}_vs_fixed_on_test"] = {
            "mse_change_pct": 100 * d.mean() / (e_fixed**2).mean(),
            "z": d.mean() / se if se > 0 else 0.0,
            "rmse_bp": {fam: math.sqrt((e**2).mean()) * BP,
                        "fixed": math.sqrt((e_fixed**2).mean()) * BP}}
    return out


def plot_frontier(test: dict, main_label: str, kappas_bp, path: Path):
    """Test-period RMSE vs expected cost, both in bp of notional."""
    colors = {"fixed": "#2a78d6", "regime": "#eb6834", "band": "#1baf7a", "placebo": "#898781"}
    fig, axes = plt.subplots(1, len(kappas_bp), figsize=(3.4 * len(kappas_bp), 3.3), sharey=True)
    for ax, kbp in zip(np.atleast_1d(axes), kappas_bp):
        kappa = kbp / BP
        groups = {
            "fixed": [r for r in rules_in(test) if family(r) == "fixed"],
            "regime": [r for r in rules_in(test) if label_of(r) == main_label],
            "placebo": [r for r in rules_in(test) if label_of(r) == f"flip:{main_label}"],
            "band": [r for r in rules_in(test) if family(r) == "band"],
        }
        styles = {"fixed": ("o", "-"), "regime": ("^", "none"), "band": ("s", "none"),
                  "placebo": ("x", "none")}
        names = {"fixed": "fixed frequency", "regime": f"regime-aware ({main_label})",
                 "placebo": "placebo (labels inverted)", "band": "no-trade band"}
        for g, rules in groups.items():
            pts = sorted((kappa * test[f"notional|{r}"].mean() * BP,
                          math.sqrt(np.mean(adjusted(test, r, kappa) ** 2)) * BP) for r in rules)
            if not pts:
                continue
            xs, ys = zip(*pts)
            marker, line = styles[g]
            ax.plot(xs, ys, marker=marker, linestyle=line, color=colors[g], ms=5,
                    mfc=colors[g] if g == "fixed" else "none", mew=1.2, lw=1.0, label=names[g])
        ax.set_title(f"cost κ = {kbp:g} bp", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("expected cost (bp of notional)", color="#52514e")
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#898781", labelcolor="#52514e")
    np.atleast_1d(axes)[0].set_ylabel("RMSE about cost-free mean (bp)", color="#52514e")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7, labelcolor="#52514e")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--main-label", default="vix_hyst")
    parser.add_argument("--block-mult", type=float, default=2.0,
                        help="bootstrap block length as a multiple of the maturity")
    parser.add_argument("--exclude-test", nargs=2, metavar=("FROM", "TO"), default=None,
                        help="drop test options starting in this date window (robustness)")
    parser.add_argument("--tag", default="", help="suffix for output file names")
    args = parser.parse_args(argv)
    config = json.loads((args.directory / "config.json").read_text())
    block = int(round(args.block_mult * config["maturity_days"]))
    train = load(args.directory, "train")
    test = load(args.directory, "test", tuple(args.exclude_test) if args.exclude_test else None)

    report = {"maturity_days": config["maturity_days"], "block": block,
              "main_label": args.main_label, "options": {"train": len(train["premium"]),
                                                         "test": len(test["premium"])},
              "mean_premium_bp": {"train": float(train["premium"].mean() * BP),
                                  "test": float(test["premium"].mean() * BP)},
              "kappas": {}}
    table_rows = []
    for kbp in KAPPAS_BP:
        kappa = kbp / BP
        rows = rule_table(test, kappa, block)
        table_rows += rows
        matched = [m for r in rules_in(test) if family(r) != "fixed"
                   for m in [matched_turnover(test, r, kappa, block)] if m is not None]
        report["kappas"][str(kbp)] = {
            "best_on_test_in_sample": min(rows, key=lambda r: r["rmse_adj_bp"]),
            "train_to_test": train_to_test(train, test, kappa, args.main_label, block),
            "placebo_scoreboard": placebo_scoreboard(matched, config["labels"]),
            "matched_turnover_test": matched,
        }
    report["exclude_test"] = args.exclude_test
    tag = f"_{args.tag}" if args.tag else ""
    (args.directory / f"analysis{tag}.json").write_text(json.dumps(report, indent=2))
    import csv
    with (args.directory / f"rules_test{tag}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    plot_frontier(test, args.main_label, (2.0, 10.0, 20.0), args.directory / f"fig_frontier_test{tag}")
    print(f"wrote {args.directory}/analysis{tag}.json, rules_test{tag}.csv, fig_frontier_test{tag}.*")


if __name__ == "__main__":
    main()
