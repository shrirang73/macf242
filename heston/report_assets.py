"""Generate every figure, table and number used in the report.

Writes report/numbers.tex (macros), report/tables/*.tex, report/figures/*.pdf.
Nothing in the report is typed by hand: re-running this after any change to the
results regenerates all of it.

Usage:  python -m heston.report_assets
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from .calib import call_vec
from .data import INPUTS, OUT, ROOT
from .pricer import HestonParams
from .stats import EVENTS, TEST_START, in_window, paired_ratio

warnings.filterwarnings("ignore")
REP = ROOT / "report"
FIG, TAB = REP / "figures", REP / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# colour-blind-safe palette (Okabe-Ito)
C_BS, C_H, C_MV, C_FLAT, C_MKT = "#0072B2", "#D55E00", "#009E73", "#999999", "#222222"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
NUM: dict[str, str] = {}


def macro(name: str, value, fmt: str = "{}"):
    text = fmt.format(value)
    # strategy and worked-example numbers are used in running text: typographic minus
    if name.startswith(("Strat", "Ex")) and text.startswith("-"):
        text = "$-$" + text[1:]
    NUM[name] = text


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def shade_events(ax, ymax_text=None):
    for i, (n, (s, e)) in enumerate(EVENTS.items()):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#cccccc", alpha=0.45, lw=0)


def load():
    p = pd.read_csv(ROOT / "results" / "params.csv", parse_dates=["date"])
    o = pd.read_csv(ROOT / "results" / "oos.csv.gz", parse_dates=["date", "expiry"])
    h = pd.read_csv(ROOT / "results" / "hedge.csv.gz", parse_dates=["date", "expiry"])
    ch = pd.read_csv(OUT / "chains.csv.gz", parse_dates=["date", "expiry"])
    reg = pd.read_csv(INPUTS / "regime_labels.csv", parse_dates=["date"])
    o = o.merge(reg[["date", "vix", "vix_hyst"]], on="date", how="left")
    h = h.merge(reg[["date", "vix", "vix_hyst"]], on="date", how="left")
    p = p.merge(reg[["date", "vix"]], on="date", how="left")
    return p, o, h, ch, reg


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) * 100      # vol points


MODELS = [("err_bs_atm", "BS flat ATM vol", C_FLAT), ("err_bs_own", "BS own vol (sticky strike)", C_BS),
          ("err_heston", "Heston (calibrated)", C_H)]


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def tex_table(name, header, rows, colfmt, caption_note=None):
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\" if r != "rule" else r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    import re
    text = "\n".join(lines) + "\n"
    # typographic minus for negative numbers (but not in ranges such as 2010--18 or dates)
    text = re.sub(r"(?<![\w$\\-])-(?=\d)", "$-$", text)
    (TAB / f"{name}.tex").write_text(text)


def table_pricing(o):
    samples = [("All days", o), ("Train 2010--18", o[o.date < TEST_START]),
               ("Test 2019--26", o[o.date >= TEST_START]),
               ("Test, calm", o[(o.date >= TEST_START) & (o.vix_hyst == 0)]),
               ("Test, stress", o[(o.date >= TEST_START) & (o.vix_hyst == 1)])]
    cv = EVENTS["COVID-19 crash"]
    samples.append(("COVID-19 crash", in_window(o, *cv)))
    samples.append(("Test excl.\\ COVID", o[(o.date >= TEST_START) & ~o.date.between(*cv)]))
    rows = []
    for name, d in samples:
        r = [name, f"{len(d):,}".replace(",", "\\,")] + [f"{rmse(d[c]):.2f}" for c, _, _ in MODELS]
        r.append(f"{100 * (np.mean(d.err_heston ** 2) / np.mean(d.err_bs_own ** 2) - 1):+.0f}\\%")
        rows.append(r)
    tex_table("pricing", ["Sample", "Option-days", "BS flat", "BS own", "Heston", "Heston vs BS own"],
              rows, "lrrrrr")
    t = o[o.date >= TEST_START]
    macro("PriceHestonTest", f"{rmse(t.err_heston):.2f}")
    macro("PriceBSOwnTest", f"{rmse(t.err_bs_own):.2f}")
    macro("PriceBSAtmTest", f"{rmse(t.err_bs_atm):.2f}")
    macro("PriceHestonAll", f"{rmse(o.err_heston):.2f}")
    macro("PriceBSOwnAll", f"{rmse(o.err_bs_own):.2f}")
    macro("PriceBSAtmAll", f"{rmse(o.err_bs_atm):.2f}")
    # how many times larger the flat-vol Black-Scholes error is than Heston's (headline number)
    macro("PriceFlatOverHeston", f"{rmse(o.err_bs_atm) / rmse(o.err_heston):.1f}")
    macro("NOptionDays", f"{len(o):,}".replace(",", "\\,"))


def table_hedge(h):
    cv = EVENTS["COVID-19 crash"]
    samples = [("All days", h), ("Train 2010--18", h[h.date < TEST_START]),
               ("Test 2019--26", h[h.date >= TEST_START]),
               ("Test, calm", h[(h.date >= TEST_START) & (h.vix_hyst == 0)]),
               ("Test, stress", h[(h.date >= TEST_START) & (h.vix_hyst == 1)]),
               ("Test excl.\\ COVID", h[(h.date >= TEST_START) & ~h.date.between(*cv)])]
    samples += [(n, in_window(h, s, e)) for n, (s, e) in EVENTS.items()]
    rows, store = [], {}
    for name, d in samples:
        d = d.dropna(subset=["e_bs_own", "e_heston", "e_heston_mv"])
        sd = {c: d[c].std() * 1e4 for c in ("e_none", "e_bs_own", "e_heston", "e_heston_mv")}
        r1 = paired_ratio(d, "e_heston_mv", "e_bs_own", kind="var", n_boot=1000)
        r2 = paired_ratio(d, "e_heston", "e_bs_own", kind="var", n_boot=1000)
        store[name] = (r1, r2, sd, len(d))
        sig = lambda r: "$^{*}$" if abs(r["z"]) >= 2 else ""
        rows.append([name, str(len(d)), f"{sd['e_none']:.0f}", f"{sd['e_bs_own']:.1f}",
                     f"{sd['e_heston']:.1f}", f"{sd['e_heston_mv']:.1f}",
                     f"{r2['change_pct']:+.0f}\\%{sig(r2)}", f"{r1['change_pct']:+.0f}\\%{sig(r1)}"])
        if name == "Test 2019--26":
            for k, (a, b) in {"Hedge": (r1, r2)}.items():
                macro("MVvsBSTest", f"{r1['change_pct']:+.0f}")
                macro("MVvsBSTestZ", f"{r1['z']:.1f}")
                macro("HvsBSTest", f"{r2['change_pct']:+.0f}")
                macro("HvsBSTestZ", f"{r2['z']:.1f}")
                macro("HedgeSDBS", f"{sd['e_bs_own']:.1f}")
                macro("HedgeSDMV", f"{sd['e_heston_mv']:.1f}")
                macro("HedgeSDH", f"{sd['e_heston']:.1f}")
                macro("HedgeSDNone", f"{sd['e_none']:.0f}")
        if name == "Test 2019--26":
            r3 = paired_ratio(d, "e_heston_mv", "e_heston", kind="var", n_boot=1000)
            macro("MVvsHTest", f"{r3['change_pct']:+.0f}")
            macro("MVvsHTestZ", f"{r3['z']:.1f}")
        if name == "Test excl.\\ COVID":
            macro("MVvsBSExCovid", f"{r1['change_pct']:+.0f}")
            macro("MVvsBSExCovidZ", f"{r1['z']:.1f}")
        if name == "COVID-19 crash":
            macro("MVvsBSCovid", f"{r1['change_pct']:+.0f}")
            macro("MVvsBSCovidZ", f"{r1['z']:.1f}")
            macro("NCovid", str(len(d)))
        if name == "All days":
            macro("NHedges", f"{len(d):,}".replace(",", "\\,"))
            macro("MVvsBSAll", f"{r1['change_pct']:+.0f}")
            macro("MVvsBSAllZ", f"{r1['z']:.1f}")
    tex_table("hedging", ["Sample", "$n$", "None", "BS", "Heston", "MV",
                          "Heston/BS", "MV/BS"], rows, "lrrrrrrr")
    return store


def table_params(p):
    rows = []
    for name, d in [("Train 2010--18", p[p.date < TEST_START]), ("Test 2019--26", p[p.date >= TEST_START]),
                    ("COVID-19 crash", in_window(p, *EVENTS["COVID-19 crash"])), ("All days", p)]:
        rows.append([name, f"{d.v0.median():.3f}", f"{d.theta.median():.3f}", f"{d.xi.median():.2f}",
                     f"{d.rho.median():.2f}", f"{100 * (d.kappa > 14.9).mean():.0f}\\%",
                     f"{100 * d.rmse_iv.median():.2f}"])
    tex_table("params", ["Sample", "$v_0$", "$\\theta$", "$\\xi$", "$\\rho$", "$\\kappa$ at bound",
                         "Fit RMSE"], rows, "lrrrrrr")
    macro("KappaBoundPct", f"{100 * (p.kappa > 14.9).mean():.0f}")
    macro("FitMedian", f"{100 * p.rmse_iv.median():.2f}")
    macro("RhoMedian", f"{p.rho.median():.2f}")
    macro("XiMedian", f"{p.xi.median():.2f}")
    macro("NDays", f"{len(p):,}".replace(",", "\\,"))
    macro("FitBad", str(int((p.rmse_iv > 0.03).sum())))


def table_data(ch):
    rows = [["Option quotes (OTM, traded)", f"{len(ch):,}".replace(",", "\\,")],
            ["Trading days", f"{ch.date.nunique():,}".replace(",", "\\,")],
            ["First / last day", f"{ch.date.min().date()} / {ch.date.max().date()}"],
            ["Quotes per day (median)", f"{ch.groupby('date').size().median():.0f}"],
            ["Monthly expiries per day (median)", f"{ch.groupby('date').expiry.nunique().median():.0f}"],
            ["Median ATM-region implied vol", f"{100 * ch.iv.median():.1f}\\%"]]
    tex_table("data", ["Item", "Value"], rows, "lr")
    macro("NQuotes", f"{len(ch):,}".replace(",", "\\,"))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def surface_pair(p, ch, dates):
    """3D: market implied vols (dots) against the fitted Heston surface."""
    fig = plt.figure(figsize=(7.2, 3.3))
    for i, d in enumerate(dates):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        g = ch[ch.date == d]
        pr = p[p.date == d].iloc[0]
        hp = HestonParams(*pr[["v0", "kappa", "theta", "xi", "rho"]].astype(float))
        lm = np.linspace(-0.10, 0.10, 31)
        td = np.linspace(g.tau_days.min(), g.tau_days.max(), 25)
        LM, TD = np.meshgrid(lm, td)
        from .data import black_iv
        price = call_vec(1.0, np.exp(LM.ravel()), TD.ravel() / 252, hp)
        iv = np.array([black_iv(c, 1.0, k, t / 252) for c, k, t in
                       zip(price, np.exp(LM.ravel()), TD.ravel())]).reshape(LM.shape)
        ax.plot_surface(LM, TD, iv * 100, cmap="viridis", alpha=0.75, linewidth=0, antialiased=True)
        lmk = np.log(g.strike / g.F)
        ax.scatter(lmk, g.tau_days, g.iv * 100, s=5, c=C_H, depthshade=False)
        ax.set_xlabel("log-moneyness"); ax.set_ylabel("days to expiry"); ax.set_zlabel("implied vol (%)")
        ax.set_xlim(-0.10, 0.10)
        ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.view_init(elev=24, azim=-58)
        ax.set_title(f"{d.date()}  (fit RMSE {100 * pr.rmse_iv:.2f} pts)")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "surface3d")


def oos_surface3d(o):
    """3D: Heston minus BS-own out-of-sample RMSE over moneyness x maturity."""
    t = o[o.date >= TEST_START].copy()
    # coarse bins so that every cell holds many observations (no empty-cell artefacts)
    lb = np.array([-0.12, -0.07, -0.025, 0.025, 0.07, 0.12])
    tb = np.array([5, 10, 21, 35, 70])
    t["mb"] = pd.cut(t.logm, lb, labels=False)
    t["tb"] = pd.cut(t.tau_days, tb, labels=False)
    Z = np.full((len(tb) - 1, len(lb) - 1), np.nan)
    ZB = Z.copy()
    for i in range(len(tb) - 1):
        for j in range(len(lb) - 1):
            d = t[(t.tb == i) & (t.mb == j)]
            if len(d) >= 200:        # mean absolute error: robust to a few tail days
                Z[i, j] = 100 * d.err_heston.abs().mean()
                ZB[i, j] = 100 * d.err_bs_own.abs().mean()
    xm = 0.5 * (lb[:-1] + lb[1:])
    ym = 0.5 * (tb[:-1] + tb[1:])
    X, Y = np.meshgrid(xm, ym)
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    assert np.isfinite(Z).all(), "empty cell in the error surface"
    # smooth (bilinear) interpolation of the coarse grid, for display only
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(ym, xm, Z, kx=1, ky=1)
    xf, yf = np.linspace(xm[0], xm[-1], 40), np.linspace(ym[0], ym[-1], 40)
    XF, YF = np.meshgrid(xf, yf)
    ax.plot_surface(XF, YF, spl(yf, xf), cmap="magma_r", alpha=0.95, linewidth=0, antialiased=True)
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("days to expiry")
    ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax.view_init(elev=28, azim=-125)
    ax.tick_params(labelsize=7)
    ax.set_title("Heston mean absolute error (vol pts)")
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(Z - ZB, origin="lower", aspect="auto", cmap="RdBu_r",
                    extent=[lb[0], lb[-1], tb[0], tb[-1]], vmin=-2.5, vmax=2.5)
    ax2.grid(False)
    ax2.set_xlabel("log-moneyness"); ax2.set_ylabel("days to expiry")
    ax2.set_title("Heston minus BS own (vol pts)")
    fig.colorbar(im, ax=ax2, shrink=0.85)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "oos_surface")


def params_ts(p):
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
    s = p.set_index("date")
    roll = lambda x: x.rolling(21, min_periods=5).median()
    axs[0].plot(s.index, 100 * np.sqrt(s.v0), color=C_H, lw=0.7, label=r"Heston $\sqrt{v_0}$")
    axs[0].plot(s.index, s.vix, color=C_MKT, lw=0.6, alpha=0.7, label="India VIX")
    axs[0].set_ylabel("vol (%)"); axs[0].legend(loc="upper right", ncol=2)
    axs[1].plot(s.index, roll(s.rho), color=C_BS, lw=0.9); axs[1].set_ylabel(r"$\rho$")
    axs[2].plot(s.index, roll(s.xi), color=C_MV, lw=0.9); axs[2].set_ylabel(r"$\xi$")
    axs[3].plot(s.index, 100 * s.rmse_iv, color=C_FLAT, lw=0.6)
    axs[3].plot(s.index, roll(100 * s.rmse_iv), color=C_MKT, lw=0.9)
    axs[3].set_ylabel("in-sample fit\n(vol pts)"); axs[3].set_ylim(0, 4)
    for ax in axs:
        shade_events(ax)
    names = list(EVENTS)
    for i, n in enumerate(names):
        mid = pd.Timestamp(EVENTS[n][0]) + (pd.Timestamp(EVENTS[n][1]) - pd.Timestamp(EVENTS[n][0])) / 2
        axs[0].text(mid, axs[0].get_ylim()[1] * 0.93, str(i + 1), ha="center", fontsize=8, weight="bold")
    fig.align_ylabels(axs)
    fig.tight_layout()
    save(fig, "params_ts")


def oos_by_year(o):
    o = o.assign(year=o.date.dt.year)
    g = o.groupby("year")
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = 0.27
    yrs = sorted(o.year.unique())
    for k, (c, lab, col) in enumerate(MODELS):
        ax.bar(np.arange(len(yrs)) + (k - 1) * w, [rmse(g.get_group(y)[c]) for y in yrs], w,
               color=col, label=lab)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs, rotation=45, fontsize=7.5)
    ax.set_ylabel("one-day-ahead RMSE\n(implied-vol points)")
    ax.set_ylim(0, 5.3)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "oos_by_year")


def oos_by_moneyness(o):
    t = o[o.date >= TEST_START].copy()
    bins = np.array([-0.12, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.12])
    t["b"] = pd.cut(t.logm, bins, labels=False)
    mid = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for c, lab, col in MODELS:
        ax.plot(mid, [rmse(t[t.b == i][c]) for i in range(len(mid))], "o-", ms=3, lw=1.2,
                color=col, label=lab)
    ax.set_xlabel("log-moneyness $\\ln(K/F)$"); ax.set_ylabel("RMSE (vol pts)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "oos_by_moneyness")


def hedge_dist(h, store):
    t = h[h.date >= TEST_START].dropna(subset=["e_bs_own", "e_heston", "e_heston_mv"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    bins = np.linspace(-0.04, 0.04, 70)
    for c, lab, col in [("e_bs_own", "BS own vol", C_BS), ("e_heston", "Heston", C_H),
                        ("e_heston_mv", "Heston MV", C_MV)]:
        axs[0].hist(t[c].clip(-0.04, 0.04), bins=bins, histtype="step", color=col, lw=1.2, label=lab)
    axs[0].set_xlabel("hedging error (fraction of $F_0$)"); axs[0].set_ylabel("options")
    axs[0].legend(); axs[0].set_title("Test period 2019--26")
    names = ["Test 2019--26", "Test excl.\\ COVID"] + list(EVENTS)
    labels = ["Test 2019--26", "Test excl. COVID"] + [str(i + 1) for i in range(len(EVENTS))]
    labels = ["Test", "Test ex-\nCOVID"] + [f"event {i + 1}" for i in range(len(EVENTS))]
    y = np.arange(len(names))[::-1]
    for yy, n in zip(y, names):
        r1 = store[n][0]
        axs[1].errorbar(r1["change_pct"], yy, xerr=[[r1["change_pct"] - r1["lo"]], [r1["hi"] - r1["change_pct"]]],
                        fmt="o", color=C_MV, ms=4, capsize=2, lw=1)
    axs[1].axvline(0, color=C_MKT, lw=0.8)
    axs[1].set_yticks(y); axs[1].set_yticklabels([n.replace("\\ ", " ") for n in names], fontsize=7)
    axs[1].set_xlabel("Heston MV vs BS: change in hedging variance (%)")
    axs[1].set_title("Paired, 95% block-bootstrap interval")
    fig.tight_layout()
    save(fig, "hedge_dist")


def cumulative_pnl(h):
    t = h[h.date >= TEST_START].dropna(subset=["e_bs_own", "e_heston_mv"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.plot(t.date, (t.e_bs_own.abs()).rolling(63).mean() * 1e4, color=C_BS, lw=1, label="BS own vol")
    ax.plot(t.date, (t.e_heston_mv.abs()).rolling(63).mean() * 1e4, color=C_MV, lw=1, label="Heston MV")
    ax.plot(t.date, (t.e_heston.abs()).rolling(63).mean() * 1e4, color=C_H, lw=1, label="Heston")
    shade_events(ax)
    ax.set_ylabel("mean |hedging error|\n(bp of $F_0$, 63-option avg)")
    ax.set_xlim(pd.Timestamp(TEST_START), t.date.max())
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "hedge_rolling")


def delta_surface(p):
    """3D: Heston minimum-variance delta minus BS delta, on a calibrated day."""
    from .hedge import heston_deltas, bs_delta
    d = pd.Timestamp("2024-06-03")
    pr = p[p.date <= d].iloc[-1]
    hp = HestonParams(*pr[["v0", "kappa", "theta", "xi", "rho"]].astype(float))
    lm = np.linspace(-0.08, 0.08, 25)
    td = np.linspace(8, 63, 20)
    LM, TD = np.meshgrid(lm, td)
    m, tau = np.exp(LM.ravel()), TD.ravel() / 252
    dh, dmv = heston_deltas(m, tau, hp)
    vol = np.sqrt(0.5 * (hp.v0 + hp.theta))
    db = bs_delta(m, tau, vol)
    fig = plt.figure(figsize=(7.2, 3.1))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(LM, TD, dmv.reshape(LM.shape), cmap="viridis", alpha=0.85, linewidth=0)
    ax.set_xlabel("log-moneyness $\\ln(K/F)$"); ax.set_ylabel("days to expiry"); ax.set_zlabel("MV delta")
    ax.view_init(elev=22, azim=-60); ax.tick_params(labelsize=7)
    ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax2.plot_surface(LM, TD, (dmv - db).reshape(LM.shape), cmap="RdBu_r", alpha=0.9, linewidth=0)
    ax2.set_xlabel("log-moneyness $\\ln(K/F)$"); ax2.set_ylabel("days to expiry")
    ax2.set_zlabel("MV delta $-$ BS delta"); ax2.view_init(elev=22, azim=-60); ax2.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "delta_surface")
    macro("DeltaDate", str(pr.date.date()))
    macro("DeltaGapAtm", f"{float((dmv - db).reshape(LM.shape)[:, 12].mean()):+.3f}")


def simulation_assets():
    """Table, figure and macros for the simulated-Heston hedging benchmark."""
    f = ROOT / "results" / "simulation.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    names = {"bs_own": "BS own vol", "heston": "Heston delta", "heston_mv": "Heston MV"}
    cols = {"bs_own": C_BS, "heston": C_H, "heston_mv": C_MV}
    # table: sd (bp) by interval for each scenario and hedge, plus MV vs BS at daily
    rows = []
    for sc in s.scenario.unique():
        d = s[s.scenario == sc]
        for hedge in ("bs_own", "heston", "heston_mv"):
            x = d[d.hedge == hedge].set_index("interval_days")
            r = [sc.capitalize() if hedge == "bs_own" else "", names[hedge]]
            r += [f"{x.loc[h, 'sd_bp']:.1f}" for h in sorted(d.interval_days.unique())]
            daily = x.loc[1.0]
            r.append("--" if hedge == "bs_own" else
                     f"{daily.vs_bs_pct:+.0f}\\% [{daily.vs_bs_lo:+.0f}, {daily.vs_bs_hi:+.0f}]")
            rows.append(r)
        rows.append("rule")
    rows = rows[:-1]
    ints = sorted(s.interval_days.unique())
    lab = lambda h: "4/day" if h < 1 else (f"{int(h)} day" + ("s" if h > 1 else ""))
    tex_table("simulation", ["Parameters", "Hedge"] + [lab(h) for h in ints] + ["Var.\\ vs BS (daily)"],
              rows, "ll" + "r" * len(ints) + "r")
    for sc, tag in (("calibrated", "Cal"), ("time-series", "TS")):
        d = s[(s.scenario == sc) & (s.interval_days == 1.0)].set_index("hedge")
        macro(f"Sim{tag}MVvsBS", f"{d.loc['heston_mv', 'vs_bs_pct']:+.0f}")
        macro(f"Sim{tag}HvsBS", f"{d.loc['heston', 'vs_bs_pct']:+.0f}")
        macro(f"Sim{tag}SDBS", f"{d.loc['bs_own', 'sd_bp']:.1f}")
        macro(f"Sim{tag}SDMV", f"{d.loc['heston_mv', 'sd_bp']:.1f}")
        q = s[(s.scenario == sc) & (s.hedge == "heston_mv")].set_index("interval_days")
        macro(f"Sim{tag}MVFour", f"{q.loc[ints[0], 'sd_bp']:.1f}")
        macro(f"Sim{tag}MVWeek", f"{q.loc[ints[-1], 'sd_bp']:.1f}")
    pc = s[s.scenario == "calibrated"].iloc[0]
    macro("SimCalParams", "(v_0,\\kappa,\\theta,\\xi,\\rho)=(%.3f,\\,%.1f,\\,%.3f,\\,%.2f,\\,%.2f)" % (
        pc.param_v0, pc.param_kappa, pc.param_theta, pc.param_xi, pc.param_rho))
    # figure: sd vs rebalancing interval, one panel per scenario
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
    for ax, sc in zip(axs, s.scenario.unique()):
        d = s[s.scenario == sc]
        for hedge in ("bs_own", "heston", "heston_mv"):
            x = d[d.hedge == hedge].sort_values("interval_days")
            ax.plot(x.interval_days, x.sd_bp, "o-", ms=3.5, lw=1.3, color=cols[hedge], label=names[hedge])
        ax.set_xscale("log"); ax.set_xticks(ints); ax.set_xticklabels([lab(h) for h in ints])
        ax.minorticks_off()
        ax.set_xlabel("rebalancing interval"); ax.set_title(f"{sc.capitalize()} parameters")
    axs[0].set_ylabel("sd of hedging error (bp)")
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "simulation")


def strategy_assets():
    """Strategy table, equity-curve figure, worked one-trade example (macros)."""
    from . import strategy as st
    out, curves, h, _ = st.run(verbose=False)
    lbl = {"none": "Unhedged", "bs": "BS hedge", "mv": "Heston MV hedge"}
    rows = []
    for cost in ("low", "high"):
        for hedge in ("none", "bs", "mv"):
            for rule in ("always", "skip stress"):
                t = out[(out.cost == cost) & (out.hedge == hedge) & (out.rule == rule) & (out.period == "test")].iloc[0]
                tr = out[(out.cost == cost) & (out.hedge == hedge) & (out.rule == rule) & (out.period == "train")].iloc[0]
                rows.append([cost.capitalize() if (hedge, rule) == ("none", "always") else "",
                             lbl[hedge], "skip stress" if rule != "always" else "every day",
                             f"{t.ann_ret_pct:+.2f}", f"{t.ann_vol_pct:.2f}", f"{t.sharpe:.2f}",
                             f"{t.max_dd_pct:.1f}", f"{t.worst_day_pct:.2f}", f"{tr.sharpe:.2f}"])
        rows.append("rule")
    tex_table("strategy", ["Costs", "Hedge", "Entry rule", "Return/yr", "Vol/yr", "Sharpe",
                           "Max DD", "Worst day", "Sharpe (train)"], rows[:-1], "lllrrrrrr")
    g = lambda c, hd, r, col, per="test": out[(out.cost == c) & (out.hedge == hd) & (out.rule == r)
                                          & (out.period == per)].iloc[0][col]
    macro("StratBSRet", f"{g('low', 'bs', 'always', 'ann_ret_pct'):+.1f}")
    macro("StratBSSharpe", f"{g('low', 'bs', 'always', 'sharpe'):.2f}")
    macro("StratBSDD", f"{g('low', 'bs', 'always', 'max_dd_pct'):.1f}")
    macro("StratMVRet", f"{g('low', 'mv', 'always', 'ann_ret_pct'):+.1f}")
    macro("StratMVSharpe", f"{g('low', 'mv', 'always', 'sharpe'):.2f}")
    macro("StratMVDD", f"{g('low', 'mv', 'always', 'max_dd_pct'):.1f}")
    macro("StratNoneRet", f"{g('low', 'none', 'always', 'ann_ret_pct'):+.1f}")
    macro("StratNoneDD", f"{g('low', 'none', 'always', 'max_dd_pct'):.0f}")
    macro("StratBSHighRet", f"{g('high', 'bs', 'always', 'ann_ret_pct'):+.2f}")
    macro("StratMVHighRet", f"{g('high', 'mv', 'always', 'ann_ret_pct'):+.2f}")
    macro("StratBSWorstDay", f"{g('low', 'bs', 'always', 'worst_day_pct'):.1f}")
    macro("StratMVWorstDay", f"{g('low', 'mv', 'always', 'worst_day_pct'):.1f}")
    macro("StratMVSkipSharpe", f"{g('low', 'mv', 'skip stress', 'sharpe'):.2f}")
    macro("StratMeanGrossBS", f"{1e4 * h.e_bs_own.mean():+.1f}")
    macro("StratMeanGrossMV", f"{1e4 * h.e_heston_mv.mean():+.1f}")
    macro("StratMeanNone", f"{1e4 * h.e_none.mean():+.1f}")
    macro("StratTurnMV", f"{h.turn_heston_mv.mean():.1f}")
    macro("StratPremium", f"{1e4 * (h.c0 / h.F0).mean():.0f}")

    # equity curves (low costs) and drawdown of the MV-hedged book
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    styles = [(("low", "none", "always"), "Unhedged", C_FLAT, "-"),
              (("low", "bs", "always"), "BS hedge", C_BS, "-"),
              (("low", "mv", "always"), "Heston MV hedge", C_MV, "-"),
              (("low", "mv", "skip stress"), "Heston MV, skip stress", C_MV, "--")]
    for key, lab, col, ls in styles:
        m = curves[key]
        x = m.index
        axs[0].plot(x, 100 * m.cumsum(), ls, color=col, lw=1.2, label=lab)
    axs[0].axvspan(pd.Timestamp("2010-07-01"), pd.Timestamp(TEST_START), color="#eeeeee", lw=0, zorder=0)
    axs[0].text(pd.Timestamp("2014-06-01"), axs[0].get_ylim()[1] * 0.9, "train", fontsize=8, color="#666")
    axs[0].set_ylabel("cumulative P&L (% of notional)")
    axs[0].legend(fontsize=7, loc="lower left")
    for key, lab, col, ls in styles[1:3]:
        m = curves[key]
        eq = m.cumsum()
        axs[1].plot(m.index, 100 * (eq - eq.cummax()), ls, color=col, lw=1.1, label=lab)
    axs[1].set_ylabel("drawdown (% of notional)")
    axs[1].set_title("Hedged books: drawdown")
    import matplotlib.dates as mdates
    axs[1].xaxis.set_major_locator(mdates.YearLocator(4))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axs:
        s, e = EVENTS["COVID-19 crash"]
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#f3c6c6", alpha=0.6, lw=0, zorder=0)
    fig.tight_layout()
    save(fig, "strategy")

    # worked example: the test-period trade whose MV-hedged result is the median
    t = h[h.date >= TEST_START].copy()
    ex = t.iloc[(t.e_heston_mv - t.e_heston_mv.median()).abs().argsort().iloc[0]]
    pay = max(ex.F0 * 0 + (ex.c0 - ex.e_none * ex.F0), 0.0)        # payoff = c0 - e_none*F0
    hedge_gain = (ex.e_heston_mv - ex.e_none) * ex.F0
    macro("ExDate", str(ex.date.date())); macro("ExExpiry", str(ex.expiry.date()))
    macro("ExF", f"{ex.F0:,.0f}".replace(",", "\\,")); macro("ExK", f"{ex.K:,.0f}".replace(",", "\\,"))
    macro("ExPrem", f"{ex.c0:.1f}"); macro("ExPay", f"{pay:.1f}")
    macro("ExHedge", f"{hedge_gain:+.1f}"); macro("ExNet", f"{ex.e_heston_mv * ex.F0:+.1f}")
    macro("ExNetBp", f"{1e4 * ex.e_heston_mv:+.1f}")
    if pay > 0:
        story = (f"The index rises, so the call finishes worth {pay:.1f} points, which the desk pays to the buyer; "
                 f"the long-futures hedge earns {hedge_gain:.1f} points back:")
    else:
        story = ("The index falls, so the call expires worthless; "
                 f"the long-futures hedge loses {abs(hedge_gain):.1f} points as the index drops:")
    NUM["ExStory"] = story
    macro("StratMeanNoneAbs", f"{abs(1e4 * h.e_none.mean()):.1f}")


def main():
    p, o, h, ch, reg = load()
    simulation_assets()
    strategy_assets()
    table_pricing(o)
    store = table_hedge(h)
    table_params(p)
    table_data(ch)
    # two illustration dates: a calm day and the COVID peak (best-fitting day near it)
    calm = pd.Timestamp("2024-06-03")
    cov = p[(p.date >= "2020-03-16") & (p.date <= "2020-03-31")].sort_values("rmse_iv").iloc[0].date
    surface_pair(p, ch, [calm, cov])
    macro("SurfDateA", str(calm.date())); macro("SurfDateB", str(cov.date()))
    oos_surface3d(o)
    params_ts(p)
    oos_by_year(o)
    oos_by_moneyness(o)
    hedge_dist(h, store)
    cumulative_pnl(h)
    delta_surface(p)
    (REP / "numbers.tex").write_text("\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUM.items())) + "\n")
    print("wrote", len(NUM), "macros;", len(list(FIG.glob('*.pdf'))), "figures;", len(list(TAB.glob('*.tex'))), "tables")


if __name__ == "__main__":
    main()
