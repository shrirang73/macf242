"""Export the data behind the interactive demo (demo/hedged_desk.html).

Everything comes from the verified pipeline outputs; nothing is invented:
  results/strategy_daily.csv   daily book components (gross P&L, futures turnover,
                               option premium base) per hedge and entry-day regime
  results/hedge.csv.gz         one row per option sold
  results/hedge_paths.csv.gz   each option's daily futures price, value, hedge ratios
  results/params.csv           daily Heston calibration (for the formula card)
  data/inputs/                 NIFTY 50 closes and India VIX
  data/chains.csv.gz           the latest day's option chain (for the trade ticket)

The page recomputes costs, leverage, the stress rule and every statistic live
from these components, with the same formulas as heston/strategy.py.

Usage:  python -m heston.demo_data      (writes demo/demo_data.json and demo/hedged_desk.html)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .calib import call_vec  # noqa: F401  (kept for readers: pricing used by heston_deltas)
from .data import INPUTS, OUT, ROOT, YEAR
from .hedge import bs_delta, heston_deltas
from .pricer import HestonParams
from .stats import EVENTS

DEMO = ROOT / "demo"
TARGET_DAYS, MIN_DAYS = 21, 10


def r(x, nd):
    """Round an array for compact JSON; NaN -> None."""
    a = np.round(np.asarray(x, dtype=float), nd)
    return [None if not np.isfinite(v) else (int(v) if nd == 0 else float(v)) for v in a]


def latest_ticket(chains, params):
    """The option the desk would sell on the last day of data, and its hedges."""
    d = chains["date"].max()
    g = chains[(chains["date"] == d) & (chains["tau_days"] >= MIN_DAYS)]
    exp = g.iloc[(g["tau_days"] - TARGET_DAYS).abs().argsort()[:1]]["expiry"].iloc[0]
    ge = g[g["expiry"] == exp]
    row = ge.iloc[np.abs(np.log(ge["strike"] / ge["F"])).argmin()]
    pr = params[params["date"] <= d].iloc[-1]
    hp = HestonParams(*pr[["v0", "kappa", "theta", "xi", "rho"]].astype(float))
    m, tau = np.array([row["strike"] / row["F"]]), np.array([row["tau"]])
    dh, dmv = heston_deltas(m, tau, hp)
    return dict(date=str(d.date()), expiry=str(pd.Timestamp(exp).date()), strike=float(row["strike"]),
                F=round(float(row["F"]), 2), premium=round(float(row["call"]), 2),
                traded_as=str(row["instrument"]), iv=round(100 * float(row["iv"]), 2),
                tau_days=int(row["tau_days"]), bs_delta=round(float(bs_delta(m[0], tau[0], row["iv"])), 4),
                heston_delta=round(float(dh[0]), 4), mv_delta=round(float(dmv[0]), 4),
                rho=round(float(hp.rho), 3), xi=round(float(hp.xi), 3), vol0=round(100 * np.sqrt(hp.v0), 2),
                params_date=str(pr["date"].date()))


def build():
    comp = pd.read_csv(ROOT / "results" / "strategy_daily.csv", header=[0, 1, 2], index_col=0, parse_dates=True)
    h = pd.read_csv(ROOT / "results" / "hedge.csv.gz", parse_dates=["date", "expiry"])
    p = pd.read_csv(ROOT / "results" / "hedge_paths.csv.gz", parse_dates=["date"]).sort_values(["opt", "k"])
    params = pd.read_csv(ROOT / "results" / "params.csv", parse_dates=["date"])
    nifty = pd.read_csv(INPUTS / "nifty50.csv", parse_dates=["date"]).set_index("date")["close"]
    reg = pd.read_csv(INPUTS / "regime_labels.csv", parse_dates=["date"]).set_index("date")
    chains = pd.read_csv(OUT / "chains.csv.gz", parse_dates=["date", "expiry"])

    days = comp.index
    pos = {d: i for i, d in enumerate(days)}
    # every path date must be a book day, and paths must be consecutive book days
    start = p[p["k"] == 0].set_index("opt")["date"].map(pos)
    assert start.notna().all()
    chk = p["date"].map(pos).to_numpy() - start.loc[p["opt"]].to_numpy() - p["k"].to_numpy()
    assert (chk == 0).all(), "path dates are not consecutive book days"

    # realised volatility over the NEXT 21 trading days (what the insurance actually had to pay for)
    lr = np.log(nifty).diff()
    rv_fwd = lr[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(YEAR) * 100

    comp_out = {}
    for hedge in ("none", "bs", "mv"):
        comp_out[hedge] = {grp: {c: r(1e4 * comp[(hedge, grp, c)].to_numpy(), 4)
                                 for c in ("gross", "fut", "opt")} for grp in ("calm", "stress")}

    entry = p[p["k"] == 0].set_index("opt")
    rho = params.set_index("date")["rho"].reindex(h["date"]).to_numpy()
    xi = params.set_index("date")["xi"].reindex(h["date"]).to_numpy()
    stress = reg["vix_hyst"].reindex(h["date"]).fillna(0).to_numpy().astype(int)
    trades = dict(d=start.loc[h.index].astype(int).tolist(), n=(p.groupby("opt").size() - 1).loc[h.index].tolist(),
                  K=r(h["K"], 2), F0=r(h["F0"], 2), c0=r(h["c0"], 2), s=stress.tolist(),
                  e_none=r(1e4 * h["e_none"], 2), e_bs=r(1e4 * h["e_bs_own"], 2), e_mv=r(1e4 * h["e_heston_mv"], 2),
                  dh=r(entry["phi_heston"].loc[h.index], 4), rho=r(rho, 3), xi=r(xi, 3),
                  exp=[str(x.date()) for x in h["expiry"]])
    paths = dict(F=r(p["F"], 2), V=r(p["V"], 2), bs=r(p["phi_bs_own"], 4), mv=r(p["phi_heston_mv"], 4))

    events = []
    for name, (s, e) in EVENTS.items():
        w = nifty[(nifty.index >= s) & (nifty.index <= e)]
        dd = float((w / w.cummax() - 1).min() * 100)
        events.append(dict(name=name, start=s, end=e, nifty_dd=round(dd, 1)))

    data = dict(
        meta=dict(first=str(days[0].date()), last=str(days[-1].date()), test_start="2019-01-01", book=21,
                  costs=dict(low=[1.0, 1.0], high=[5.0, 3.0]), n_trades=int(len(h)),
                  n_quotes=int(len(chains)), n_days_calibrated=int(len(params)),
                  premium_bp=round(float(1e4 * (h["c0"] / h["F0"]).mean()), 1)),
        days=[str(d.date()) for d in days],
        nifty=r(nifty.reindex(days).ffill(), 2), vix=r(reg["vix"].reindex(days).ffill(), 2),
        rv=r(rv_fwd.reindex(days), 2), stress=reg["vix_hyst"].reindex(days).fillna(0).astype(int).tolist(),
        comp=comp_out, trades=trades, paths=paths, events=events,
        latest=latest_ticket(chains, params),
    )
    return data


def main():
    DEMO.mkdir(exist_ok=True)
    data = build()
    text = json.dumps(data, separators=(",", ":"))
    (DEMO / "demo_data.json").write_text(text)
    tpl = (DEMO / "template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/" in tpl
    page = tpl.replace("/*__DATA__*/", text.replace("</", "<\\/"))
    (DEMO / "hedged_desk.html").write_text(page, encoding="utf-8")
    print(f"data {len(text) / 1e6:.2f} MB; page {len(page) / 1e6:.2f} MB; trades {data['meta']['n_trades']}; "
          f"days {len(data['days'])}; latest ticket {data['latest']}")


if __name__ == "__main__":
    main()
