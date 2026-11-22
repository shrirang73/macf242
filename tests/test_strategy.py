"""Hand-worked checks of the strategy's daily accounting (two options, three days)."""
import numpy as np
import pandas as pd
import pytest

from heston import strategy as st

D = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])


def tiny():
    # option 0: sold on day 1 (calm); option 1: sold on day 2 (stress)
    h = pd.DataFrame({"date": [D[0], D[1]], "F0": [100.0, 100.0], "c0": [2.0, 3.0], "vix_hyst": [0, 1]})
    p = pd.DataFrame({
        "opt": [0, 0, 0, 1, 1], "k": [0, 1, 2, 0, 1], "date": [D[0], D[1], D[2], D[1], D[2]],
        "F": [100.0, 101.0, 102.0, 101.0, 99.0], "V": [2.0, 2.6, 2.0, 3.0, 1.0],
        "phi_bs_own": [0.5, 0.55, np.nan, 0.4, np.nan]})
    p["phi_heston_mv"] = p["phi_bs_own"]
    return h, p


def test_daily_gross_pnl_by_hand():
    comp = st.daily_components(*tiny())
    calm, stress = comp[("bs", "calm")], comp[("bs", "stress")]
    # option 0: day 2: -(2.6-2) + 0.5*1 = -0.1 ; day 3: -(2-2.6) + 0.55*1 = 1.15   (per F0 = 100)
    assert calm.loc[D[1], "gross"] == pytest.approx(-0.1 / 100 / 21)
    assert calm.loc[D[2], "gross"] == pytest.approx(1.15 / 100 / 21)
    # option 1: day 3: -(1-3) + 0.4*(99-101) = 1.2
    assert stress.loc[D[2], "gross"] == pytest.approx(1.2 / 100 / 21)
    # unhedged: only the option value changes
    assert comp[("none", "calm")].loc[D[2], "gross"] == pytest.approx(0.6 / 100 / 21)


def test_costs_turnover_and_premium():
    comp = st.daily_components(*tiny())
    calm = comp[("bs", "calm")]
    assert calm.loc[D[0], "fut"] == pytest.approx(0.5 * 100 / 100 / 21)          # open 0.5 futures
    assert calm.loc[D[1], "fut"] == pytest.approx(0.05 * 101 / 100 / 21)         # adjust to 0.55
    assert calm.loc[D[2], "fut"] == pytest.approx(0.0)                            # settled, no trade
    assert calm.loc[D[0], "opt"] == pytest.approx(2.0 / 100 / 21)                 # premium on sale day
    assert comp[("none", "calm")]["fut"].sum() == 0.0


def test_skip_stress_drops_options_sold_in_stress():
    comp = st.daily_components(*tiny())
    always = st.net_series(comp, "bs", "low", "always")
    skip = st.net_series(comp, "bs", "low", "skip stress")
    fut, spread = st.COSTS["low"]
    stress = comp[("bs", "stress")]
    expected = stress["gross"] - fut * stress["fut"] - spread * stress["opt"]
    assert np.allclose(always - skip, expected)


def test_metrics_drawdown():
    r = pd.Series([0.01, -0.03, 0.01, 0.02], index=pd.date_range("2021-01-01", periods=4))
    assert st.metrics(r)["max_dd_pct"] == pytest.approx(-3.0)
