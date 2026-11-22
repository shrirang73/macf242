import pandas as pd

from heston.data import trading_calendar


def test_calendar_counts_days_to_expiries_after_the_data_end():
    known = pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"])   # Mon, Tue, Wed
    cal = trading_calendar(known)
    date, expiry = pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-27")
    # (date, expiry]: Oct 20, 21 (known) + Oct 22, 23, 26, 27 (weekdays) = 6 trading days
    n = cal.searchsorted(expiry, side="right") - cal.searchsorted(date, side="right")
    assert n == 6
    assert cal[:3].equals(pd.DatetimeIndex(known)) and cal.is_monotonic_increasing
