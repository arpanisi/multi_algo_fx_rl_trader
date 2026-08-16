"""
Regression tests for the Tier 2 Train-window slice size (code review Finding 1).
The slice must be the first calendar year of the computed Train window (~BARS_PER_YEAR
bars in chronological order), computed from the locked constant rather than a
hardcoded row count, and must never exceed the full Train window.
"""
import pandas as pd

from config.settings import BARS_PER_YEAR
from scripts.run_tier2_small import tier2_train_slice


def _timestamps(n):
    return pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"))


def test_tier2_slice_is_first_calendar_year_not_hardcoded_5000():
    train_ts = _timestamps(BARS_PER_YEAR + 5000)
    sliced = tier2_train_slice(train_ts)
    assert len(sliced) == BARS_PER_YEAR
    assert list(sliced) == list(train_ts[:BARS_PER_YEAR])
    assert len(sliced) != 5000


def test_tier2_slice_is_chronological_head_of_train_window():
    train_ts = _timestamps(2 * BARS_PER_YEAR)
    sliced = tier2_train_slice(train_ts)
    assert sliced.equals(train_ts[:BARS_PER_YEAR])
    assert sliced.is_monotonic_increasing


def test_tier2_slice_falls_back_to_full_train_window_when_shorter_than_calendar_year():
    train_ts = _timestamps(100)
    sliced = tier2_train_slice(train_ts)
    assert len(sliced) == 100
    assert sliced.equals(train_ts)


def test_tier2_slice_never_larger_than_train_window():
    train_ts = _timestamps(BARS_PER_YEAR // 2)
    sliced = tier2_train_slice(train_ts)
    assert len(sliced) <= len(train_ts)
    assert sliced.equals(train_ts)
