"""
Unit tests for Steps 1, 2, and 3: Data Pipeline, Technical Indicators, and State Representation.
"""
import pytest
import numpy as np
import pandas as pd
from config.settings import ALL_PAIRS, TRADED_PAIR, CONTEXT_PAIRS, STATE_DIM
from src.data.pipeline import (
    generate_synthetic_ticks, resample_15m_bars, build_aligned_15m_panel, compute_chronological_splits
)
from src.features.indicators import (
    compute_all_eurusd_indicators, compute_ema, compute_macd,
    compute_bollinger_width, compute_stochastic
)
from src.features.state import StateBuilder, StateFeatureNormalizer, expected_state_dim
import src.data.pipeline as pipeline_module

def test_15m_resampling():
    df_1m = generate_synthetic_ticks("EURUSD", start_date="2020-01-01", end_date="2020-01-02")
    df_15m = resample_15m_bars(df_1m)
    assert not df_15m.empty
    assert "close" in df_15m.columns
    # 15m bars should have fewer rows than 1m bars
    assert len(df_15m) < len(df_1m)

def test_indicators_determinism_and_warmup():
    df_1m = generate_synthetic_ticks("EURUSD", start_date="2020-01-01", end_date="2020-01-10")
    df_15m = resample_15m_bars(df_1m)
    df_ind = compute_all_eurusd_indicators(df_15m)

    assert df_ind.shape[1] == 10
    # First w-1 bars should be NaN due to warmup
    assert df_ind["ind_sma20"].iloc[0:19].isna().all()
    assert not df_ind["ind_sma20"].iloc[25:].isna().any()

def test_state_vector_dimensionality_scales_with_present_context_pairs():
    # Build synthetic panel for 3 pairs
    records = []
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    dates = pd.date_range("2020-01-01", "2020-01-05", freq="15min", tz="UTC")
    dates = dates[dates.dayofweek < 5]

    for pair in pairs:
        base = 1.10 if pair == "EURUSD" else (1.30 if pair == "GBPUSD" else 130.0)
        prices = base + np.random.randn(len(dates)) * 0.001
        df_p = pd.DataFrame({
            "open": prices, "high": prices + 0.0001, "low": prices - 0.0001,
            "close": prices, "volume": 100, "pair": pair
        }, index=dates)
        records.append(df_p)

    df_panel = pd.concat(records)
    
    sb = StateBuilder(df_panel)
    # Test valid timestamp after warmup
    target_ts = dates[40]
    raw_state = sb.get_state_vector(target_ts, current_position=1.0, entry_price=1.10)
    
    assert sb.context_pairs_present == ["GBPUSD", "USDJPY"]
    assert sb.state_dim == expected_state_dim(2) == 26
    assert len(raw_state) == 26
    assert not np.isnan(raw_state).any()
    # The context slice contains exactly the two real context-pair returns present in the panel.
    assert raw_state[8:10].shape == (2,)

def test_full_context_panel_state_vector_dimensionality():
    records = []
    pairs = [TRADED_PAIR] + CONTEXT_PAIRS
    dates = pd.date_range("2020-01-01", periods=60, freq="15min", tz="UTC")

    for i, pair in enumerate(pairs):
        base = 1.10 + i * 0.01
        prices = base * (1.0 + 0.0001 * np.arange(len(dates)))
        df_p = pd.DataFrame({
            "open": prices, "high": prices + 0.0001, "low": prices - 0.0001,
            "close": prices, "volume": 100, "pair": pair
        }, index=dates)
        records.append(df_p)

    sb = StateBuilder(pd.concat(records))
    raw_state = sb.get_state_vector(dates[40])

    assert sb.context_pairs_present == CONTEXT_PAIRS
    assert sb.state_dim == expected_state_dim(11) == STATE_DIM == 35
    assert len(raw_state) == 35
    assert raw_state[8:19].shape == (11,)
    assert np.count_nonzero(raw_state[8:19]) == 11

def test_train_normalizer_isolation():
    norm = StateFeatureNormalizer()
    train_data = np.random.randn(100, STATE_DIM) * 5.0 + 2.0
    norm.fit(train_data)

    assert norm.is_fitted
    test_data = np.random.randn(10, STATE_DIM)
    transformed = norm.transform(test_data)
    assert transformed.shape == (10, STATE_DIM)

def test_production_download_failure_does_not_generate_synthetic_data(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "HISTDATA_LOCAL_DIR", tmp_path / "histdata_fx_1m")

    def fail_download(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(pipeline_module, "hf_hub_download", fail_download)

    with pytest.raises(RuntimeError, match="Failed to fetch real HistData ticks for EURUSD"):
        pipeline_module.download_or_load_pair_ticks("EURUSD")

def test_build_panel_uses_reported_gaps_to_drop_long_gap(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "PANEL_CACHE_PATH", tmp_path / "fx_panel_15m.parquet")
    dates_1m = pd.date_range("2020-01-01 00:00", periods=180, freq="1min", tz="UTC")

    def fake_ticks(symbol):
        base = 1.10 if symbol == "EURUSD" else 1.30
        prices = np.full(len(dates_1m), base)
        return pd.DataFrame({
            "open": prices,
            "high": prices + 0.0001,
            "low": prices - 0.0001,
            "close": prices,
            "volume": 100,
        }, index=dates_1m)

    def fake_gaps(symbol):
        if symbol == "EURUSD":
            return pd.DataFrame({
                "length": [pd.Timedelta("75min")],
                "start": [pd.Timestamp("2020-01-01 01:00", tz="UTC")],
                "end": [pd.Timestamp("2020-01-01 02:00", tz="UTC")],
            })
        return pd.DataFrame(columns=["length", "start", "end"])

    monkeypatch.setattr(pipeline_module, "download_or_load_pair_ticks", fake_ticks)
    monkeypatch.setattr(pipeline_module, "download_or_load_pair_gaps", fake_gaps)

    panel = pipeline_module.build_aligned_15m_panel(pairs=["EURUSD", "GBPUSD"], force_refresh=True)
    dropped_gap_ts = pd.Timestamp("2020-01-01 01:30", tz="UTC")
    assert dropped_gap_ts not in panel.index

def test_ema_and_macd_warmup_values_are_nan():
    close = pd.Series(np.linspace(1.0, 2.0, 60))

    ema = compute_ema(close, window=20)
    assert ema.iloc[:19].isna().all()
    assert not pd.isna(ema.iloc[19])

    macd = compute_macd(close, fast=12, slow=26, signal=9)
    first_valid = macd.first_valid_index()
    assert first_valid == 33
    assert macd.iloc[:first_valid].isna().all()

def test_bollinger_and_stochastic_single_scalar_definitions():
    close = pd.Series(np.linspace(10.0, 30.0, 30))
    high = close + 1.0
    low = close - 1.0

    bollinger = compute_bollinger_width(close, window=20, k=2.0)
    sma = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std(ddof=1)
    expected_bollinger = ((sma + 2.0 * rolling_std) - (sma - 2.0 * rolling_std)) / sma
    pd.testing.assert_series_equal(bollinger, expected_bollinger)

    stochastic = compute_stochastic(high, low, close, window=14)
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    expected_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    pd.testing.assert_series_equal(stochastic, expected_k)


def test_warmup_guard_rejects_macd_warmup_bars_and_prevents_nan_normalizer():
    records = []
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    dates = pd.date_range("2020-01-01", periods=60, freq="15min", tz="UTC")

    for pair in pairs:
        base = 1.10 if pair == "EURUSD" else (1.30 if pair == "GBPUSD" else 130.0)
        prices = base + np.random.randn(len(dates)) * 0.001
        df_p = pd.DataFrame({
            "open": prices, "high": prices + 0.0001, "low": prices - 0.0001,
            "close": prices, "volume": 100, "pair": pair
        }, index=dates)
        records.append(df_p)

    df_panel = pd.concat(records)
    sb = StateBuilder(df_panel)

    # Bars 0 through 32 (indices 0..32, 33 bars total) are within MACD warmup and MUST raise ValueError
    for idx in range(33):
        ts = dates[idx]
        with pytest.raises(ValueError, match="within indicator warmup period|NaN"):
            sb.get_state_vector(ts)

    # Index 33 (the 34th bar) is the first bar past MACD warmup and MUST return a valid state vector
    valid_state = sb.get_state_vector(dates[33])
    assert len(valid_state) == expected_state_dim(2) == 26
    assert not np.isnan(valid_state).any()

    # Verify building raw_states using try/except produces clean normalizer
    raw_states = []
    for ts in dates[30:]:
        try:
            raw_states.append(sb.get_state_vector(ts))
        except ValueError:
            continue

    assert len(raw_states) > 0
    normalizer = StateFeatureNormalizer()
    normalizer.fit(np.array(raw_states))
    assert normalizer.is_fitted
    assert not np.isnan(normalizer.mean).any()
    assert not np.isnan(normalizer.std).any()


def test_normalizer_fit_raises_on_nan_input():
    norm = StateFeatureNormalizer()
    nan_data = np.random.randn(10, STATE_DIM)
    nan_data[2, 5] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        norm.fit(nan_data)
