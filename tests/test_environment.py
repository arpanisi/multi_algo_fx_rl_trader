"""
Unit tests for Step 4 Trading Environment.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.pipeline import generate_synthetic_ticks, resample_15m_bars
from src.environment.trading_env import FxTradingEnv
import src.environment.trading_env as trading_env_module

@pytest.fixture
def sample_panel():
    records = []
    pairs = ["EURUSD", "GBPUSD"]
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="15min", tz="UTC")
    dates = dates[dates.dayofweek < 5]

    for pair in pairs:
        prices = 1.10 + np.random.randn(len(dates)) * 0.0005
        df = pd.DataFrame({
            "open": prices, "high": prices + 0.0001, "low": prices - 0.0001,
            "close": prices, "volume": 100, "pair": pair
        }, index=dates)
        records.append(df)

    return pd.concat(records)

def test_environment_gym_interface(sample_panel):
    env = FxTradingEnv(sample_panel, use_commission=True)
    obs, info = env.reset()
    assert env.state_dim == 25
    assert obs.shape == (25,)
    assert env.observation_space.shape == (25,)

    # Take BUY action (2 = +1.0)
    obs, reward, terminated, truncated, info = env.step(2)
    assert "portfolio_value" in info
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)

def test_frictionless_holding(sample_panel):
    """An unchanged position incurs zero transaction costs across consecutive holding steps."""
    env = FxTradingEnv(sample_panel, use_commission=True)
    env.reset()
    
    # Open long position (BUY = 2)
    _, _, _, _, info1 = env.step(2)
    # Hold long position for 5 bars (BUY = 2)
    for _ in range(5):
        _, _, _, _, info_hold = env.step(2)
        assert info_hold["num_trades"] == 0

def test_reversal_double_trade_cost(sample_panel):
    """Direct reversal short -> long incurs 2 trades (double cost)."""
    env = FxTradingEnv(sample_panel, use_commission=True)
    env.reset()
    
    # Open short (SELL = 0)
    env.step(0)
    # Reverse to long (BUY = 2)
    _, _, _, _, info_rev = env.step(2)
    assert info_rev["num_trades"] == 2

def test_reward_and_portfolio_value_use_elapsed_price_interval(monkeypatch):
    """Hand-computed path: action at t sets next holding; reward at t realizes prior holding."""
    monkeypatch.setattr(trading_env_module, "SPREAD_PRICE_UNITS", 0.0)
    monkeypatch.setattr(trading_env_module, "COMMISSION_FRAC", 0.0)

    dates = pd.date_range("2020-01-01", periods=37, freq="15min", tz="UTC")
    eurusd_close = [1.0] * 33 + [1.10, 1.11, 1.10, 1.12]
    records = []
    for pair in ["EURUSD", "GBPUSD"]:
        prices = eurusd_close if pair == "EURUSD" else [1.30] * len(dates)
        records.append(pd.DataFrame({
            "open": prices,
            "high": np.asarray(prices) + 0.0001,
            "low": np.asarray(prices) - 0.0001,
            "close": prices,
            "volume": 100,
            "pair": pair,
        }, index=dates))

    env = FxTradingEnv(pd.concat(records), use_commission=False)
    env.reset()

    _, reward0, terminated0, _, info0 = env.step(2)
    assert not terminated0
    assert info0["portfolio_value"] == pytest.approx(1.0)
    assert reward0 == pytest.approx(0.0)

    expected_v1 = 1.11 / 1.10
    _, reward1, terminated1, _, info1 = env.step(2)
    assert not terminated1
    assert info1["portfolio_value"] == pytest.approx(expected_v1)
    assert reward1 == pytest.approx(np.log(expected_v1 / 1.0))

    expected_v2 = expected_v1 * (1.10 / 1.11)
    _, reward2, terminated2, _, info2 = env.step(1)
    assert not terminated2
    assert info2["portfolio_value"] == pytest.approx(expected_v2)
    assert reward2 == pytest.approx(np.log(expected_v2 / expected_v1))

    _, reward3, terminated3, _, info3 = env.step(0)
    assert terminated3
    assert info3["portfolio_value"] == pytest.approx(expected_v2)
    assert reward3 == pytest.approx(0.0)
    assert info3["position"] == -1.0
