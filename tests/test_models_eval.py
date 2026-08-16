"""
Unit tests for Steps 5, 6, and 7: Model Training, Optuna Tuning, Evaluation Metrics, and Baselines.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.pipeline import generate_synthetic_ticks, resample_15m_bars
from src.environment.trading_env import FxTradingEnv
from src.models.train import create_sb3_model, train_agent
from src.evaluation.metrics import (
    calculate_annualized_sharpe, calculate_sortino_ratio, calculate_cumulative_return,
    calculate_max_drawdown, compute_all_performance_metrics
)
from src.evaluation.baseline import evaluate_sma_baseline_on_env, generate_sma_crossover_actions
from src.evaluation.evaluate import evaluate_buy_and_hold_benchmark

@pytest.fixture
def sample_env():
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

    panel_df = pd.concat(records)
    return FxTradingEnv(panel_df, use_commission=True)

def test_sb3_model_instantiation(sample_env):
    dqn = create_sb3_model("DQN", sample_env)
    ppo = create_sb3_model("PPO", sample_env)
    a2c = create_sb3_model("A2C", sample_env)

    assert dqn is not None
    assert ppo is not None
    assert a2c is not None

def test_evaluation_metrics_calculations():
    # 10 bars of +0.001 log return
    rewards = [0.001] * 10
    sharpe = calculate_annualized_sharpe(rewards, bars_per_year=24960)
    cum_ret = calculate_cumulative_return(rewards)
    mdd = calculate_max_drawdown(rewards)

    assert cum_ret == pytest.approx(np.exp(0.01) - 1.0)
    assert mdd == pytest.approx(0.0)

def test_sma_crossover_baseline_execution(sample_env):
    metrics = evaluate_sma_baseline_on_env(sample_env, name="SMA_Test")
    assert "sharpe_ratio" in metrics
    assert "cumulative_return" in metrics
    assert metrics["num_bars"] > 0

def test_buy_and_hold_benchmark_execution(sample_env):
    metrics = evaluate_buy_and_hold_benchmark(sample_env, name="BH_Test")
    assert "sharpe_ratio" in metrics
    assert "cumulative_return" in metrics
    assert metrics["num_bars"] > 0

def test_sma_crossover_holds_position_on_warmup_and_ties():
    dates = pd.date_range("2020-01-01", periods=65, freq="15min", tz="UTC")
    close = pd.Series([100.0] * 50 + [110.0] * 10 + [100.0] * 5, index=dates)
    panel_df = pd.DataFrame({
        "open": close.values,
        "high": close.values + 0.1,
        "low": close.values - 0.1,
        "close": close.values,
        "volume": 100,
        "pair": "EURUSD",
    }, index=dates)

    actions = generate_sma_crossover_actions(panel_df, fast_window=10, slow_window=50)

    assert (actions.iloc[:49] == 1).all()
    first_buy_idx = actions[actions == 2].index[0]
    assert first_buy_idx >= dates[50]
    assert actions.loc[first_buy_idx] == 2
    tie_panel = panel_df.copy()
    tie_panel["close"] = 100.0
    tie_actions = generate_sma_crossover_actions(tie_panel, fast_window=10, slow_window=50)
    assert (tie_actions == 1).all()
