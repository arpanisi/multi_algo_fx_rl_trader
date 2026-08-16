"""
Step 7 Rule-Based Baseline module (SMA(10)/SMA(50) crossover).
Evaluated through exact Step 4 environment for sanity-checking RL algorithms.
"""
import numpy as np
import pandas as pd
from config.settings import TRADED_PAIR
from src.features.indicators import compute_sma
from src.environment.trading_env import FxTradingEnv
from src.evaluation.metrics import compute_all_performance_metrics

def generate_sma_crossover_actions(panel_df: pd.DataFrame, fast_window: int = 10, slow_window: int = 50) -> pd.Series:
    """
    Generates action signals (0 = SELL, 1 = CLOSE, 2 = BUY) for SMA(10)/SMA(50) crossover on EURUSD close.
    Undefined comparisons and exact ties hold the current position by reissuing its action.
    """
    pivot_close = panel_df.pivot(columns="pair", values="close")[TRADED_PAIR]
    sma_fast = compute_sma(pivot_close, window=fast_window)
    sma_slow = compute_sma(pivot_close, window=slow_window)

    actions = pd.Series(index=pivot_close.index, dtype=int)
    current_action = 1
    for ts in pivot_close.index:
        if sma_fast.loc[ts] > sma_slow.loc[ts]:
            current_action = 2  # BUY (+1)
        elif sma_fast.loc[ts] < sma_slow.loc[ts]:
            current_action = 0  # SELL (-1)
        actions.loc[ts] = current_action
    return actions

def evaluate_sma_baseline_on_env(env: FxTradingEnv, name: str = "SMA_Crossover_Baseline") -> dict:
    """
    Evaluates the SMA(10)/SMA(50) crossover baseline through the FxTradingEnv.
    """
    actions_series = generate_sma_crossover_actions(env.panel_df)
    obs, info = env.reset()
    done = False
    rewards = []

    while not done:
        ts = env.timestamps[env.current_step]
        act = actions_series.get(ts, 1)  # Default CLOSE
        obs, reward, terminated, truncated, info = env.step(act)
        rewards.append(reward)
        done = terminated or truncated

    metrics = compute_all_performance_metrics(rewards, name=name)
    return metrics
