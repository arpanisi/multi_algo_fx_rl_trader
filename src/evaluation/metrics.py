"""
Evaluation metrics calculation module.
Annualized Sharpe ratio (sqrt(24960)), Sortino ratio, Cumulative Return, and Max Drawdown.
"""
import numpy as np
import pandas as pd
from config.settings import BARS_PER_YEAR

def calculate_annualized_sharpe(rewards: list[float], bars_per_year: int = BARS_PER_YEAR) -> float:
    """
    SR = sqrt(24960) * mean(r_t) / std(r_t)
    """
    rets = np.array(rewards)
    if len(rets) == 0:
        return 0.0
    std_val = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    if std_val == 0.0:
        return 0.0
    return float(np.sqrt(bars_per_year) * (np.mean(rets) / std_val))

def calculate_sortino_ratio(rewards: list[float], bars_per_year: int = BARS_PER_YEAR) -> float:
    """
    Sortino = sqrt(24960) * mean(r_t) / std(negative r_t)
    """
    rets = np.array(rewards)
    if len(rets) == 0:
        return 0.0
    downside_rets = rets[rets < 0]
    if len(downside_rets) <= 1:
        return 0.0
    downside_std = float(np.std(downside_rets, ddof=1))
    if downside_std == 0.0:
        return 0.0
    return float(np.sqrt(bars_per_year) * (np.mean(rets) / downside_std))

def calculate_cumulative_return(rewards: list[float]) -> float:
    """
    Cumulative Return = prod(exp(r_t)) - 1
    """
    rets = np.array(rewards)
    if len(rets) == 0:
        return 0.0
    return float(np.exp(np.sum(rets)) - 1.0)

def calculate_max_drawdown(rewards: list[float]) -> float:
    """
    MDD = min_t (Trough_t - Peak_t) / Peak_t
    """
    rets = np.array(rewards)
    if len(rets) == 0:
        return 0.0
    equity_curve = np.cumprod(np.exp(rets))
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    return float(np.min(drawdowns))

def compute_all_performance_metrics(rewards: list[float], name: str = "Policy") -> dict:
    """
    Returns complete metrics dictionary for reward series.
    """
    return {
        "name": name,
        "sharpe_ratio": calculate_annualized_sharpe(rewards),
        "sortino_ratio": calculate_sortino_ratio(rewards),
        "cumulative_return": calculate_cumulative_return(rewards),
        "max_drawdown": calculate_max_drawdown(rewards),
        "num_bars": len(rewards)
    }
