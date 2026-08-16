"""
Evaluation pipeline module building the 6-combination results table
(3 algorithms x 2 cost settings) alongside Buy-and-Hold and SMA Crossover baselines.
"""
import numpy as np
import pandas as pd
from config.settings import TRADED_PAIR, SPREAD_PRICE_UNITS
from src.environment.trading_env import FxTradingEnv
from src.evaluation.metrics import compute_all_performance_metrics
from src.evaluation.baseline import evaluate_sma_baseline_on_env
from src.models.tuning import evaluate_model_on_env

def evaluate_buy_and_hold_benchmark(env: FxTradingEnv, name: str = "Buy_and_Hold_Benchmark") -> dict:
    """
    Buy-and-Hold: Buy 1 unit (+1.0) on first step and hold through end of window.
    Pays spread cost once on opening trade.
    """
    obs, info = env.reset()
    done = False
    rewards = []

    # Step 1: Open BUY position (action = 2)
    obs, reward, terminated, truncated, info = env.step(2)
    rewards.append(reward)
    done = terminated or truncated

    # Remaining steps: Hold position (action = 2)
    while not done:
        obs, reward, terminated, truncated, info = env.step(2)
        rewards.append(reward)
        done = terminated or truncated

    metrics = compute_all_performance_metrics(rewards, name=name)
    return metrics

def run_full_evaluation_suite(
    trained_models: dict, # {(algo, cost_setting): model}
    val_envs: dict,       # {cost_setting: FxTradingEnv}
    eval_envs: dict       # {cost_setting: FxTradingEnv}
) -> pd.DataFrame:
    """
    Runs full evaluation across all 6 algorithm x cost setting combinations
    over Validation and Evaluation windows, plus Buy-and-Hold and SMA Crossover baselines.
    """
    records = []

    # 1. Evaluate RL Models
    for (algo, cost_setting), model in trained_models.items():
        val_env = val_envs[cost_setting]
        eval_env = eval_envs[cost_setting]

        val_sr, val_rewards = evaluate_model_on_env(model, val_env)
        eval_sr, eval_rewards = evaluate_model_on_env(model, eval_env)

        val_metrics = compute_all_performance_metrics(val_rewards, name=f"{algo}_{cost_setting}")
        eval_metrics = compute_all_performance_metrics(eval_rewards, name=f"{algo}_{cost_setting}")

        rec = {
            "Algorithm": algo,
            "Cost_Setting": cost_setting,
            "Val_Sharpe": val_metrics["sharpe_ratio"],
            "Val_Sortino": val_metrics["sortino_ratio"],
            "Val_CumRet": val_metrics["cumulative_return"],
            "Val_MDD": val_metrics["max_drawdown"],
            "Eval_Sharpe": eval_metrics["sharpe_ratio"],
            "Eval_Sortino": eval_metrics["sortino_ratio"],
            "Eval_CumRet": eval_metrics["cumulative_return"],
            "Eval_MDD": eval_metrics["max_drawdown"]
        }
        records.append(rec)

    # 2. Evaluate Rule-Based Baseline (SMA Crossover)
    for cost_setting in ["zero-commission", "commission"]:
        val_sma = evaluate_sma_baseline_on_env(val_envs[cost_setting], name=f"SMA_Crossover_{cost_setting}")
        eval_sma = evaluate_sma_baseline_on_env(eval_envs[cost_setting], name=f"SMA_Crossover_{cost_setting}")
        rec_sma = {
            "Algorithm": "SMA_Baseline",
            "Cost_Setting": cost_setting,
            "Val_Sharpe": val_sma["sharpe_ratio"],
            "Val_Sortino": val_sma["sortino_ratio"],
            "Val_CumRet": val_sma["cumulative_return"],
            "Val_MDD": val_sma["max_drawdown"],
            "Eval_Sharpe": eval_sma["sharpe_ratio"],
            "Eval_Sortino": eval_sma["sortino_ratio"],
            "Eval_CumRet": eval_sma["cumulative_return"],
            "Eval_MDD": eval_sma["max_drawdown"]
        }
        records.append(rec_sma)

    # 3. Evaluate Buy-and-Hold Benchmark
    val_bh = evaluate_buy_and_hold_benchmark(val_envs["zero-commission"])
    eval_bh = evaluate_buy_and_hold_benchmark(eval_envs["zero-commission"])

    rec_bh = {
        "Algorithm": "Buy_and_Hold",
        "Cost_Setting": "benchmark",
        "Val_Sharpe": val_bh["sharpe_ratio"],
        "Val_Sortino": val_bh["sortino_ratio"],
        "Val_CumRet": val_bh["cumulative_return"],
        "Val_MDD": val_bh["max_drawdown"],
        "Eval_Sharpe": eval_bh["sharpe_ratio"],
        "Eval_Sortino": eval_bh["sortino_ratio"],
        "Eval_CumRet": eval_bh["cumulative_return"],
        "Eval_MDD": eval_bh["max_drawdown"]
    }
    records.append(rec_bh)

    df_report = pd.DataFrame(records)
    return df_report
