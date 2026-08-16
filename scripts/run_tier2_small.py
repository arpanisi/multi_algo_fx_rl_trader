"""
Tier 2 — Small Test Script.
Full 12-pair universe, 1 calendar year Train window, 3 algorithms (DQN, PPO, A2C),
both cost settings (zero-commission and commission), reduced Optuna budget (5 trials each), 5 passes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import BARS_PER_YEAR
from config.settings import CONTEXT_PAIRS
from src.data.pipeline import build_aligned_15m_panel, compute_chronological_splits
from src.features.state import StateBuilder, StateFeatureNormalizer
from src.environment.trading_env import FxTradingEnv
from src.models.train import train_agent
from src.models.tuning import tune_algorithm_hyperparameters
from src.evaluation.evaluate import run_full_evaluation_suite

def tier2_train_slice(train_timestamps, bars_per_year=BARS_PER_YEAR):
    """First calendar year of the Train window (~bars_per_year bars, chronological order).
    Falls back to the full Train window when it is shorter than a calendar year
    (Tier 2 is a sub-slice of Train, never larger than Train itself)."""
    return train_timestamps[:bars_per_year]

def main():
    print("=== Launching Tier 2 FX RL Small Test ===")
    panel_df = build_aligned_15m_panel()
    splits = compute_chronological_splits(panel_df)

    sb = StateBuilder(panel_df)
    train_ts = tier2_train_slice(splits["train_timestamps"])  # First calendar year (~24,960 bars)
    val_ts = splits["val_timestamps"][:1000]
    eval_ts = splits["eval_timestamps"][:1000]

    # Fit Z-score normalizer on Train split only
    raw_states = []
    for ts in train_ts[30:]:
        try:
            raw_states.append(sb.get_state_vector(ts))
        except ValueError:
            continue
    
    normalizer = StateFeatureNormalizer()
    normalizer.fit(np.array(raw_states))
    print(f"State vector length: {sb.state_dim}", flush=True)
    print(f"Context pairs present ({len(sb.context_pairs_present)}): {sb.context_pairs_present}", flush=True)
    missing_context_pairs = [pair for pair in CONTEXT_PAIRS if pair not in sb.context_pairs_present]
    print(f"Missing context pairs: {missing_context_pairs}", flush=True)

    # Environments for zero-commission and commission
    train_envs = {
        "zero-commission": FxTradingEnv(panel_df, timestamps=train_ts, normalizer=normalizer, use_commission=False),
        "commission": FxTradingEnv(panel_df, timestamps=train_ts, normalizer=normalizer, use_commission=True)
    }
    val_envs = {
        "zero-commission": FxTradingEnv(panel_df, timestamps=val_ts, normalizer=normalizer, use_commission=False),
        "commission": FxTradingEnv(panel_df, timestamps=val_ts, normalizer=normalizer, use_commission=True)
    }
    eval_envs = {
        "zero-commission": FxTradingEnv(panel_df, timestamps=eval_ts, normalizer=normalizer, use_commission=False),
        "commission": FxTradingEnv(panel_df, timestamps=eval_ts, normalizer=normalizer, use_commission=True)
    }

    algos = ["DQN", "PPO", "A2C"]
    trained_models = {}

    for algo in algos:
        for cost_setting in ["zero-commission", "commission"]:
            print(f"\n--- Tuning & Training {algo} ({cost_setting}) ---")
            best_params = tune_algorithm_hyperparameters(
                algorithm=algo,
                train_env=train_envs[cost_setting],
                val_env=val_envs[cost_setting],
                n_trials=5,
                passes=3
            )
            print(f"Best Params for {algo} ({cost_setting}): {best_params}")

            model = train_agent(
                algorithm=algo,
                env=train_envs[cost_setting],
                passes=5,
                hyperparams=best_params
            )
            trained_models[(algo, cost_setting)] = model

    # Evaluate full suite
    print("\nRunning Evaluation Suite...")
    df_report = run_full_evaluation_suite(trained_models, val_envs, eval_envs)
    
    print("\n================ TIER 2 RESULTS TABLE ================")
    print(df_report.to_string(index=False))
    print("\nTier 2 Small Test completed successfully!")

if __name__ == "__main__":
    main()
