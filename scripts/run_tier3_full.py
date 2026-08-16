"""
Tier 3 — Full-Scale Benchmark Run.
Full Train/Validation/Evaluation split, all 12 pairs, DQN, PPO, A2C, both cost settings,
full Optuna budgets (40 trials for DQN, 20 for PPO/A2C), full 20-pass training budget.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import OUTPUTS_DIR, OPTUNA_TRIALS_DQN, OPTUNA_TRIALS_PPO, OPTUNA_TRIALS_A2C
from src.data.pipeline import build_aligned_15m_panel, compute_chronological_splits
from src.features.state import StateBuilder, StateFeatureNormalizer
from src.environment.trading_env import FxTradingEnv
from src.models.train import train_agent
from src.models.tuning import tune_algorithm_hyperparameters
from src.evaluation.evaluate import run_full_evaluation_suite

def main():
    print("=== Launching Tier 3 Full FX RL Benchmark Run ===")
    panel_df = build_aligned_15m_panel()
    splits = compute_chronological_splits(panel_df)

    sb = StateBuilder(panel_df)
    train_ts = splits["train_timestamps"]
    val_ts = splits["val_timestamps"]
    eval_ts = splits["eval_timestamps"]

    print(f"Train Range: {splits['train_start']} -> {splits['train_end']} ({len(train_ts)} bars)")
    print(f"Val Range:   {splits['val_start']} -> {splits['val_end']} ({len(val_ts)} bars)")
    print(f"Eval Range:  {splits['eval_start']} -> {splits['eval_end']} ({len(eval_ts)} bars)")

    # Fit Z-score normalizer on Train split only
    raw_states = []
    for ts in train_ts[30:]:
        try:
            raw_states.append(sb.get_state_vector(ts))
        except ValueError:
            continue
    
    normalizer = StateFeatureNormalizer()
    normalizer.fit(np.array(raw_states))

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
    trial_budgets = {
        "DQN": OPTUNA_TRIALS_DQN,
        "PPO": OPTUNA_TRIALS_PPO,
        "A2C": OPTUNA_TRIALS_A2C
    }
    trained_models = {}

    for algo in algos:
        for cost_setting in ["zero-commission", "commission"]:
            print(f"\n=======================================================")
            print(f"Tuning & Training {algo} under {cost_setting}")
            print(f"=======================================================")
            best_params = tune_algorithm_hyperparameters(
                algorithm=algo,
                train_env=train_envs[cost_setting],
                val_env=val_envs[cost_setting],
                n_trials=trial_budgets[algo],
                passes=10
            )
            print(f"Optimal Params for {algo} ({cost_setting}): {best_params}")

            print(f"Training {algo} for 20 full passes over Train window...")
            model = train_agent(
                algorithm=algo,
                env=train_envs[cost_setting],
                passes=20,
                hyperparams=best_params
            )
            trained_models[(algo, cost_setting)] = model

    # Evaluate full suite
    print("\nExecuting Full Out-of-Sample Evaluation Suite...")
    df_report = run_full_evaluation_suite(trained_models, val_envs, eval_envs)
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "multi_algo_eval_results.csv"
    df_report.to_csv(output_path, index=False)

    print("\n================ FINAL TIER 3 BENCHMARK RESULTS TABLE ================")
    print(df_report.to_string(index=False))
    print(f"\nFinal evaluation report saved to {output_path}")

if __name__ == "__main__":
    main()
