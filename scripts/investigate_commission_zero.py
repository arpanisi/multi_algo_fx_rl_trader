"""
Investigation script for the commission-mode all-zero anomaly.

Mirrors scripts/run_tier2_small.py's exact data pipeline, environment
construction, tuning, and training calls (same functions, same arguments,
same seeds) for COMMISSION MODE ONLY, across all three algorithms, and adds
direct fill-count instrumentation (summing info["num_trades"] from the real
environment step() return, not just reading the summary Sharpe/MDD) over a
full deterministic post-training pass on both the validation and evaluation
windows. This does not modify any RL training/tuning/environment code -- it
only calls the same functions the runbook's own script calls, with the same
parameters, and adds observation on top.
"""
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import BARS_PER_YEAR, CONTEXT_PAIRS
from src.data.pipeline import build_aligned_15m_panel, compute_chronological_splits
from src.features.state import StateBuilder, StateFeatureNormalizer
from src.environment.trading_env import FxTradingEnv
from src.models.train import train_agent
from src.models.tuning import tune_algorithm_hyperparameters
from src.evaluation.metrics import compute_all_performance_metrics


def tier2_train_slice(train_timestamps, bars_per_year=BARS_PER_YEAR):
    return train_timestamps[:bars_per_year]


def instrumented_pass(model, env, label):
    """Runs one full deterministic pass and returns (rewards, fill_log)."""
    obs, info = env.reset()
    done = False
    rewards = []
    fill_log = []  # (step, action, target_position, num_trades, position_after)
    step_i = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info["num_trades"] > 0:
            fill_log.append((step_i, action, info["position"], info["num_trades"]))
        step_i += 1
        done = terminated or truncated
    total_fills = sum(f[3] for f in fill_log)
    print(f"[{label}] steps={step_i} real_fill_events={len(fill_log)} total_trade_units={total_fills}")
    if fill_log:
        print(f"[{label}] first 5 fill events (step, action, position_after, num_trades): {fill_log[:5]}")
        print(f"[{label}] last 5 fill events: {fill_log[-5:]}")
    else:
        print(f"[{label}] NO FILLS AT ALL across the full deterministic pass.")
    # action distribution regardless of whether it produced a fill
    return rewards, fill_log


def main():
    print("=== Commission-mode zero-result investigation ===")
    panel_df = build_aligned_15m_panel()
    splits = compute_chronological_splits(panel_df)

    sb = StateBuilder(panel_df)
    train_ts = tier2_train_slice(splits["train_timestamps"])
    val_ts = splits["val_timestamps"][:1000]
    eval_ts = splits["eval_timestamps"][:1000]

    raw_states = []
    for ts in train_ts[30:]:
        try:
            raw_states.append(sb.get_state_vector(ts))
        except ValueError:
            continue
    normalizer = StateFeatureNormalizer()
    normalizer.fit(np.array(raw_states))

    train_env = FxTradingEnv(panel_df, timestamps=train_ts, normalizer=normalizer, use_commission=True)
    val_env = FxTradingEnv(panel_df, timestamps=val_ts, normalizer=normalizer, use_commission=True)
    eval_env = FxTradingEnv(panel_df, timestamps=eval_ts, normalizer=normalizer, use_commission=True)

    for algo in ["DQN", "PPO", "A2C"]:
        print(f"\n--- {algo} (commission) ---")
        best_params = tune_algorithm_hyperparameters(
            algorithm=algo, train_env=train_env, val_env=val_env, n_trials=5, passes=3,
        )
        print(f"[{algo}] Best Params: {best_params}")

        model = train_agent(algorithm=algo, env=train_env, passes=5, hyperparams=best_params)

        val_rewards, val_fills = instrumented_pass(model, val_env, f"{algo}/val")
        eval_rewards, eval_fills = instrumented_pass(model, eval_env, f"{algo}/eval")

        val_metrics = compute_all_performance_metrics(val_rewards, name=f"{algo}_commission_val")
        eval_metrics = compute_all_performance_metrics(eval_rewards, name=f"{algo}_commission_eval")
        print(f"[{algo}] val_metrics={val_metrics}")
        print(f"[{algo}] eval_metrics={eval_metrics}")

        # Raw action distribution over the validation pass, independent of fills,
        # to distinguish "policy always outputs CLOSE" from "policy outputs BUY/SELL
        # but the environment/cost model still nets to zero portfolio movement".
        obs, _ = val_env.reset()
        done = False
        actions_taken = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(int(action))
            obs, reward, terminated, truncated, info = val_env.step(int(action))
            done = terminated or truncated
        from collections import Counter
        print(f"[{algo}] val action distribution (0=SELL,1=CLOSE,2=BUY): {Counter(actions_taken)}")


if __name__ == "__main__":
    main()
