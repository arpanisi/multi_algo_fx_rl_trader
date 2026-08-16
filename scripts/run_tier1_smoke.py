"""
Tier 1 — Smoke Test Script.
Runs 3 pairs (EURUSD + 2 context pairs) over 2 weeks of Train window with PPO under zero-commission.
Confirms end-to-end execution of data, state assembly, environment, SB3 training, and evaluation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.pipeline import generate_synthetic_ticks, resample_15m_bars, compute_chronological_splits
from src.features.state import StateBuilder, StateFeatureNormalizer
from src.environment.trading_env import FxTradingEnv
from src.models.train import train_agent
from src.evaluation.metrics import compute_all_performance_metrics
from src.models.tuning import evaluate_model_on_env

def main():
    print("=== Launching Tier 1 FX RL Smoke Test ===")
    
    # Generate 2 weeks of synthetic 1m data for 3 pairs
    records = []
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    dates = pd.date_range("2020-01-01", "2020-01-14", freq="15min", tz="UTC")
    dates = dates[dates.dayofweek < 5]

    for p in pairs:
        df_1m = generate_synthetic_ticks(p, start_date="2020-01-01", end_date="2020-01-14")
        df_15m = resample_15m_bars(df_1m)
        df_15m["pair"] = p
        records.append(df_15m)

    panel_df = pd.concat(records)
    
    # Build state normalizer on train slice
    sb = StateBuilder(panel_df)
    train_ts = sb.timestamps[:100]
    
    raw_states = []
    for ts in train_ts[30:]:
        try:
            raw_states.append(sb.get_state_vector(ts))
        except ValueError:
            continue
    
    normalizer = StateFeatureNormalizer()
    normalizer.fit(np.array(raw_states))
    print(f"State vector length: {sb.state_dim}")
    print(f"Context pairs present: {sb.context_pairs_present}")

    # Environment
    env = FxTradingEnv(panel_df, timestamps=sb.timestamps, normalizer=normalizer, use_commission=False)
    print(f"Environment observation length: {env.observation_space.shape[0]}")
    
    print("Training PPO Agent (1 pass smoke)...")
    model = train_agent("PPO", env, passes=1)
    
    sr, rewards = evaluate_model_on_env(model, env)
    metrics = compute_all_performance_metrics(rewards, name="Tier1_PPO_Smoke")

    print("\n--- Tier 1 Smoke Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTier 1 Smoke Test completed successfully!")

if __name__ == "__main__":
    main()
