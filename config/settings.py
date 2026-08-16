"""
Global system settings and constants for fx-market multi-algo RL trader.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

TRADED_PAIR = "EURUSD"
CONTEXT_PAIRS = [
    "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "EURCHF", "EURGBP", "AUDJPY"
]
ALL_PAIRS = [TRADED_PAIR] + CONTEXT_PAIRS

HF_DATASET_ID = "elthariel/histdata_fx_1m"
BAR_RESAMPLE_FREQ = "15min"
MAX_FORWARD_FILL_BARS = 4  # 1 hour max

BARS_PER_YEAR = 24960  # 96 bars/day * 5 days/week * 52 weeks

# --- Cost Parameters ---
SPREAD_PIPS = 1.5
SPREAD_PRICE_UNITS = 0.00015
COMMISSION_BPS = 0.5
COMMISSION_FRAC = 0.00005

# --- State Parameters ---
STATE_DIM = 35
EURUSD_RETURNS_LOOKBACK = 8

# --- Optuna & Training Defaults ---
TRAIN_EPISODES_PASSES = 20
NET_ARCH = [128, 128]

OPTUNA_TRIALS_DQN = 40
OPTUNA_TRIALS_PPO = 20
OPTUNA_TRIALS_A2C = 20
