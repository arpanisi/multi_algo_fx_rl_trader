# Multi-Algorithm FX Reinforcement Learning Trader Runbook

This project runs from `fx-market/multi_algo_fx_rl_trader`.

---

## 1. Overview & Data Contract

### Target Outcome
Built a reinforcement-learning FX trading system comparing value-based (`DQN`) and policy-gradient (`PPO`, `A2C`) algorithms via Stable-Baselines3 with Optuna hyperparameter tuning. Trades `EURUSD` using 11 other major currency pairs (`GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`, `USDCAD`, `NZDUSD`, `EURJPY`, `GBPJPY`, `EURCHF`, `EURGBP`, `AUDJPY`) as cross-market state context, operating on 15-minute bars aggregated from 1-minute HistData FX data. Features explicit double-charged transaction cost modeling (1.5 pips spread + 0.5 bps fee) and chronological out-of-sample evaluation against passive `EURUSD` Buy-and-Hold and rule-based SMA crossover baselines.

### Data and Output Contract
- `data/` holds local or downloaded dataset parquet files (`histdata_fx_1m/`, `fx_panel_15m.parquet`). Ignored by Git; only `.gitkeep` tracked.
- `outputs/` holds generated benchmark reports, Optuna trial logs, and `multi_algo_eval_results.csv`. Ignored by Git; only `.gitkeep` tracked.
- `src/` holds all implementation code.
- `tests/` holds all automated unit tests.

---

## 2. Environment Setup

The project runs using the local Python virtual environment `.venv`.

### Virtual Environment Creation & Dependencies
```bash
# Create venv if not present
python3 -m venv .venv

# Activate environment and install dependencies
./.venv/bin/python -m pip install -r requirements.txt
```

### Environment Configuration (`.env`)
Create or edit `.env` inside `fx-market/multi_algo_fx_rl_trader/`:
```env
PYTHONPATH=.
```

---

## 3. Automated Test Verification

Before running execution tiers, verify system components via pytest:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/
```

Expected output: `11 passed`.

---

## 4. Execution Tiers

### Tier 1 — Smoke Test (Local Plumbing Check)
Exercises data aggregation, 35-feature state representation, Gymnasium `FxTradingEnv`, PPO agent training, and evaluation metrics over 3 pairs for 2 weeks.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier1_smoke.py
```

### Tier 2 — Small Test (12-Pair Universe & Reduced Optuna)
Runs full 12-pair universe, 1-year Train window sub-slice, all 3 algorithms (`DQN`, `PPO`, `A2C`), both cost settings (`zero-commission` and `commission`), reduced Optuna budget (5 trials each), and evaluation against Buy-and-Hold and SMA Crossover baselines.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier2_small.py
```

### Tier 3 — Full-Scale Benchmark Run
Runs the complete Train / Validation (3 yrs) / Evaluation (3 yrs final) chronological splits across all 12 pairs, all 3 algorithms (`DQN`, `PPO`, `A2C`), both cost settings, full Optuna budgets (40 trials for DQN, 20 for PPO/A2C), and 20-pass training budget, outputting results to `outputs/multi_algo_eval_results.csv`.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier3_full.py
```

---

## 5. Vast.ai GPU Deployment Guide (for Accelerated RL Training & Optuna Sweeps)

For accelerating SB3 multi-algorithm training (`DQN`, `PPO`, `A2C`) and Optuna hyperparameter sweeps on GPU nodes via Vast.ai:

### Instance Provisioning on Vast.ai
1. Select an instance with an **NVIDIA A10G (24GB)**, **RTX 4090 (24GB)**, or **A100 (40GB/80GB)**.
2. Select an official PyTorch / CUDA 12.1+ Docker image (e.g. `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`).
3. Launch instance and SSH into the node.

### Environment Setup on Vast GPU Node
```bash
# Clone or upload repository to GPU node
cd fx-market/multi_algo_fx_rl_trader

# Install PyTorch with CUDA acceleration and project requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Running Accelerated Optuna & Training Sweeps on GPU
Execute Tier 3 benchmark with CUDA hardware acceleration:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_tier3_full.py
```
