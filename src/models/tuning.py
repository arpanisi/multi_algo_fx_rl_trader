"""
Optuna Hyperparameter Tuning module for tuning DQN, PPO, and A2C on Validation Sharpe.
Ensures strict validation isolation — evaluation data is physically unreachable.
"""
import optuna
import numpy as np
import pandas as pd
from config.settings import BARS_PER_YEAR
from src.environment.trading_env import FxTradingEnv
from src.models.train import train_agent

optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_model_on_env(model, val_env: FxTradingEnv) -> tuple[float, list[float]]:
    """
    Runs single full sequential pass on val_env using model.
    Returns (annualized_sharpe, reward_list).
    """
    obs, info = val_env.reset()
    done = False
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(int(action))
        rewards.append(reward)
        done = terminated or truncated

    rets = np.array(rewards)
    if len(rets) == 0 or np.std(rets) == 0:
        return 0.0, rewards

    sharpe = float(np.sqrt(BARS_PER_YEAR) * (np.mean(rets) / np.std(rets, ddof=1)))
    return sharpe, rewards

def tune_algorithm_hyperparameters(
    algorithm: str,
    train_env: FxTradingEnv,
    val_env: FxTradingEnv,
    n_trials: int = 20,
    passes: int = 5,
    seed: int = 42
) -> dict:
    """
    Tunes hyperparameter configuration for algorithm on val_env Sharpe using Optuna.
    """
    algo_upper = algorithm.upper()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        params = {
            "learning_rate": lr,
            "gamma": gamma,
            "batch_size": batch_size
        }

        if algo_upper == "DQN":
            params["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.1, 0.4)
            params["final_eps"] = trial.suggest_float("final_eps", 0.01, 0.1)
        elif algo_upper == "PPO":
            params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
            params["ent_coef"] = trial.suggest_float("ent_coef", 0.0, 0.02)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)
        elif algo_upper == "A2C":
            params["ent_coef"] = trial.suggest_float("ent_coef", 0.0, 0.02)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)

        try:
            model = train_agent(
                algorithm=algorithm,
                env=train_env,
                passes=passes,
                hyperparams=params,
                seed=seed + trial.number
            )
            val_sharpe, _ = evaluate_model_on_env(model, val_env)
            return val_sharpe
        except Exception as e:
            return -999.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return study.best_params
