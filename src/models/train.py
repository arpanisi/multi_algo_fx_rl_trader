"""
SB3 Model Training module for DQN, PPO, and A2C algorithms.
"""
import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
from config.settings import NET_ARCH, TRAIN_EPISODES_PASSES
from src.environment.trading_env import FxTradingEnv

def create_sb3_model(
    algorithm: str,
    env: FxTradingEnv,
    hyperparams: dict = None,
    seed: int = 42
):
    """
    Instantiates an SB3 model (DQN, PPO, or A2C) with a 2x128 MLP architecture.
    """
    hyperparams = hyperparams or {}
    algo_upper = algorithm.upper()

    policy_kw = dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH) if algo_upper in ["PPO", "A2C"] else NET_ARCH)

    if algo_upper == "DQN":
        learning_rate = hyperparams.get("learning_rate", 1e-4)
        buffer_size = hyperparams.get("buffer_size", 10000)
        batch_size = hyperparams.get("batch_size", 64)
        gamma = hyperparams.get("gamma", 0.99)
        exploration_fraction = hyperparams.get("exploration_fraction", 0.2)
        final_eps = hyperparams.get("final_eps", 0.02)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=final_eps,
            policy_kwargs=dict(net_arch=NET_ARCH),
            verbose=0,
            seed=seed
        )
    elif algo_upper == "PPO":
        learning_rate = hyperparams.get("learning_rate", 3e-4)
        n_steps = hyperparams.get("n_steps", 128)
        batch_size = hyperparams.get("batch_size", 64)
        gamma = hyperparams.get("gamma", 0.99)
        gae_lambda = hyperparams.get("gae_lambda", 0.95)
        clip_range = hyperparams.get("clip_range", 0.2)
        ent_coef = hyperparams.get("ent_coef", 0.0)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)),
            verbose=0,
            seed=seed
        )
    elif algo_upper == "A2C":
        learning_rate = hyperparams.get("learning_rate", 7e-4)
        n_steps = hyperparams.get("n_steps", 5)
        gamma = hyperparams.get("gamma", 0.99)
        gae_lambda = hyperparams.get("gae_lambda", 1.0)
        ent_coef = hyperparams.get("ent_coef", 0.0)

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)),
            verbose=0,
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model

def train_agent(
    algorithm: str,
    env: FxTradingEnv,
    passes: int = TRAIN_EPISODES_PASSES,
    hyperparams: dict = None,
    seed: int = 42
):
    """
    Trains the SB3 agent for passes * len(env.timestamps) total timesteps.
    """
    model = create_sb3_model(algorithm, env, hyperparams, seed)
    total_timesteps = passes * len(env.timestamps)
    model.learn(total_timesteps=total_timesteps)
    return model
