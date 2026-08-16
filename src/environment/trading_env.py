"""
Gymnasium-compliant FX Trading Environment.
Enforces exact MOC action timing, portfolio value tracking V_t, double-charged reversal transaction costs,
and portfolio log-return reward r_t = log(V_t / V_{t-1}).
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config.settings import (
    TRADED_PAIR, SPREAD_PRICE_UNITS, COMMISSION_FRAC
)
from src.features.state import StateBuilder, StateFeatureNormalizer

class FxTradingEnv(gym.Env):
    """
    FX Trading Gymnasium Environment for EURUSD trading with context pairs.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        panel_df: pd.DataFrame,
        timestamps: list[pd.Timestamp] = None,
        normalizer: StateFeatureNormalizer = None,
        use_commission: bool = True
    ):
        super().__init__()
        self.panel_df = panel_df
        self.state_builder = StateBuilder(panel_df)
        self.state_dim = self.state_builder.state_dim
        
        all_timestamps = self.state_builder.timestamps.tolist()
        # Filter timestamps where state vector can be built without raising ValueError (data-driven warmup check)
        valid_timestamps = []
        for ts in all_timestamps:
            try:
                self.state_builder.get_state_vector(ts)
                valid_timestamps.append(ts)
            except ValueError:
                continue
        
        if timestamps is not None:
            self.timestamps = [ts for ts in timestamps if ts in valid_timestamps]
        else:
            self.timestamps = valid_timestamps

        if not self.timestamps:
            raise ValueError("No valid timestamps provided for environment after warmup filtering.")

        self.normalizer = normalizer
        self.use_commission = use_commission

        # Discrete Action Space: 0 = SELL (-1), 1 = CLOSE (0), 2 = BUY (+1)
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: -1.0, 1: 0.0, 2: 1.0}

        # Observation Space: panel-specific continuous state features.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 1.0
        self.current_position = 0.0
        self.entry_price = None

        ts = self.timestamps[self.current_step]
        raw_state = self.state_builder.get_state_vector(
            timestamp=ts,
            current_position=self.current_position,
            entry_price=self.entry_price
        )

        obs = self.normalizer.transform(raw_state) if self.normalizer else raw_state
        return obs.astype(np.float32), {}

    def step(self, action: int):
        target_position = self.action_map[action]
        ts_current = self.timestamps[self.current_step]

        # Check if episode is ending
        is_last_step = (self.current_step >= len(self.timestamps) - 1)

        # Prices
        p_current = float(self.state_builder.pivot_close.loc[ts_current, TRADED_PAIR])

        # 1. Pre-cost portfolio value V_t^{pre-cost}
        if self.current_position != 0.0 and self.current_step > 0 and p_current > 0:
            ts_prev = self.timestamps[self.current_step - 1]
            p_prev = float(self.state_builder.pivot_close.loc[ts_prev, TRADED_PAIR])
            price_ratio = p_current / p_prev if p_prev > 0 else 1.0
            v_pre_cost = self.portfolio_value * np.exp(self.current_position * np.log(price_ratio))
        else:
            v_pre_cost = self.portfolio_value

        # 2. Transaction cost calculation
        # Position change count (1 trade for open/close, 2 trades for direct reversal)
        num_trades = 0
        if self.current_position != target_position:
            if self.current_position == 0.0 or target_position == 0.0:
                num_trades = 1
            else:  # Reversal short -> long or long -> short
                num_trades = 2

        spread_frac = SPREAD_PRICE_UNITS / p_current if p_current > 0 else 0.0
        comm_frac = COMMISSION_FRAC if self.use_commission else 0.0
        total_cost_frac = num_trades * (spread_frac + comm_frac)

        # 3. Post-cost portfolio value V_t
        new_portfolio_value = max(1e-12, v_pre_cost * (1.0 - total_cost_frac))

        # 4. Reward r_t = log(V_t / V_{t-1})
        reward = float(np.log(new_portfolio_value / max(1e-12, self.portfolio_value)))

        # Update position & entry price
        if target_position != self.current_position:
            if target_position != 0.0:
                self.entry_price = p_current
            else:
                self.entry_price = None
            self.current_position = target_position

        self.portfolio_value = new_portfolio_value
        self.current_step += 1

        terminated = is_last_step
        truncated = False

        if not terminated:
            ts_next_step = self.timestamps[self.current_step]
            raw_state = self.state_builder.get_state_vector(
                timestamp=ts_next_step,
                current_position=self.current_position,
                entry_price=self.entry_price
            )
            obs = self.normalizer.transform(raw_state) if self.normalizer else raw_state
        else:
            obs = np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.current_position,
            "num_trades": num_trades
        }

        return obs.astype(np.float32), reward, terminated, truncated, info
