"""
State Representation module building the per-panel state vector and applying
Train-window-fitted Z-score normalization with strict no-lookahead.
"""
import numpy as np
import pandas as pd
from config.settings import TRADED_PAIR, CONTEXT_PAIRS, EURUSD_RETURNS_LOOKBACK
from src.features.indicators import compute_all_eurusd_indicators


EURUSD_RETURN_FEATURES = EURUSD_RETURNS_LOOKBACK
TECHNICAL_INDICATOR_FEATURES = 10
CALENDAR_FEATURES = 4
POSITION_FEATURES = 1
UNREALIZED_RETURN_FEATURES = 1


def expected_state_dim(n_context_pairs: int) -> int:
    return (
        EURUSD_RETURN_FEATURES
        + int(n_context_pairs)
        + TECHNICAL_INDICATOR_FEATURES
        + CALENDAR_FEATURES
        + POSITION_FEATURES
        + UNREALIZED_RETURN_FEATURES
    )


class StateFeatureNormalizer:
    """
    Z-score normalizer fitted strictly on Train split data only.
    """
    def __init__(self):
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.is_fitted: bool = False

    @property
    def state_dim(self) -> int:
        if not self.is_fitted:
            raise ValueError("StateFeatureNormalizer has no state_dim before fit.")
        return int(self.mean.shape[0])

    def fit(self, state_matrix: np.ndarray):
        """Fits mean and std on raw state matrix."""
        if state_matrix.ndim != 2 or state_matrix.shape[0] == 0 or state_matrix.shape[1] == 0:
            raise ValueError("state_matrix passed to StateFeatureNormalizer.fit must be a non-empty 2D matrix")
        if np.isnan(state_matrix).any():
            raise ValueError("state_matrix passed to StateFeatureNormalizer.fit contains NaN values")
        self.mean = np.mean(state_matrix, axis=0)
        self.std = np.std(state_matrix, axis=0)
        if np.isnan(self.mean).any() or np.isnan(self.std).any():
            raise ValueError("Fitted normalizer mean or std contains NaN values")
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1.0, self.std)
        self.is_fitted = True

    def transform(self, state_vector: np.ndarray) -> np.ndarray:
        """Applies fitted z-score normalization to state vector or matrix."""
        if not self.is_fitted:
            raise ValueError("StateFeatureNormalizer must be fitted on Training split before transforming.")
        if state_vector.shape[-1] != self.state_dim:
            raise ValueError(f"State dimension mismatch: normalizer fitted on {self.state_dim}, got {state_vector.shape[-1]}")
        return (state_vector - self.mean) / self.std

class StateBuilder:
    """
    Constructs the state vector for every bar in the panel.
    """
    def __init__(self, panel_df: pd.DataFrame):
        self.panel_df = panel_df
        self._prepare_data()

    def _prepare_data(self):
        """Precomputes returns, indicators, and calendar features across the timeline."""
        # Pivot close prices: columns = pairs, index = timestamp
        self.pivot_close = self.panel_df.pivot(columns="pair", values="close").sort_index()
        self.pivot_high = self.panel_df.pivot(columns="pair", values="high").sort_index()
        self.pivot_low = self.panel_df.pivot(columns="pair", values="low").sort_index()
        self.context_pairs_present = [pair for pair in CONTEXT_PAIRS if pair in self.pivot_close.columns]
        self.state_dim = expected_state_dim(len(self.context_pairs_present))
        
        # Log returns
        self.log_returns = np.log(self.pivot_close / self.pivot_close.shift(1))

        # EURUSD technical indicators
        df_eurusd = pd.DataFrame({
            "close": self.pivot_close[TRADED_PAIR],
            "high": self.pivot_high[TRADED_PAIR],
            "low": self.pivot_low[TRADED_PAIR]
        })
        self.indicators_df = compute_all_eurusd_indicators(df_eurusd)
        self.timestamps = self.pivot_close.index

    def get_state_vector(
        self,
        timestamp: pd.Timestamp,
        current_position: float = 0.0,
        entry_price: float = None
    ) -> np.ndarray:
        """
        Builds raw state vector for bar at timestamp.
        """
        if timestamp not in self.timestamps:
            raise KeyError(f"Timestamp {timestamp} not found in panel.")

        idx = self.timestamps.get_loc(timestamp)
        if idx < EURUSD_RETURNS_LOOKBACK:
            # Need warmup history
            raise ValueError(f"Bar at index {idx} ({timestamp}) is within indicator warmup period.")

        # 1. EURUSD log returns (8 values: t, t-1, ..., t-7)
        eurusd_rets = self.log_returns[TRADED_PAIR].iloc[idx - EURUSD_RETURNS_LOOKBACK + 1 : idx + 1].values[::-1]

        # 2. Context-pair log returns at t, one feature for each configured context pair present
        # in the actual panel. Absent pairs are omitted, not zero-padded.
        context_rets = []
        for pair in self.context_pairs_present:
            val = self.log_returns[pair].iloc[idx]
            if pd.isna(val):
                raise ValueError(f"Bar at index {idx} ({timestamp}) contains NaN return for context pair {pair}.")
            context_rets.append(float(val))

        # 3. Technical indicators (10 values at t)
        ind_vals = self.indicators_df.iloc[idx].values
        if np.isnan(ind_vals).any():
            raise ValueError(f"Bar at index {idx} ({timestamp}) contains NaN indicator values (within indicator warmup period).")

        # 4. Calendar features (4 values: sin/cos hour and dayofweek)
        hour = timestamp.hour + timestamp.minute / 60.0
        dow = timestamp.dayofweek
        cal_features = [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * dow / 7.0),
            np.cos(2.0 * np.pi * dow / 7.0)
        ]

        # 5. Current position (1 value: -1, 0, 1)
        pos_feature = [float(current_position)]

        # 6. Unrealized position return (1 value)
        if current_position == 0.0 or entry_price is None or entry_price <= 0:
            unrealized_ret = [0.0]
        else:
            current_price = self.pivot_close[TRADED_PAIR].iloc[idx]
            unrealized_ret = [float(current_position * np.log(current_price / entry_price))]

        # Concatenate into the actual panel-specific state vector.
        raw_state = np.concatenate([
            eurusd_rets,
            context_rets,
            ind_vals,
            cal_features,
            pos_feature,
            unrealized_ret
        ])

        if len(raw_state) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(raw_state)}")

        if np.isnan(raw_state).any():
            raise ValueError(f"Bar at index {idx} ({timestamp}) contains NaN state features.")

        return raw_state
