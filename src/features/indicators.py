"""
Technical indicators module for EURUSD bars.
Computes 10 fixed indicators strictly causally using historical price data through bar t.
"""
import numpy as np
import pandas as pd
from config.settings import BARS_PER_YEAR

def compute_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=window).mean()

def compute_ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average withheld until the full warmup window exists."""
    return close.ewm(span=window, min_periods=window, adjust=False).mean()

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0..100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -1.0 * delta.clip(upper=0.0)
    
    avg_gain = gain.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-12, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line minus Signal line."""
    ema_fast = compute_ema(close, window=fast)
    ema_slow = compute_ema(close, window=slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line - signal_line

def compute_bollinger_width(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    """Single-scalar Bollinger definition: normalized band width, (upper - lower) / SMA."""
    sma = compute_sma(close, window=window)
    rolling_std = close.rolling(window=window).std(ddof=1)
    upper = sma + k * rolling_std
    lower = sma - k * rolling_std
    return (upper - lower) / np.where(sma == 0, 1e-12, sma)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Single-scalar Stochastic definition: raw %K."""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    denom = highest_high - lowest_low
    denom_safe = np.where(denom == 0, 1e-12, denom)
    k_percent = 100.0 * (close - lowest_low) / denom_safe
    return k_percent

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Average Directional Index."""
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr = compute_atr(high, low, close, window=window)
    atr_safe = np.where(atr == 0, 1e-12, atr)
    
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0/window, min_periods=window, adjust=False).mean() / atr_safe
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0/window, min_periods=window, adjust=False).mean() / atr_safe
    
    di_sum = plus_di + minus_di
    di_sum_safe = np.where(di_sum == 0, 1e-12, di_sum)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum_safe
    adx = dx.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
    return adx

def compute_roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of Change."""
    prev = close.shift(window)
    prev_safe = np.where(prev == 0, 1e-12, prev)
    return 100.0 * (close - prev) / prev_safe

def compute_realized_volatility(close: pd.Series, window: int = 20, bars_per_year: int = BARS_PER_YEAR) -> pd.Series:
    """Annualized Realized Volatility."""
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window).std(ddof=1)
    return rolling_std * np.sqrt(bars_per_year)

def compute_all_eurusd_indicators(df_eurusd: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 10 indicators on EURUSD OHLC dataframe.
    Returns DataFrame with 10 indicator columns.
    """
    c = df_eurusd["close"]
    h = df_eurusd["high"]
    l = df_eurusd["low"]

    df_ind = pd.DataFrame(index=df_eurusd.index)
    df_ind["ind_sma20"] = compute_sma(c, 20)
    df_ind["ind_ema20"] = compute_ema(c, 20)
    df_ind["ind_rsi14"] = compute_rsi(c, 14)
    df_ind["ind_macd"] = compute_macd(c, 12, 26, 9)
    df_ind["ind_bollinger"] = compute_bollinger_width(c, 20, 2.0)
    df_ind["ind_atr14"] = compute_atr(h, l, c, 14)
    df_ind["ind_stochastic"] = compute_stochastic(h, l, c, 14)
    df_ind["ind_adx14"] = compute_adx(h, l, c, 14)
    df_ind["ind_roc10"] = compute_roc(c, 10)
    df_ind["ind_realized_vol20"] = compute_realized_volatility(c, 20)
    
    return df_ind
