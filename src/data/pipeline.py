"""
Data Pipeline for HistData FX 1-minute to 15-minute bar panel aggregation,
12-pair intersection alignment, 4-bar forward fill, and chronological train/val/eval date splitting.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from config.settings import (
    DATA_DIR, ALL_PAIRS, TRADED_PAIR, BAR_RESAMPLE_FREQ, MAX_FORWARD_FILL_BARS,
    HF_DATASET_ID
)

HISTDATA_LOCAL_DIR = DATA_DIR / "histdata_fx_1m"
PANEL_CACHE_PATH = DATA_DIR / "fx_panel_15m.parquet"

def download_or_load_pair_ticks(symbol: str) -> pd.DataFrame:
    """
    Downloads or loads tick data for a symbol (e.g. 'EURUSD' -> folder 'eurusd/ticks.parquet').
    Returns a DataFrame with index = datetime (UTC), columns = [open, high, low, close, volume].
    """
    sym_lower = symbol.lower()
    local_pair_dir = HISTDATA_LOCAL_DIR / sym_lower
    local_pair_dir.mkdir(parents=True, exist_ok=True)
    file_path = local_pair_dir / "ticks.parquet"

    if file_path.exists():
        df = pd.read_parquet(file_path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
        return df.sort_index()

    print(f"Downloading HistData for {symbol} from Hugging Face...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=f"{sym_lower}/ticks.parquet",
            repo_type="dataset",
            local_dir=HISTDATA_LOCAL_DIR
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch real HistData ticks for {symbol}: {e}") from e

    df = pd.read_parquet(downloaded)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    return df.sort_index()

def download_or_load_pair_gaps(symbol: str) -> pd.DataFrame:
    """
    Downloads or loads reported no-trading gaps for a symbol.
    Returns columns [length, start, end] with UTC datetimes.
    """
    sym_lower = symbol.lower()
    local_pair_dir = HISTDATA_LOCAL_DIR / sym_lower
    local_pair_dir.mkdir(parents=True, exist_ok=True)
    file_path = local_pair_dir / "gaps.parquet"

    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        print(f"Downloading HistData gaps for {symbol} from Hugging Face...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=f"{sym_lower}/gaps.parquet",
                repo_type="dataset",
                local_dir=HISTDATA_LOCAL_DIR
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch real HistData gaps for {symbol}: {e}") from e
        df = pd.read_parquet(downloaded)

    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values("start") if "start" in df.columns else df

def generate_synthetic_ticks(symbol: str, start_date: str = "2015-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Generates synthetic tick data for offline testing or fallback.
    """
    dates = pd.date_range(start_date, end_date, freq="1min", tz="UTC")
    # Exclude weekends
    dates = dates[dates.dayofweek < 5]
    N = len(dates)
    
    base_price = 1.10 if "EUR" in symbol else (130.0 if "JPY" in symbol else 1.30)
    np.random.seed(42 + hash(symbol) % 1000)
    returns = np.random.randn(N) * 0.0001
    price_path = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": price_path,
        "high": price_path + 0.0001,
        "low": price_path - 0.0001,
        "close": price_path,
        "volume": np.random.randint(10, 500, size=N, dtype=np.uint64)
    }, index=dates)
    return df

def resample_15m_bars(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 1m bars to 15m bars:
    open=first, high=max, low=min, close=last, volume=sum.
    """
    resampled = df_1m.resample(BAR_RESAMPLE_FREQ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna(subset=["close"])
    return resampled

def build_gap_mask(master_index: pd.DatetimeIndex, gaps_df: pd.DataFrame) -> pd.Series:
    """
    Marks master-index bars covered by reported HistData no-trading gaps.
    """
    mask = pd.Series(False, index=master_index)
    if gaps_df.empty:
        return mask
    valid_gaps = gaps_df.dropna(subset=["start", "end"])
    if valid_gaps.empty:
        return mask

    starts = pd.to_datetime(valid_gaps["start"], utc=True).to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(valid_gaps["end"], utc=True).to_numpy(dtype="datetime64[ns]")
    index_values = master_index.to_numpy(dtype="datetime64[ns]")
    start_pos = np.searchsorted(index_values, starts, side="left")
    end_pos = np.searchsorted(index_values, ends, side="right")
    in_range = start_pos < end_pos
    start_pos = start_pos[in_range]
    end_pos = end_pos[in_range]
    if len(start_pos) == 0:
        return mask

    deltas = np.zeros(len(master_index) + 1, dtype=np.int32)
    np.add.at(deltas, start_pos, 1)
    np.add.at(deltas, end_pos, -1)
    mask.iloc[:] = np.cumsum(deltas[:-1]) > 0
    return mask

def build_aligned_15m_panel(pairs: list[str] = ALL_PAIRS, force_refresh: bool = False) -> pd.DataFrame:
    """
    Builds the aligned 15-minute bar panel for all 12 pairs:
    1. Resamples to 15m.
    2. Finds intersection date range across all 12 pairs.
    3. Handles missing bars (forward fills up to 4 consecutive 15m bars, excludes rows exceeding cap).
    Returns MultiIndex DataFrame (Date, Pair) or wide DataFrame with column prefix.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PANEL_CACHE_PATH.exists() and not force_refresh:
        return pd.read_parquet(PANEL_CACHE_PATH)

    dict_15m = {}
    dict_gaps = {}
    starts = []
    ends = []

    for pair in pairs:
        df_1m = download_or_load_pair_ticks(pair)
        dict_gaps[pair] = download_or_load_pair_gaps(pair)
        df_15m = resample_15m_bars(df_1m)
        dict_15m[pair] = df_15m
        
        valid_idx = df_15m.dropna(subset=["close"]).index
        if not valid_idx.empty:
            starts.append(valid_idx.min())
            ends.append(valid_idx.max())

    # Intersection coverage range
    intersection_start = max(starts)
    intersection_end = min(ends)

    # Master 15m timeline across intersection
    master_index = pd.date_range(intersection_start, intersection_end, freq=BAR_RESAMPLE_FREQ, tz="UTC")

    # Combine into wide format per feature
    records = []
    for pair in pairs:
        df_pair = dict_15m[pair].reindex(master_index)
        gap_mask = build_gap_mask(master_index, dict_gaps[pair])
        
        # Forward fill up to 4 consecutive missing bars (1 hr)
        # Identify gap length from the reported gaps schema, plus any missing bars after resampling.
        is_missing = gap_mask | df_pair["close"].isna()
        gap_groups = (~is_missing).cumsum()
        gap_lengths = is_missing.groupby(gap_groups).transform("sum")

        # Fill missing values if gap <= MAX_FORWARD_FILL_BARS
        df_filled = df_pair.copy()
        df_filled.loc[gap_mask, ["open", "high", "low", "close", "volume"]] = np.nan
        mask_fillable = is_missing & (gap_lengths <= MAX_FORWARD_FILL_BARS)
        
        df_filled.loc[mask_fillable, "close"] = df_filled["close"].ffill()
        df_filled.loc[mask_fillable, "open"] = df_filled["close"]
        df_filled.loc[mask_fillable, "high"] = df_filled["close"]
        df_filled.loc[mask_fillable, "low"] = df_filled["close"]
        df_filled.loc[mask_fillable, "volume"] = 0

        df_filled["pair"] = pair
        df_filled["valid"] = ~df_filled["close"].isna()
        records.append(df_filled)

    df_all = pd.concat(records)
    
    # Drop timestamps where ANY pair is still missing (gap > 4 bars)
    valid_counts = df_all.groupby(df_all.index)["valid"].sum()
    complete_timestamps = valid_counts[valid_counts == len(pairs)].index

    df_clean = df_all.loc[df_all.index.isin(complete_timestamps)].drop(columns=["valid"])
    df_clean.to_parquet(PANEL_CACHE_PATH)
    return df_clean

def compute_chronological_splits(panel_df: pd.DataFrame) -> dict:
    """
    Computes non-overlapping Train, Validation, and Evaluation date splits from the panel timeline:
    Evaluation = final 3 calendar years of the intersection range;
    Validation = 3 calendar years immediately preceding Evaluation;
    Train = everything before Validation start.
    """
    timestamps = panel_df.index.unique().sort_values()
    max_ts = timestamps.max()

    # Evaluation = final 3 calendar years
    eval_start_year = max_ts.year - 2
    eval_start = pd.Timestamp(f"{eval_start_year}-01-01", tz="UTC")

    # Validation = 3 calendar years prior to Evaluation
    val_start_year = eval_start_year - 3
    val_start = pd.Timestamp(f"{val_start_year}-01-01", tz="UTC")

    train_ts = timestamps[timestamps < val_start]
    val_ts = timestamps[(timestamps >= val_start) & (timestamps < eval_start)]
    eval_ts = timestamps[timestamps >= eval_start]

    return {
        "train_timestamps": train_ts,
        "val_timestamps": val_ts,
        "eval_timestamps": eval_ts,
        "train_start": str(train_ts.min()) if len(train_ts) > 0 else None,
        "train_end": str(train_ts.max()) if len(train_ts) > 0 else None,
        "val_start": str(val_ts.min()) if len(val_ts) > 0 else None,
        "val_end": str(val_ts.max()) if len(val_ts) > 0 else None,
        "eval_start": str(eval_ts.min()) if len(eval_ts) > 0 else None,
        "eval_end": str(eval_ts.max()) if len(eval_ts) > 0 else None,
    }
