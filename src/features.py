from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_SPLIT, FEATURE_COLUMNS


@dataclass
class FeatureResult:
    df: pd.DataFrame
    output_path: Path


def _rsi_14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14).mean()


def _resolve_column(df: pd.DataFrame, base: str) -> str:
    if base in df.columns:
        return base
    candidates = [c for c in df.columns if str(c).startswith(f"{base}_")]
    if base == "Close":
        candidates = [c for c in candidates if not str(c).startswith("Adj_Close")]
    if not candidates:
        raise KeyError(f"Missing required column for base '{base}'. Available: {list(df.columns)}")
    return sorted(candidates)[0]


def build_features(raw_path: Path, symbol: str) -> FeatureResult:
    df = pd.read_csv(raw_path)
    if "Date" not in df.columns:
        raise ValueError("Raw data must contain a Date column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    close_col = _resolve_column(df, "Close")
    high_col = _resolve_column(df, "High")
    low_col = _resolve_column(df, "Low")

    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    log_close = np.log(close)
    df["log_return"] = log_close.diff()

    for i in range(1, 15):
        df[f"ret_{i}"] = df["log_return"].shift(i)

    df["ma_7"] = close.rolling(7).mean()
    df["ma_21"] = close.rolling(21).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["rsi_14"] = _rsi_14(close)

    df["vol_7"] = df["log_return"].rolling(7).std()
    df["vol_21"] = df["log_return"].rolling(21).std()
    df["atr_14"] = _atr_14(high, low, close)

    df["day_of_week"] = df.index.dayofweek.astype(int)
    df["month"] = df.index.month.astype(int)

    lr = df["log_return"]
    df["y_h1"] = lr.shift(-1)
    df["y_h3"] = lr.shift(-1) + lr.shift(-2) + lr.shift(-3)
    df["y_h7"] = (
        lr.shift(-1)
        + lr.shift(-2)
        + lr.shift(-3)
        + lr.shift(-4)
        + lr.shift(-5)
        + lr.shift(-6)
        + lr.shift(-7)
    )

    df = df.dropna().copy()

    train_end = pd.to_datetime(DEFAULT_SPLIT.train_end)
    val_end = pd.to_datetime(DEFAULT_SPLIT.val_end)

    def _label_split(dt: pd.Timestamp) -> str:
        if dt <= train_end:
            return "train"
        if dt <= val_end:
            return "val"
        return "test"

    df["split"] = [
        _label_split(dt) for dt in df.index
    ]

    # keep only needed columns
    keep_cols = FEATURE_COLUMNS + ["y_h1", "y_h3", "y_h7", "split"]
    df = df[keep_cols]

    out_path = DATA_DIR / f"processed_{symbol.lower()}.csv"
    df.to_csv(out_path, index=True)

    return FeatureResult(df=df, output_path=out_path)


if __name__ == "__main__":
    raise SystemExit("Run via train or main entrypoint.")
