from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SYMBOLS = {
    "USDTRY": "USDTRY=X",
    "EURTRY": "EURTRY=X",
    "GBPTRY": "GBPTRY=X",
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

# Fallback tickers if primary Yahoo symbols fail
SYMBOL_ALIASES = {
    "XAUUSD": ["GC=F", "XAUUSD=X"],
    "XAGUSD": ["SI=F", "XAGUSD=X"],
}

START_DATE = "2005-01-01"

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "log_return",
    *[f"ret_{i}" for i in range(1, 15)],
    "ma_7",
    "ma_21",
    "ema_12",
    "ema_26",
    "rsi_14",
    "vol_7",
    "vol_21",
    "atr_14",
    "day_of_week",
    "month",
]

TARGET_COLUMNS = ["y_h1", "y_h3", "y_h7"]


@dataclass(frozen=True)
class Split:
    train_end: str
    val_end: str


DEFAULT_SPLIT = Split(train_end=TRAIN_END, val_end=VAL_END)
