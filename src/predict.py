from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, MODELS_DIR
from .features import build_features
from .utils import load_json


def _load_processed(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"processed_{symbol.lower()}.csv"
    if not path.exists():
        raw_path = DATA_DIR / f"raw_{symbol.lower()}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw data for {symbol}. Run download first.")
        build_features(raw_path, symbol)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def _load_model(symbol: str, horizon: str):
    path = MODELS_DIR / f"{symbol.lower()}_{horizon}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing model {path}. Run train first.")
    return joblib.load(path)


def _risk_label(vol_7: float, thresholds: Dict[str, float]) -> str:
    if vol_7 >= thresholds["high"]:
        return "HIGH"
    if vol_7 >= thresholds["low"]:
        return "MED"
    return "LOW"


def _trend(row: pd.Series) -> str:
    ema_up = row["ema_12"] > row["ema_26"]
    ma_up = row["ma_7"] > row["ma_21"]
    ret_up = row["log_return"] > 0

    ema_down = row["ema_12"] < row["ema_26"]
    ma_down = row["ma_7"] < row["ma_21"]
    ret_down = row["log_return"] < 0

    if ema_up and ma_up and ret_up:
        return "UP"
    if ema_down and ma_down and ret_down:
        return "DOWN"
    return "FLAT"


def predict_symbol(symbol: str) -> Dict[str, object]:
    df = _load_processed(symbol)
    if df.empty:
        raise RuntimeError(f"No processed data for {symbol}.")

    latest = df.iloc[-1]
    X_latest = latest[FEATURE_COLUMNS].to_frame().T

    model_h1 = _load_model(symbol, "h1")
    model_h3 = _load_model(symbol, "h3")
    model_h7 = _load_model(symbol, "h7")

    pred_h1 = float(model_h1.predict(X_latest)[0])
    pred_h3 = float(model_h3.predict(X_latest)[0])
    pred_h7 = float(model_h7.predict(X_latest)[0])

    thresholds = load_json(MODELS_DIR / f"{symbol.lower()}_risk_thresholds.json")
    risk = _risk_label(float(latest["vol_7"]), thresholds)
    trend = _trend(latest)

    return {
        "date": latest.name.strftime("%Y-%m-%d"),
        "symbol": symbol,
        "forecast": {"h1": pred_h1, "h3": pred_h3, "h7": pred_h7},
        "risk": risk,
        "_trend": trend,
        "_vol_7": float(latest["vol_7"]),
    }


if __name__ == "__main__":
    raise SystemExit("Run via main entrypoint.")
