from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DATA_DIR, FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR, SYMBOLS, RANDOM_SEED
from .features import build_features
from .utils import ensure_dir, save_json, set_seed


def _load_processed(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"processed_{symbol.lower()}.csv"
    if not path.exists():
        raw_path = DATA_DIR / f"raw_{symbol.lower()}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw data for {symbol}. Run download first.")
        build_features(raw_path, symbol)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def _train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=RANDOM_SEED)
    model.fit(X, y)
    return model


def _eval(model: GradientBoostingRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"mae": float(mae), "rmse": float(rmse)}


def train_symbol(symbol: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = _load_processed(symbol)
    train, val, test = _split(df)

    X_train = train[FEATURE_COLUMNS]
    X_val = val[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    ensure_dir(MODELS_DIR)

    for horizon in ["h1", "h3", "h7"]:
        y_col = f"y_{horizon}"
        y_train = train[y_col]
        y_val = val[y_col]
        y_test = test[y_col]

        model = _train_model(X_train, y_train)
        model_path = MODELS_DIR / f"{symbol.lower()}_{horizon}.pkl"
        joblib.dump(model, model_path)

        metrics[horizon] = {
            "val": _eval(model, X_val, y_val) if len(val) > 0 else {},
            "test": _eval(model, X_test, y_test) if len(test) > 0 else {},
        }

    # Risk thresholds from training volatility
    vol = train["vol_7"].dropna()
    q_low = float(vol.quantile(0.33))
    q_high = float(vol.quantile(0.66))
    thresholds = {"low": q_low, "high": q_high}
    save_json(MODELS_DIR / f"{symbol.lower()}_risk_thresholds.json", thresholds)

    return metrics


def train_all() -> None:
    set_seed(RANDOM_SEED)
    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for symbol in SYMBOLS.keys():
        all_metrics[symbol] = train_symbol(symbol)

    ensure_dir(REPORTS_DIR)
    save_json(REPORTS_DIR / "metrics.json", all_metrics)


if __name__ == "__main__":
    train_all()
