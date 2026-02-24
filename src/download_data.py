from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

from .config import DATA_DIR, START_DATE, SYMBOLS, SYMBOL_ALIASES
from .utils import ensure_dir


def _download_yf(yf_symbol: str) -> pd.DataFrame:
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(
        yf_symbol,
        start=START_DATE,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return df


def download_symbol(symbol: str, yf_symbol: str) -> Path:
    ensure_dir(DATA_DIR)
    candidates = [yf_symbol]
    candidates = list(dict.fromkeys(SYMBOL_ALIASES.get(symbol, []) + candidates))

    df = pd.DataFrame()
    used_symbol = None
    for cand in candidates:
        df = _download_yf(cand)
        if not df.empty:
            used_symbol = cand
            break
    if df.empty:
        tried = ", ".join(candidates)
        raise RuntimeError(f"No data downloaded for {symbol}. Tried: {tried}")

    df = df.reset_index()
    def _clean_col(c: object) -> str:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if p not in (None, "")]
            name = "_".join(parts)
        else:
            name = str(c)
        return name.replace(" ", "_")

    df.columns = [_clean_col(c) for c in df.columns]

    out_path = DATA_DIR / f"raw_{symbol.lower()}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def download_all() -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for symbol, yf_symbol in SYMBOLS.items():
        paths[symbol] = download_symbol(symbol, yf_symbol)
    return paths


if __name__ == "__main__":
    download_all()
