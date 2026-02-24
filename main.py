from __future__ import annotations

import argparse
import json
import sys
from typing import List

from src.advisor import generate_insight
from src.config import SYMBOLS
from src.download_data import download_all
from src.predict import predict_symbol
from src.train import train_all


def _print_json_line(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    print(line)


def _predict(symbols: List[str]) -> None:
    for symbol in symbols:
        pred = predict_symbol(symbol)
        insight = generate_insight(pred)
        output = {
            "date": pred["date"],
            "symbol": pred["symbol"],
            "forecast": pred["forecast"],
            "risk": pred["risk"],
            "insight": insight,
        }
        _print_json_line(output)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="FX Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download raw FX data")
    sub.add_parser("train", help="Train models")

    predict_parser = sub.add_parser("predict", help="Run predictions")
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument("--symbol", type=str, help="Symbol, e.g., USDTRY")
    predict_group.add_argument("--all", action="store_true", help="Predict all symbols")

    args = parser.parse_args()

    if args.command == "download":
        download_all()
        return

    if args.command == "train":
        train_all()
        return

    if args.command == "predict":
        if args.all:
            _predict(list(SYMBOLS.keys()))
            return
        symbol = args.symbol.upper().strip()
        if symbol not in SYMBOLS:
            raise SystemExit(f"Unknown symbol: {symbol}. Use one of {list(SYMBOLS.keys())}")
        _predict([symbol])
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
