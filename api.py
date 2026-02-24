from __future__ import annotations

from typing import Optional
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException

from src.advisor import generate_insight
from src.config import SYMBOLS, REPORTS_DIR
from src.predict import predict_symbol
from src.download_data import download_all
from src.train import train_all
from src.utils import load_json, save_json, ensure_dir

app = FastAPI(title="FX Assistant API", version="1.0")

UPDATE_INTERVAL_HOURS = 6
LAST_UPDATE_PATH = REPORTS_DIR / "last_update.json"
_update_lock = Lock()


def _get_last_update() -> Optional[datetime]:
    if not LAST_UPDATE_PATH.exists():
        return None
    data = load_json(LAST_UPDATE_PATH)
    ts = data.get("last_update")
    if not ts:
        return None
    return datetime.fromisoformat(ts)


def _set_last_update(dt: datetime) -> None:
    ensure_dir(LAST_UPDATE_PATH.parent)
    save_json(LAST_UPDATE_PATH, {"last_update": dt.isoformat()})


def _maybe_update() -> None:
    now = datetime.now(timezone.utc)
    last = _get_last_update()
    if last is not None and now - last < timedelta(hours=UPDATE_INTERVAL_HOURS):
        return
    with _update_lock:
        # Re-check after acquiring lock
        last = _get_last_update()
        if last is not None and now - last < timedelta(hours=UPDATE_INTERVAL_HOURS):
            return
        download_all()
        train_all()
        _set_last_update(now)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/symbols")
def symbols() -> dict:
    return {"symbols": list(SYMBOLS.keys())}


@app.get("/predict")
def predict(symbol: Optional[str] = None) -> list[dict] | dict:
    _maybe_update()
    if symbol:
        sym = symbol.upper().strip()
        if sym not in SYMBOLS:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {sym}")
        pred = predict_symbol(sym)
        insight = generate_insight(pred)
        return {
            "date": pred["date"],
            "symbol": pred["symbol"],
            "forecast": pred["forecast"],
            "risk": pred["risk"],
            "insight": insight,
        }

    outputs = []
    for sym in SYMBOLS.keys():
        pred = predict_symbol(sym)
        insight = generate_insight(pred)
        outputs.append(
            {
                "date": pred["date"],
                "symbol": pred["symbol"],
                "forecast": pred["forecast"],
                "risk": pred["risk"],
                "insight": insight,
            }
        )
    return outputs
