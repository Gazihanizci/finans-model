from __future__ import annotations

from typing import Dict


def _trend_text(trend: str) -> str:
    if trend == "UP":
        return "yukarı"
    if trend == "DOWN":
        return "aşağı"
    return "yatay"


def _risk_text(risk: str) -> str:
    if risk == "HIGH":
        return "yüksek volatilite"
    if risk == "MED":
        return "orta volatilite"
    return "düşük volatilite"


def generate_insight(pred: Dict[str, object]) -> str:
    trend = str(pred.get("_trend", "FLAT"))
    risk = str(pred.get("risk", "MED"))

    trend_text = _trend_text(trend)
    risk_text = _risk_text(risk)

    if risk == "HIGH":
        suggestion = "Kesinlik yok; temkinli kalıp küçük kademe ile ilerlemek ve ani hareket riskini azaltmak uygun olabilir."
    elif trend == "UP":
        suggestion = "Kesinlik yok; temkinli olmak kaydıyla kademeli alım düşünülebilir."
    elif trend == "DOWN":
        suggestion = "Kesinlik yok; korumacı kalmak ve ağırlık artırmamak daha uygun olabilir."
    else:
        suggestion = "Kesinlik yok; bekle-gör yaklaşımı ve risk azaltma tercih edilebilir."

    return f"Trend {trend_text} görünüyor; {risk_text} var. {suggestion}"


if __name__ == "__main__":
    raise SystemExit("Run via main entrypoint.")
