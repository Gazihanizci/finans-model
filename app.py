import json
import sys
from datetime import datetime

import streamlit as st

from src.advisor import generate_insight
from src.config import SYMBOLS
from src.download_data import download_all
from src.predict import predict_symbol
from src.train import train_all

st.set_page_config(page_title="FX Assistant", layout="wide")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

st.title("FX Assistant ? USDTRY & EURTRY")
st.caption("Yerel veriler, yerel modeller. ??kt?lar bilgilendirme ama?l?d?r.")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("??lem")
    if st.button("Veri indir"):
        with st.spinner("Veriler indiriliyor..."):
            download_all()
        st.success("Veri indirme tamamland?.")

    if st.button("Model e?it"):
        with st.spinner("Modeller e?itiliyor..."):
            train_all()
        st.success("E?itim tamamland?.")

    st.subheader("Tahmin")
    mode = st.radio("Kapsam", ["T?m?", "Tek"], horizontal=True)
    selected = None
    if mode == "Tek":
        selected = st.selectbox("Sembol", list(SYMBOLS.keys()))

    if st.button("Tahmin ?ret"):
        with st.spinner("Tahminler hesaplan?yor..."):
            symbols = list(SYMBOLS.keys()) if mode == "T?m?" else [selected]
            results = []
            for sym in symbols:
                pred = predict_symbol(sym)
                insight = generate_insight(pred)
                out = {
                    "date": pred["date"],
                    "symbol": pred["symbol"],
                    "forecast": pred["forecast"],
                    "risk": pred["risk"],
                    "insight": insight,
                }
                results.append(out)

        st.session_state["results"] = results

with col_right:
    st.subheader("??kt?lar")
    results = st.session_state.get("results", [])
    if not results:
        st.info("Hen?z ??kt? yok. Soldan 'Tahmin ?ret' ile olu?turabilirsiniz.")
    else:
        for item in results:
            st.markdown(f"**{item['symbol']}** ? {item['date']}")
            st.json(item)
            st.markdown("---")

st.caption("Not: Bu ??kt? yat?r?m tavsiyesi de?ildir.")
