# FX Assistant

A complete from-scratch Python project that downloads daily FX data, engineers features, trains models, and produces a daily JSON assistant output for USDTRY and EURTRY.

**Return definition**: This project uses **log returns** based on Close prices. Forecasts (`h1`, `h3`, `h7`) are the sum of future daily log returns over the horizon (1, 3, 7 days).

**Risk mapping**: Risk is derived from the recent 7-day rolling volatility of returns (`vol_7`). Thresholds are computed from the training split using 33% and 66% quantiles and saved per symbol.

## Setup
1. Create and activate a virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run
Download data (2005-01-01 to today):

```bash
python main.py download
```

Train models:

```bash
python main.py train
```

Predict for a single symbol:

```bash
python main.py predict --symbol USDTRY
```

Predict for all symbols (prints JSON lines):

```bash
python main.py predict --all
```

## Example Output
```json
{"date":"2026-02-19","symbol":"USDTRY","forecast":{"h1":0.0012,"h3":0.0028,"h7":0.0061},"risk":"MED","insight":"Trend yukar? g?r?n?yor; volatilite orta seviyede. Kesinlik yok; temkinli kal?p kademeli yakla??m d???n?lebilir."}
```

## Notes
- No database is used. All data is stored locally in `data/`.
- Models are saved in `models/` and metrics in `reports/`.
- The project is designed to run on Windows and Linux.
