#!/usr/bin/env python3
"""Stock Markets Analytics Zoomcamp 2026 - Módulo 2: Trabajar con los datos en Pandas (Homework 2).

Ejecutar:
    uv run python hw2_questions.py

Fuentes: iposcoop.com (tablas HTML), Yahoo Finance (yfinance) y el parquet del curso en Google Drive.
Todo se cachea en data/ la primera vez. Las tablas de iposcoop cambian con el tiempo, así que los
conteos pueden diferir en una o dos filas de los del enunciado (lo aclara el propio homework).
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def read_html_table(url: str, cache: str) -> pd.DataFrame:
    """read_html sobre io.StringIO (evita el OSError de pandas+lxml en Windows)."""
    path = DATA / cache
    if not path.exists():
        path.write_text(requests.get(url, headers=HEADERS, timeout=60).text, encoding="utf-8")
    return pd.read_html(io.StringIO(path.read_text(encoding="utf-8")))[0]


def to_number(value) -> float:
    """'$8.00' -> 8.0, '1,234' -> 1234.0, '-' / vacío -> NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Q1. IPOs retiradas por tipo de empresa
# ---------------------------------------------------------------------------
header("Q1. Withdrawn IPOs by company type")
filed = read_html_table("https://www.iposcoop.com/ipos-recently-filed/", "ipos_recently_filed.html")
filed["File Date"] = pd.to_datetime(filed["File Date"])
withdrawn = filed[(filed["Expected To Trade"] == "Withdrawn") & (filed["File Date"] < "2026-09-11")].copy()
print("IPOs retiradas antes del 11/09/2026:", len(withdrawn))

# el orden importa: gana la primera regla que matchea (palabras exactas)
RULES = [
    ("Technologies", ["Technologies"]),
    ("Acquisition Corp", ["Acquisition Corp", "Acquisition Corporation", "Corp"]),
    ("Inc.", ["Inc", "Incorporated"]),
    ("Group", ["Group"]),
    ("Limited", ["Ltd", "Limited"]),
    ("Holdings", ["Holdings", "Holding"]),
]


def company_type(name: str) -> str:
    for label, patterns in RULES:
        if any(re.search(rf"\b{re.escape(p)}\b", name) for p in patterns):
            return label
    return "Other"


withdrawn["Company Type"] = withdrawn["Company"].apply(company_type)
withdrawn["Avg_price"] = (withdrawn["Price Low"].map(to_number) + withdrawn["Price High"].map(to_number)) / 2
withdrawn["Shares (millions)"] = withdrawn["Shares (millions)"].map(to_number)
withdrawn["Est $ Vol (millions)"] = withdrawn["Est $ Vol (millions)"].map(to_number)
withdrawn["Shares_offered_value"] = (withdrawn["Shares (millions)"] * withdrawn["Avg_price"]).fillna(
    withdrawn["Est $ Vol (millions)"])

print(withdrawn[["File Date", "Company", "Company Type", "Avg_price", "Shares (millions)",
                 "Shares_offered_value"]].to_string(index=False))
by_type = withdrawn.groupby("Company Type")["Shares_offered_value"].agg(["count", "sum"]).sort_values(
    "sum", ascending=False)
print("\n", by_type.round(1).to_string())
print(f"-> Clase con más valor retirado: {by_type.index[0]} ({by_type['sum'].iloc[0]:.1f} M$)")

# ---------------------------------------------------------------------------
# Q2. Sharpe ratio mediano de las IPOs 2025 (antes del 1/09/2025) al 11/09/2026
# ---------------------------------------------------------------------------
header("Q2. Median Sharpe ratio of 2025 IPOs")
ipos = read_html_table("https://www.iposcoop.com/2025-pricings/", "ipos_2025.html")
ipos["Offer Date"] = pd.to_datetime(ipos["Offer Date"], format="%m/%d/%Y")
print("IPOs 2025:", len(ipos))
ipos = ipos[(ipos["Offer Date"] < "2025-09-01") & (ipos["Return"] != "0.00%")]
tickers = sorted(ipos["Symbol"].unique())
print("IPOs antes del 1/09/2025 y con retorno != 0%:", len(tickers))

stocks_path = DATA / "stocks_df_ipos_2025.parquet"
if not stocks_path.exists():
    frames = []
    for t in tickers:
        df = yf.download(t, start="2025-01-01", end="2026-09-12", auto_adjust=True, progress=False)
        if df.empty:  # probablemente deslistada: yfinance no devuelve nada
            continue
        df.columns = df.columns.get_level_values(0)
        df["Ticker"] = t
        frames.append(df.reset_index())
    pd.concat(frames, ignore_index=True).to_parquet(stocks_path)
stocks_df = pd.read_parquet(stocks_path).sort_values(["Ticker", "Date"])
print("Tickers con datos en Yahoo Finance:", stocks_df["Ticker"].nunique())

g = stocks_df.groupby("Ticker")["Close"]
stocks_df["growth_252d"] = stocks_df["Close"] / g.shift(252)
stocks_df["volatility"] = g.transform(lambda s: s.rolling(30).std()) * np.sqrt(252)
stocks_df["Sharpe"] = (stocks_df["growth_252d"] - 0.05) / stocks_df["volatility"]

snapshot = stocks_df[stocks_df["Date"] == "2026-09-11"]
print(snapshot[["growth_252d", "volatility", "Sharpe"]].describe().round(3).to_string())
print(f"-> Sharpe mediano al 11/09/2026: {snapshot['Sharpe'].median():.3f}")
print(f"growth_252d: mediana {snapshot['growth_252d'].median():.3f} vs media {snapshot['growth_252d'].mean():.3f}")

# ---------------------------------------------------------------------------
# Q3. Estrategia de mantener N meses (1 a 12) desde el primer día
# ---------------------------------------------------------------------------
header("Q3. Fixed months holding strategy")
for m in range(1, 13):
    stocks_df[f"future_growth_{m}_m"] = g.shift(-21 * m) / stocks_df["Close"]
min_dates = stocks_df.groupby("Ticker")["Date"].min().reset_index().rename(columns={"Date": "min_date"})
first_day = stocks_df.merge(min_dates, left_on=["Ticker", "Date"], right_on=["Ticker", "min_date"], how="inner")
cols = [f"future_growth_{m}_m" for m in range(1, 13)]
stats = first_day[cols].describe().T
print(stats[["count", "mean", "25%", "50%", "75%"]].round(3).to_string())
best = stats["50%"].idxmax()
print(f"-> Mejor horizonte: {best} con mediana {stats.loc[best, '50%']:.3f}")

# ---------------------------------------------------------------------------
# Q4. Estrategia RSI < 30 (25 años)
# ---------------------------------------------------------------------------
header("Q4. RSI < 30 strategy")
parquet = DATA / "data.parquet"
if not parquet.exists():
    import gdown
    gdown.download("https://drive.google.com/uc?id=1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-", str(parquet), quiet=True)
df = pd.read_parquet(parquet, engine="pyarrow")
print("Shape del dataset:", df.shape)
selected_df = df[(df["rsi"] < 30) & (df["Date"] >= "2000-01-01") & (df["Date"] <= "2025-06-01")]
net_income = 1000 * (selected_df["growth_future_30d"] - 1).sum()
print(f"Trades: {len(selected_df)}")
print(f"Retorno medio a 30 días: {(selected_df['growth_future_30d'].mean() - 1):.2%}")
print(f"Win rate: {(selected_df['growth_future_30d'] > 1).mean():.2%}")
print(f"-> Ganancia neta: ${net_income:,.0f} (~{net_income / 1000:.0f} mil)")
