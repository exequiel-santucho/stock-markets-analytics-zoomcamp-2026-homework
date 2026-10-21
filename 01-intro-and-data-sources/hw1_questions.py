#!/usr/bin/env python3
"""Stock Markets Analytics Zoomcamp 2026 - Módulo 1: Intro y fuentes de datos (Homework 1).

Ejecutar:
    uv run python hw1_questions.py

Los datos se bajan de Wikipedia y Yahoo Finance (yfinance) y se cachean en data/.
Como son fuentes "vivas", los resultados pueden moverse un poco con el tiempo; por eso las
fechas de corte quedan fijas (las del enunciado).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")  # acentos OK en la consola de Windows
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def download_close(ticker: str, cache: str, **kwargs) -> pd.Series:
    """Baja precios diarios con yfinance (una sola vez) y devuelve la serie Close."""
    path = DATA / cache
    if not path.exists():
        df = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
        df.columns = df.columns.get_level_values(0)  # yfinance devuelve MultiIndex aun con 1 ticker
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["Close"].dropna()


# ---------------------------------------------------------------------------
# Q1. S&P 500: año con más incorporaciones (desde 2020)
# ---------------------------------------------------------------------------
header("Q1. S&P 500 additions per year")
sp500_path = DATA / "sp500_companies.csv"
if not sp500_path.exists():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=HEADERS, timeout=30).text
    pd.read_html(io.StringIO(html))[0].to_csv(sp500_path, index=False)
sp500 = pd.read_csv(sp500_path)
sp500 = sp500[["Symbol", "Security", "Date added"]].copy()
sp500["year_added"] = pd.to_datetime(sp500["Date added"], errors="coerce").dt.year
per_year = sp500["year_added"].value_counts().sort_index()
print("Empresas en la lista:", len(sp500))
print(per_year.loc[2015:].to_string())
full_years = per_year.loc[2020:2025]  # 2026 todavía no es un año completo
print(f"-> Año con más incorporaciones (2020-2025): {full_years.idxmax()} ({full_years.max()} empresas)")

today = pd.Timestamp("2026-08-21")
years_in = (today - pd.to_datetime(sp500["Date added"], errors="coerce")).dt.days / 365.25
print(f"Adicional: empresas con más de 20 años en el índice: {(years_in > 20).sum()}")

# ---------------------------------------------------------------------------
# Q2. Índices mundiales: YTD al 21/08/2026 vs S&P 500
# ---------------------------------------------------------------------------
header("Q2. World indexes YTD (2026-01-01 .. 2026-08-21)")
INDEXES = {
    "United States - S&P 500": "^GSPC",
    "China - Shanghai Composite": "000001.SS",
    "Hong Kong - Hang Seng": "^HSI",
    "Australia - S&P/ASX 200": "^AXJO",
    "India - Nifty 50": "^NSEI",
    "Canada - S&P/TSX": "^GSPTSE",
    "Germany - DAX": "^GDAXI",
    "United Kingdom - FTSE 100": "^FTSE",
    "Japan - Nikkei 225": "^N225",
    "Mexico - IPC": "^MXX",
    "Brazil - Ibovespa": "^BVSP",
}
ytd = {}
for name, ticker in INDEXES.items():
    # end es exclusivo en yfinance: igual que en el enunciado, end='2026-08-21'
    close = download_close(ticker, f"ytd_{ticker.strip('^')}.csv", start="2026-01-01", end="2026-08-21")
    ytd[name] = close.iloc[-1] / close.iloc[0] - 1
ytd = pd.Series(ytd).sort_values(ascending=False)
print((ytd * 100).round(2).to_string())
us = ytd["United States - S&P 500"]
better = (ytd.drop("United States - S&P 500") > us).sum()
print(f"-> Índices con mejor YTD que el S&P 500 ({us:.2%}): {better} de 10")

# Adicional: 3, 5 y 10 años (hasta el 21/08/2026)
long_term = {}
for name, ticker in INDEXES.items():
    close = download_close(ticker, f"long_{ticker.strip('^')}.csv", start="2016-08-01", end="2026-08-22")
    row = {}
    for years in (3, 5, 10):
        start = close.index[-1] - pd.DateOffset(years=years)
        row[f"{years}y"] = close.iloc[-1] / close.loc[start:].iloc[0] - 1
    long_term[name] = row
long_term = pd.DataFrame(long_term).T
print("\nAdicional: retorno acumulado a 3/5/10 años (%):")
print((long_term * 100).round(1).to_string())
print("Índices que le ganan al S&P 500:",
      {c: int((long_term[c].drop("United States - S&P 500") > long_term.loc["United States - S&P 500", c]).sum())
       for c in long_term.columns})

# ---------------------------------------------------------------------------
# Q3. Correcciones del S&P 500 (caídas >= 5% desde el máximo histórico)
# ---------------------------------------------------------------------------
header("Q3. S&P 500 corrections (drawdown >= 5%)")
spx = download_close("^GSPC", "gspc_1950.csv", start="1950-01-01", end="2026-09-15")
is_ath = spx >= spx.cummax()  # nuevo máximo histórico (o igual al anterior)
ath_dates = spx.index[is_ath]

rows = []
for start, end in zip(ath_dates[:-1], ath_dates[1:]):
    between = spx.loc[start:end]
    if len(between) <= 2:  # máximos consecutivos sin nada en el medio
        continue
    high = spx.loc[start]
    low_date = between.idxmin()
    low = between.min()
    rows.append({"ath_date": start, "low_date": low_date, "high": high, "low": low,
                 "drawdown_pct": (high - low) / high * 100,
                 "duration_days": (low_date - start).days})
corr = pd.DataFrame(rows)
corr = corr[corr["drawdown_pct"] >= 5].sort_values("drawdown_pct", ascending=False)
print("Correcciones encontradas:", len(corr))
print("Top 10:")
for r in corr.head(10).itertuples():
    print(f"  {r.ath_date:%Y-%m-%d} a {r.low_date:%Y-%m-%d}: {r.drawdown_pct:.1f}% en {r.duration_days} días")
print("\nPercentiles (25/50/75):")
print(corr[["drawdown_pct", "duration_days"]].quantile([0.25, 0.5, 0.75]).round(1).to_string())
print(f"-> Mediana del drawdown: {corr['drawdown_pct'].median():.1f}%")

# ---------------------------------------------------------------------------
# Q4. AMZN: retorno a 2 días después de sorpresas positivas de ganancias
# ---------------------------------------------------------------------------
header("Q4. AMZN earnings surprise")
earn_path = DATA / "amzn_earnings.csv"
if not earn_path.exists():
    yf.Ticker("AMZN").get_earnings_dates(limit=40).to_csv(earn_path)
earn = pd.read_csv(earn_path)
earn["date"] = pd.to_datetime(earn["Earnings Date"].str[:10])
# mismo rango que el enunciado: 25 fechas desde 2020-10-29 (la última, futura, sin EPS reportado)
earn = earn[(earn["date"] >= "2020-10-29") & (earn["date"] <= "2026-10-31")].sort_values("date")
print(f"Fechas de resultados: {len(earn)} ({earn['date'].min():%Y-%m-%d} .. {earn['date'].max():%Y-%m-%d})")

amzn = download_close("AMZN", "amzn_max.csv", period="max").to_frame("Close")
amzn = amzn.loc[:"2026-09-14"]
# ventana de 3 días: Día 1, Día 2 (=fecha del anuncio), Día 3  -> Close_Día3 / Close_Día1 - 1
amzn["ret_2d"] = amzn["Close"].shift(-1) / amzn["Close"].shift(1) - 1
print(f"Retorno 2 días, todos los días: mediana {amzn['ret_2d'].median():.2%}")

merged = earn.merge(amzn, left_on="date", right_index=True, how="inner")
merged["surprise"] = pd.to_numeric(merged["Surprise(%)"], errors="coerce")
positive = merged[merged["surprise"] > 0]
print(f"Sorpresas positivas: {len(positive)}")
print(positive[["date", "EPS Estimate", "Reported EPS", "surprise", "ret_2d"]].to_string(index=False))
print(f"-> Mediana del retorno 2 días tras sorpresa positiva: {positive['ret_2d'].median():.2%}")
print("Correlación sorpresa vs retorno:")
print(merged[["surprise", "ret_2d"]].corr().round(3).to_string())

# Adicional: mercado alcista vs bajista (precio por encima/debajo de su media de 200 días)
amzn["sma200"] = amzn["Close"].rolling(200).mean()
merged = merged.merge(amzn[["sma200"]], left_on="date", right_index=True)
merged["regime"] = np.where(merged["Close"] > merged["sma200"], "bull", "bear")
print("\nAdicional: retorno 2 días por régimen (sorpresas positivas):")
print(merged[merged["surprise"] > 0].groupby("regime")["ret_2d"].agg(["count", "median", "mean"]).round(4))
