#!/usr/bin/env python3
"""Stock Markets Analytics Zoomcamp 2026 - Módulo 3: Modelado (Homework 3).

Reproduce el pipeline del notebook `[2026]_Module_03_Colab_Time_Series_Modeling.ipynb`
(mismo dataset, mismos sets de features y mismo split temporal) y agrega lo que pide la tarea:
dummies <mes>_w<semana>, dos reglas "a mano" nuevas, un árbol de profundidad 10 y el tuning
de max_depth.

Ejecutar:
    uv run python hw3_questions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 160)

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
PARQUET = DATA / "stocks_df_combined_2026_09_18.parquet.brotli"
TARGET = "is_positive_growth_30d_future"


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 0) Dataset y sets de variables (igual que en el notebook)
# ---------------------------------------------------------------------------
if not PARQUET.exists():
    import gdown
    gdown.download("https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5", str(PARQUET), quiet=True)
df_full = pd.read_parquet(PARQUET)

GROWTH = [g for g in df_full.keys() if g.startswith("growth_") and "future" not in g]
OHLCV = ["Open", "High", "Low", "Close", "Adj Close_x", "Volume"]
CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type"]
TO_PREDICT = [g for g in df_full.keys() if "future" in g]
df_full["ln_volume"] = df_full.Volume.replace(0, np.nan).fillna(1e-9).apply(np.log)
CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative", "volatility", "ln_volume"]
TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop", "cci", "cmo", "dx", "macd", "macdsignal",
    "macdhist", "macd_ext", "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix", "macdhist_fix",
    "mfi", "minus_di", "mom", "plus_di", "dm", "ppo", "roc", "rocp", "rocr", "rocr100", "rsi", "slowk",
    "slowd", "fastk", "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr", "ad", "adosc", "obv",
    "atr", "natr", "ht_dcperiod", "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine",
    "ht_sine_leadsine", "ht_trendmod", "avgprice", "medprice", "typprice", "wclprice"]
TECHNICAL_PATTERNS = [g for g in df_full.keys() if "cdl" in g]
MACRO = ["gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom", "FEDFUNDS", "DGS1", "DGS5", "DGS10"]
NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

df = df_full[df_full.Date >= "2000-01-01"].copy()
print(f"Dataset: {df.shape[0]} filas, {df.Ticker.nunique()} tickers, {df.Date.min():%Y-%m-%d} .. {df.Date.max():%Y-%m-%d}")

# ---------------------------------------------------------------------------
# Q1. Dummies mes + semana del mes
# ---------------------------------------------------------------------------
header("Q1. Month + week-of-month dummies")
df["Weekday"] = df["Weekday"].astype(str)
df["month_wom"] = df["Date"].dt.month_name() + "_w" + ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
df["Month"] = df["Month"].dt.month.astype(str)
CATEGORICAL = CATEGORICAL + ["month_wom"]

dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
DUMMIES = dummy_variables.keys().to_list()
df_with_dummies = pd.concat([df, dummy_variables], axis=1)
wom_dummies = [d for d in DUMMIES if d.startswith("month_wom_")]
print(f"Dummies: {len(DUMMIES)} (month_wom: {len(wom_dummies)})")
print(f"Features NUMERICAL+DUMMIES: {len(NUMERICAL + DUMMIES)}")

corr = df_with_dummies[wom_dummies + [TARGET]].corr()[TARGET].drop(TARGET).to_frame("corr")
corr["abs_corr"] = corr["corr"].abs()
corr = corr.sort_values("abs_corr", ascending=False)
print(corr.head(8).round(4).to_string())
print(f"-> Mayor |corr|: {corr.index[0]} = {corr['abs_corr'].iloc[0]:.3f}")

# ---------------------------------------------------------------------------
# Split temporal 70/15/15 por fecha (igual que en el notebook)
# ---------------------------------------------------------------------------
min_date, max_date = df_with_dummies.Date.min(), df_with_dummies.Date.max()
train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * 0.7)
val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * 0.15)
df_with_dummies["split"] = np.select(
    [df_with_dummies.Date <= train_end, df_with_dummies.Date <= val_end], ["train", "validation"], "test")
new_df = df_with_dummies.copy()
print("\nSplit:", new_df.groupby("split")["Date"].agg(["min", "max", "count"]).to_string(), sep="\n")

# ---------------------------------------------------------------------------
# Q2. Reglas "a mano" nuevas (pred3, pred4)
# ---------------------------------------------------------------------------
header("Q2. New hand rules")
new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
new_df["pred2_manual_prev_g1_and_snp"] = ((new_df.growth_30d > 1) & (new_df.growth_snp500_30d > 1)).astype(int)
new_df["pred3_manual_dgs10_5"] = ((new_df.DGS10 <= 4.5) & (new_df.DGS5 <= 4)).astype(int)
new_df["pred4_manual_dgs10_fedfunds"] = ((new_df.DGS10 > 4) & (new_df.FEDFUNDS <= 4.795)).astype(int)
HAND_RULES = [c for c in new_df.keys() if c.startswith("pred")]

test_mask = new_df.split == "test"


def report_precision(columns):
    for col in columns:
        y_true, y_pred = new_df.loc[test_mask, TARGET], new_df.loc[test_mask, col]
        n_pos = int(y_pred.sum())
        prec = precision_score(y_true, y_pred, zero_division=0) if n_pos else float("nan")
        print(f"  {col:32s} predicciones=1 en TEST: {n_pos:6d}  precision: {prec:.4f}")


report_precision(HAND_RULES)

# ---------------------------------------------------------------------------
# Datos para los árboles: inf -> NaN -> 0
# ---------------------------------------------------------------------------
features = NUMERICAL + DUMMIES


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(0)


train_valid = new_df.split.isin(["train", "validation"])
X_train, y_train = clean(new_df[train_valid]), new_df.loc[train_valid, TARGET]
X_test, y_test = clean(new_df[test_mask]), new_df.loc[test_mask, TARGET]
X_valid, y_valid = clean(new_df[new_df.split == "validation"]), new_df.loc[new_df.split == "validation", TARGET]
X_all = clean(new_df)

# ---------------------------------------------------------------------------
# Q3. Árbol de profundidad 10: aciertos positivos únicos
# ---------------------------------------------------------------------------
header("Q3. Unique correct predictions of pred5_clf_10")
clf_10 = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train)
new_df["pred5_clf_10"] = clf_10.predict(X_all)
report_precision(["pred5_clf_10"])
print(export_text(clf_10, feature_names=list(X_train), max_depth=2))

positives = new_df[TARGET] == 1
new_df["only_pred5_is_correct"] = (
    positives & (new_df["pred5_clf_10"] == 1) & (new_df[HAND_RULES].sum(axis=1) == 0)).astype(int)
unique_test = int(new_df.loc[test_mask, "only_pred5_is_correct"].sum())
print(f"-> Positivos en TEST que solo detecta pred5_clf_10: {unique_test}")

# ---------------------------------------------------------------------------
# Q4. Tuning de max_depth (1..12)
# ---------------------------------------------------------------------------
header("Q4. Decision tree max_depth tuning")
results = []
for depth in range(1, 13):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    results.append({"max_depth": depth,
                    "precision_valid": precision_score(y_valid, clf.predict(X_valid), zero_division=0),
                    "precision_test": precision_score(y_test, clf.predict(X_test), zero_division=0),
                    "n_pos_test": int(clf.predict(X_test).sum())})
results = pd.DataFrame(results).set_index("max_depth")
print(results.round(4).to_string())
best_depth = int(results["precision_test"].idxmax())
print(f"-> Mejor max_depth en TEST: {best_depth} (precision {results.loc[best_depth, 'precision_test']:.4f})")

clf_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_train, y_train)
new_df["pred6_clf_best"] = clf_best.predict(X_all)
print("\nComparación de todas las predicciones en TEST:")
report_precision(HAND_RULES + ["pred5_clf_10", "pred6_clf_best"])
