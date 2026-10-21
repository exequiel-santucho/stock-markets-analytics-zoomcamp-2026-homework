#!/usr/bin/env python3
"""Stock Markets Analytics Zoomcamp 2026 - Módulo 4: Estrategia de trading y simulación (Homework 4).

Versión en script del notebook `[2026]_Module_04_Colab_Trading_Simulations.ipynb`:
dataset -> dummies -> split temporal -> reglas manuales + árboles + Random Forest ->
reglas por umbral de probabilidad -> simulación "invertir $100 en cada predicción positiva".

Cada pregunta del homework parte de una copia limpia del notebook, así que acá cada una es un
"escenario" que se corre por separado:

    uv run python hw4_pipeline.py original   # notebook tal cual (control: CAGR pred5_clf_10 = 1.1195)
    uv run python hw4_pipeline.py q1         # Q1 + Q2: sin dummies de Ticker y month_wom, umbrales RF 0.56-0.64
    uv run python hw4_pipeline.py q3         # Q3: target = growth_future_30d > 1.022

La red neuronal y la regresión logística del notebook no generan columnas pred*, (usan resultados
precalculados), así que no afectan las simulaciones y no se incluyen.
Los resultados quedan en results/sim1_<escenario>.csv.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.options.mode.chained_assignment = None

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PARQUET = DATA / "stocks_df_combined_2026_09_18.parquet.brotli"
TARGET = "is_positive_growth_30d_future"

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "original"
assert SCENARIO in {"original", "q1", "q3"}, "escenario: original | q1 | q3"


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def load_dataset() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    if not PARQUET.exists():
        import gdown
        gdown.download("https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5", str(PARQUET), quiet=True)
    return pd.read_parquet(PARQUET)


# ---------------------------------------------------------------------------
# 0) Un único DataFrame con dummies
# ---------------------------------------------------------------------------
header(f"Escenario: {SCENARIO}")
df = load_dataset()
GROWTH = [g for g in df.keys() if g.startswith("growth_") and "future" not in g]
TO_PREDICT = [g for g in df.keys() if "future" in g]
df["ln_volume"] = np.log(df.Volume + 1e-6)
df["div_payout"] = (df.Dividends > 0).astype(int)
df["stock_split"] = (df["Stock Splits"] > 0).astype(int)
CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative", "volatility",
                    "ln_volume", "div_payout", "stock_split"]
TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop", "cci", "cmo", "dx", "macd", "macdsignal",
    "macdhist", "macd_ext", "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix", "macdhist_fix",
    "mfi", "minus_di", "mom", "plus_di", "dm", "ppo", "roc", "rocp", "rocr", "rocr100", "rsi", "slowk",
    "slowd", "fastk", "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr", "ad", "adosc", "obv",
    "atr", "natr", "ht_dcperiod", "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine",
    "ht_sine_leadsine", "ht_trendmod", "avgprice", "medprice", "typprice", "wclprice"]
TECHNICAL_PATTERNS = [g for g in df.keys() if "cdl" in g]
MACRO = ["gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom", "FEDFUNDS", "DGS1", "DGS5", "DGS10"]
NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

df = df[df.Date >= "2000-01-01"].reset_index(drop=True)

if SCENARIO == "q3":
    # Q3: positivo solo si el crecimiento a 30 días supera 2.2% (≈ la mediana)
    print("Antes :", df[TARGET].mean().round(3), "de clase 1")
    df[TARGET] = (df["growth_future_30d"] > 1.022).astype(int)
    print("Ahora :", df[TARGET].mean().round(3), "de clase 1")
    print(df["growth_future_30d"].describe().round(3).to_string())

df["Month"] = df["Month"].dt.strftime("%B")
df["Weekday"] = df["Weekday"].astype("string")
df["month_wom"] = df["Month"] + "_w" + ((df.Date.dt.day - 1) // 7 + 1).astype(str)

CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
if SCENARIO == "q1":
    # Q1: menos dummies (sin una por ticker ni las 60 de mes+semana)
    CATEGORICAL = [c for c in CATEGORICAL if c not in ("Ticker", "month_wom")]
dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
DUMMIES = dummy_variables.keys().to_list()
new_df = pd.concat([df, dummy_variables], axis=1)
del df, dummy_variables
gc.collect()
FEATURES = NUMERICAL + DUMMIES
print(f"CATEGORICAL = {CATEGORICAL}")
print(f"Features NUMERICAL+DUMMIES: {len(FEATURES)}  (dummies: {len(DUMMIES)})")

# split temporal 70/15/15 por fecha
min_date, max_date = new_df.Date.min(), new_df.Date.max()
train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * 0.7)
val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * 0.15)
new_df["split"] = np.select([new_df.Date <= train_end, new_df.Date <= val_end], ["train", "validation"], "test")
print(new_df.groupby("split")["Date"].agg(["min", "max", "count"]).to_string())

# ---------------------------------------------------------------------------
# 1) Modelos
# ---------------------------------------------------------------------------
new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
new_df["pred2_manual_prev_g1_and_snp"] = ((new_df.growth_30d > 1) & (new_df.growth_snp500_30d > 1)).astype(int)
new_df["pred3_manual_dgs10_5"] = ((new_df.DGS10 <= 4.5) & (new_df.DGS5 <= 4)).astype(int)
new_df["pred4_manual_dgs10_fedfunds"] = ((new_df.DGS10 > 4) & (new_df.FEDFUNDS <= 4.795)).astype(int)

# float32: los árboles de sklearn convierten a float32 internamente (mismo resultado, la mitad de RAM)
X_all = new_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_all = new_df[TARGET]
is_test = (new_df.split == "test").to_numpy()
is_tv = new_df.split.isin(["train", "validation"]).to_numpy()
X_train_valid, y_train_valid = X_all[is_tv], y_all[is_tv]
X_test, y_test = X_all[is_test], y_all[is_test]


def fit_tree(depth: int) -> DecisionTreeClassifier:
    t0 = time.time()
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train_valid, y_train_valid)
    print(f"  DecisionTree(max_depth={depth}) entrenado en {time.time() - t0:.0f}s", flush=True)
    return clf


header("Modelos")
new_df["pred5_clf_10"] = fit_tree(10).predict(X_all)
new_df["pred6_clf_4"] = fit_tree(4).predict(X_all)
clf_best = fit_tree(7)  # "segundo mejor" del tuning precalculado del notebook
new_df["pred7_clf_second_best"] = clf_best.predict(X_all)

proba_clf = clf_best.predict_proba(X_all)[:, 1]
new_df["pred8_clf_second_best_rule_56"] = (proba_clf >= 0.56).astype(int)
new_df["pred9_clf_second_best_rule_66"] = (proba_clf >= 0.66).astype(int)

t0 = time.time()
rf_best = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42, n_jobs=-1)
rf_best.fit(X_train_valid, y_train_valid)
print(f"  RandomForest(100, max_depth=13) entrenado en {time.time() - t0:.0f}s", flush=True)
proba_rf = rf_best.predict_proba(X_all)[:, 1]
new_df["pred10_rf_best_rule_55"] = (proba_rf >= 0.55).astype(int)
new_df["pred11_rf_best_rule_65"] = (proba_rf >= 0.65).astype(int)

if SCENARIO == "q1":
    # Q2: reglas extra del RF para cubrir todo el rango 0.55..0.65
    for i, thr in enumerate(range(56, 65), start=12):
        new_df[f"pred{i}_rf_best_rule_{thr}"] = (proba_rf >= thr / 100).astype(int)

    # tabla precision/recall vs umbral del RF en TEST (lo que pide construir la Q2)
    rows = []
    for thr in np.round(np.arange(0.50, 0.71, 0.01), 2):
        pred = proba_rf[is_test] >= thr
        tp = int((pred & (y_test == 1)).sum())
        fp = int((pred & (y_test == 0)).sum())
        fn = int((~pred & (y_test == 1)).sum())
        rows.append({"threshold": thr, "tp": tp, "fp": fp, "trades(tp+fp)": tp + fp,
                     "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn)})
    print("\nRandom Forest en TEST: precision/recall por umbral")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

PREDICTIONS = [k for k in new_df.keys() if k.startswith("pred")]
print("\nPrecision en TEST:")
for p in PREDICTIONS:
    n = int(new_df.loc[is_test, p].sum())
    prec = precision_score(y_test, new_df.loc[is_test, p], zero_division=0) if n else float("nan")
    print(f"  {p:32s} positivos={n:6d}  precision={prec:.4f}")

# ---------------------------------------------------------------------------
# 3) Simulación 1: $100 en cada predicción positiva, fees 0.2% (compra+venta)
# ---------------------------------------------------------------------------
header("Simulación 1")
test_df = new_df[is_test]
sim1_results = []
for pred in PREDICTIONS:
    positive = test_df[test_df[pred] == 1]
    gross = (100 * (positive["growth_future_30d"] - 1)).sum()
    fees = -100 * 0.002 * len(positive)
    net = gross + fees
    daily = positive.groupby("Date")[pred].count()
    q75 = daily.quantile(0.75) if len(daily) else np.nan
    capital = 100 * 30 * q75  # 30 días seguidos con predicciones positivas
    cagr = ((capital + net) / capital) ** (1 / 4) if len(daily) else np.nan
    sim1_results.append({
        "prediction": pred, "sim1_count_investments": len(positive), "sim1_gross_rev": gross,
        "sim1_fees": fees, "sim1_net_rev": net,
        "sim1_fees_percentage": -fees / gross if gross > 0 else None,
        "sim1_average_net_revenue": net / len(positive) if len(positive) else None,
        "sim1_avg_investments_per_day": daily.mean() if len(daily) else np.nan,
        "sim1_capital": capital, "sim1_CAGR": cagr})
df_sim1_results = pd.DataFrame(sim1_results)
df_sim1_results["sim1_growth_capital_4y"] = (df_sim1_results.sim1_net_rev + df_sim1_results.sim1_capital) \
    / df_sim1_results.sim1_capital

cols = ["prediction", "sim1_count_investments", "sim1_net_rev", "sim1_avg_investments_per_day",
        "sim1_capital", "sim1_CAGR"]
print(df_sim1_results[cols].round(4).to_string(index=False))
RESULTS.mkdir(exist_ok=True)
df_sim1_results.to_csv(RESULTS / f"sim1_{SCENARIO}.csv", index=False)

best = df_sim1_results.loc[df_sim1_results.sim1_CAGR.idxmax()]
print(f"\nMejor CAGR: {best.prediction} = {best.sim1_CAGR:.4f} "
      f"({best.sim1_avg_investments_per_day:.2f} trades/día en promedio)")
