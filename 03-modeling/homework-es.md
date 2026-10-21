## Homework 3: Modelado (cohorte 2026)

> Enunciado original: `stock-markets-analytics-zoomcamp/cohorts/2026/homework3.md`
> Entrega: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03 (vencía el 12/10/2026)
> Código: `hw3_questions.py` (reproduce el notebook `[2026]_Module_03_Colab_Time_Series_Modeling.ipynb`)

Trabajamos con variables categóricas, entrenamos los primeros modelos (árboles de decisión) y ajustamos hiperparámetros. Dataset: 33 de las acciones más grandes de EE.UU., Europa e India, 2000-2026 (202.292 filas), con target `is_positive_growth_30d_future` (¿sube en los próximos 30 días hábiles?).

Split temporal (por fecha, sin mezclar): train 2000-01 → 2018-09 (68%), validation 2018-09 → 2022-09 (16%), test 2022-09 → 2026-09 (16%).

---

## Q1. Dummies de mes + semana del mes

¿Cuál es el valor ABSOLUTO de la correlación de la dummy `<mes>_w<semana>` más correlacionada con `is_positive_growth_30d_future`? (3 decimales)

Pasos: `semana = (día - 1) // 7 + 1`, variable `month_wom` (ej. `October_w1`), agregarla a las categóricas (`Month`, `Weekday`, `Ticker`, `ticker_type`, `month_wom`), `pd.get_dummies` → 115 dummies (60 de `month_wom`), correlación con el target, ordenar por `abs_corr`.

- 0.015
- **0.025** ✓ (`October_w4`: +0.0246)
- 0.035
- 0.045

| Dummy | corr |
|---|---|
| October_w4 | +0.0246 |
| November_w3 | +0.0233 |
| February_w1 | −0.0209 |
| January_w2 | −0.0198 |

> Las correlaciones son muy bajas: la estacionalidad existe (fin de octubre / noviembre históricamente buenos, enero-febrero flojos) pero explica muy poco por sí sola.

## Q2. Nuevas reglas "a mano" con variables macro

¿Cuál es la precision de la mejor de las nuevas reglas (`pred3` o `pred4`) en TEST? (3 decimales)

- `pred3_manual_dgs10_5`: `(DGS10 <= 4.5) & (DGS5 <= 4)`
- `pred4_manual_dgs10_fedfunds`: `(DGS10 > 4) & (FEDFUNDS <= 4.795)`

- 0.558
- 0.568
- 0.578
- **0.588** ✓ (`pred3`: 0.5877)
- 0.598

| Regla | Positivos en TEST | Precision |
|---|---|---|
| pred0 `cci > 200` | 921 | 0.565 |
| pred1 `growth_30d > 1` | 19158 | 0.579 |
| pred2 `growth_30d > 1 & growth_snp500_30d > 1` | 15244 | 0.573 |
| **pred3 `DGS10 <= 4.5 & DGS5 <= 4`** | 15910 | **0.588** |
| pred4 `DGS10 > 4 & FEDFUNDS <= 4.795` | 15921 | 0.503 |

> Referencia: el 57% de los casos de TEST son positivos, así que "comprar siempre" ya da 0.570. pred3 le gana por ~2 puntos.

## Q3. Aciertos positivos únicos del árbol de profundidad 10 (`pred5_clf_10`)

¿Cuántos registros de TEST son un verdadero positivo de `pred5_clf_10` (target = 1 y pred5 = 1) mientras **todas** las reglas a mano (`pred0`-`pred4`) predicen 0?

Pasos: `DecisionTreeClassifier(max_depth=10, random_state=42)` entrenado en TRAIN+VALIDATION, predecir sobre todo el dataset, columna `only_pred5_is_correct`, contar en TEST.

- 271
- 571
- **1171** ✓ (obtuve 1187)
- 1571

> 1187 vs 1171: diferencia mínima por versión de scikit-learn/pandas (los empates en los cortes del árbol pueden resolverse distinto). La opción más cercana es claramente 1171. El árbol tiene precision 0.591 en TEST con 8628 positivos.

## Q4. Tuning de `max_depth` (1 a 12)

¿Cuál es la profundidad óptima según la precision en TEST?

- **4** ✓ (precision 0.629)
- 6
- 8
- 12

| max_depth | Precision VALID | Precision TEST |
|---|---|---|
| 1 | 0.601 | 0.570 |
| 2 | 0.665 | 0.623 |
| 3 | 0.642 | 0.572 |
| **4** | 0.653 | **0.629** |
| 6 | 0.710 | 0.606 |
| 8 | 0.737 | 0.622 |
| 10 | 0.780 | 0.591 |
| 12 | 0.808 | 0.582 |

> La precision en VALIDATION sube siempre con la profundidad (el árbol se entrenó con esos datos: **overfitting**), mientras que en TEST no mejora después de 4. `pred6_clf_best` (depth 4) supera a todas las predicciones anteriores (0.629 > 0.58).
>
> Ojo metodológico: elegir el hiperparámetro mirando TEST (como pide la tarea) contamina el test. Lo correcto sería entrenar en TRAIN, elegir con VALIDATION y medir una sola vez en TEST.

## Q5. [Exploratoria] ¿Qué datos faltan? (texto libre)

**Respuesta (para pegar en el formulario):**

> The tree splits mostly on macro variables (cpi_core_yoy, DGS10, FEDFUNDS, DGS1) and 365-day growth of oil and Bitcoin, i.e. on the market regime rather than on stock-specific information. I would add:
> 1. **Valuation and fundamentals per stock** (P/E, forward P/E, EV/EBITDA, earnings growth, from yfinance `Ticker.info` / quarterly financials, aligned by report date) — to separate cheap from expensive stocks within the same macro regime.
> 2. **Earnings events**: days to next earnings date and the last EPS surprise (yfinance `get_earnings_dates`), since 30-day returns often include an announcement.
> 3. **Relative strength**: stock growth minus its local index growth (S&P 500, Nifty 50, CAC 40) over 30/90 days — the current features only compare with the S&P 500, which is wrong for Indian and EU stocks.
> 4. **Currency**: USD/INR and EUR/USD changes for the non-US stocks.
> 5. **Yield curve and credit spreads**: DGS10 − DGS2 (FRED `T10Y2Y`) and high-yield spread (FRED `BAMLH0A0HYM2`), classic recession indicators.
> 6. **Volatility regime**: VIX level and its 30-day change, India VIX (`^INDIAVIX`), VSTOXX.
> 7. **Analyst revisions / news sentiment** (Polygon or Alpha Vantage news API).
>
> I would also replace ticker dummies with sector dummies (fewer features, generalizes to new stocks).

---

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 0.025 |
| Q2 | 0.588 |
| Q3 | 1171 |
| Q4 | 4 |
| Q5 | texto libre |

> Verificado contra las respuestas correctas publicadas en la plataforma (Q1-Q4 coinciden).
