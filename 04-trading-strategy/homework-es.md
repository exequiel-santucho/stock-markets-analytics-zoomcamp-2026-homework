## Homework 4: Estrategia de trading y simulación (cohorte 2026)

> Enunciado original: `stock-markets-analytics-zoomcamp/cohorts/2026/homework4.md`
> Entrega: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04 (**vence el 30/10/2026, 22:59 UTC**)
> Código: `hw4_pipeline.py` · resultados en `results/`

En esta tarea hay que entender el sistema completo (dataset → modelos → reglas de decisión → simulación financiera), cambiar unas pocas líneas y volver a correr todo para interpretar los resultados.

El enunciado pide partir de una **copia limpia** del notebook `[2026]_Module_04_Colab_Trading_Simulations.ipynb` para cada pregunta. Acá cada pregunta es un "escenario" del script:

| Escenario | Qué cambia | Comando |
|---|---|---|
| `original` | nada (control) | `uv run python hw4_pipeline.py original` |
| `q1` | Q1 + Q2 (Q2 parte del notebook de Q1) | `uv run python hw4_pipeline.py q1` |
| `q3` | Q3 | `uv run python hw4_pipeline.py q3` |

**Control:** el escenario `original` da CAGR = **1.1195** para el mejor predictor (`pred5_clf_10`), el mismo valor que cita el enunciado. El pipeline reproduce el notebook.

Simulación usada en todo el módulo (`sim1`): invertir **$100 en cada predicción positiva** del período TEST (sep-2022 a sep-2026), vender a los 30 días hábiles, 0.2% de comisión ida y vuelta. Capital necesario ≈ `100 × 30 × Q75(inversiones por día)` y `CAGR = ((capital + ganancia neta) / capital) ** (1/4)`.

---

## Q1. Menos features es más

**¿Cuál es el CAGR de `pred10_rf_best_rule_55` con el nuevo setup (menos dummies)?**

**La idea:** reducir la cantidad de features puede ayudar a los modelos. Algunos (regresión logística, redes neuronales) rinden peor con demasiadas dummies, sobre todo las de cada ticker y las de mes+semana.

1. **Redefinir `CATEGORICAL`:** sacar `Ticker` y `month_wom` justo antes de `pd.get_dummies()`. Control: el setup original tiene 301 features (`NUMERICAL + DUMMIES`) y el nuevo **208** (~31% menos).
2. **Correr todo el notebook** y mirar el CAGR de `pred10_rf_best_rule_55` al final (después del gráfico de burbujas). Se espera que sea *un poco mayor* que el original (1.1195).

Resultado: **208 features** (22 dummies) ✓. `pred10_rf_best_rule_55`: 3756 operaciones en TEST, ganancia neta $12,984, capital $21,000 → **CAGR 1.1279**, el mejor de todos los predictores en este setup.

- 1.121
- **1.126** ✓ (obtuve 1.128, la opción más cercana)
- 1.136
- 1.141

> Con la configuración original el mismo predictor daba 1.111 (y el mejor era `pred5_clf_10` con 1.1195). Al sacar 93 dummies "ruidosas", el Random Forest gana ~1.7 puntos de CAGR anual.

---

## Q2. La regla de decisión más eficiente del Random Forest

**¿Cuál es el umbral de probabilidad de la segunda mejor regla del Random Forest si queremos hacer en promedio MENOS de 5 operaciones por día?**

**La idea:** construir la tabla precision/recall por umbral (`tpr_fpr_dataframe`) para el Random Forest y conectarla con el resultado financiero. Ahora no se optimiza solo la precision, sino también el **CAGR** y la **eficiencia** (operaciones por día).

Pasos:
0. Partir de la copia del notebook de Q1 (setup corto, Random Forest con `max_depth=13`, `n_estimators=100`).
1. Usar `rf_best.predict_proba(X_test)` en lugar de `clf_best` para armar `df_scores`.
2. Definir predictores para los umbrales 0.56 a 0.64 (después de `pred10_rf_best_rule_55` y `pred11_rf_best_rule_65`): `pred12_rf_best_rule_56` ... `pred20_rf_best_rule_64`.
3. En `df_sim1_results`, buscar la regla con **< 5 operaciones/día promedio** y el mejor CAGR posible.

| Regla | Operaciones en TEST | Precision | Operaciones/día | CAGR |
|---|---|---|---|---|
| pred10 (0.55) | 3756 | 0.603 | 6.06 | **1.128** |
| pred12 (0.56) | 2780 | 0.599 | 5.35 | 1.101 |
| **pred13 (0.57)** | 2086 | 0.603 | **4.85** | **1.092** |
| pred14 (0.58) | 1519 | 0.603 | 4.49 | 1.067 |
| pred15 (0.59) | 1090 | 0.602 | 4.33 | 1.057 |
| pred16 (0.60) | 745 | 0.597 | 3.94 | 1.049 |
| pred17 (0.61) | 522 | 0.630 | 3.60 | 1.054 |
| pred20 (0.64) | 153 | 0.667 | 3.06 | 1.018 |
| pred11 (0.65) | 92 | 0.641 | 3.29 | 1.009 |

- 0.54
- 0.56 (5.35 operaciones/día: no cumple)
- **0.57** ✓ (4.85 operaciones/día, CAGR 1.092)
- 0.58

**Observaciones:**
- Como se esperaba: al subir el umbral, las operaciones (TP + FP) bajan mucho, pero la precision casi no mejora hasta 0.61 (~0.60 en todo el rango 0.50-0.60).
- **Más precision no implica más CAGR:** 0.64 tiene la mejor precision (0.667) y uno de los peores CAGR (1.018). Con pocas operaciones, el capital estimado no baja en la misma proporción que la ganancia.
- 0.57 es el "segundo mejor modelo": no le gana a 0.55 en CAGR, pero es el mejor dentro del subespacio de menos de 5 operaciones por día.

---

## Q3. Redefinir el target de crecimiento

**¿Cuántas operaciones (por día, en promedio) hace el mejor predictor `pred5_clf_10` después de redefinir el target?**

0. Copia limpia del notebook.
1. Mirar la distribución de `growth_future_30d`: mediana ≈ 1.021 (+2.1%), percentil 75 ≈ 1.085. Con `growth_future_30d > 1`, ~60% de los registros son clase 1.
2. Redefinir `is_positive_growth_30d_future = growth_future_30d > 1.022` → ~50% de clase 1.
3. Volver a correr todo y ver qué predictor da el mejor CAGR (probablemente `pred5_clf_10`, con CAGR parecido y menos operaciones por día).

Resultado: el mejor predictor sigue siendo **`pred5_clf_10`**: CAGR **1.1218** (antes 1.1195), con **8.01 operaciones por día** en promedio (antes 10.26) y 7337 operaciones en total en TEST.

- 6
- 7
- **8** ✓ (8.01 operaciones/día promedio)
- 9

**Observaciones:**
- Con un target más exigente el árbol compra menos (7337 vs 9861 operaciones) y logra un CAGR algo mayor: cada compra apunta a subas que superen comisiones y ruido.
- El Random Forest con umbral fijo 0.55 casi deja de operar (171 compras en 4 años): con clases balanceadas sus probabilidades se concentran cerca de 0.5. Los umbrales hay que recalibrarlos cada vez que cambia el target.
- Las reglas manuales (pred0-pred4) dan exactamente los mismos resultados financieros que antes: no dependen del target, solo cambia cómo se mide su precision.

---

## Q4. [Exploratoria] Describí tu estrategia ideal (texto libre)

**Respuesta (para pegar en el formulario):**

> I would combine the two best models and trade less but with more conviction:
> 1. **Signal:** buy only when both the Random Forest (setup without ticker dummies) gives probability ≥ 0.57 *and* the depth-10 Decision Tree trained on the stricter target (growth > 2.2%) predicts 1. Agreement between two different models filters out a lot of false positives.
> 2. **Selection:** each day, rank the candidates by RF probability and take at most the top 3 (this caps the time spent and the capital needed).
> 3. **Sizing:** split the available cash into 30 slots (one per day of the 30-day holding period) so money is always fully used and gains are reinvested; give 1.5x size to probabilities ≥ 0.62.
> 4. **Risk management:** 8% stop-loss and 15% take-profit per position, no more than 2 open positions in the same stock or 40% of capital in one country/sector, and no new entries when the S&P 500 is below its 200-day moving average or VIX > 30.
> 5. **Evaluation:** run an exact day-by-day simulation (iterating rows, with fees and reinvestment) and compare CAGR, Sharpe ratio and max drawdown against buy-and-hold of SPY over the same TEST period.

(En español: combinar el Random Forest con umbral 0.57 y el árbol entrenado con el target exigente, comprar solo cuando coinciden, máximo 3 por día por orden de probabilidad, capital dividido en 30 partes con reinversión, stop-loss 8% / take-profit 15%, límites de concentración, no operar en mercado bajista o con VIX alto, y comparar contra comprar y mantener el S&P 500.)

---

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 1.126 (CAGR 1.128) |
| Q2 | 0.57 |
| Q3 | 8 |
| Q4 | texto libre |

> Todavía sin verificar contra la plataforma: las respuestas correctas se publican después del 30/10. Q2 y Q3 caen justo en una opción; Q1 queda a 0.002 de 1.126 (posibles diferencias de versión de pandas/scikit-learn con Colab).
