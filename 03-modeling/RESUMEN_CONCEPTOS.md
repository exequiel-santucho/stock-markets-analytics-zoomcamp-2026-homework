# Módulo 3: Modelado analítico - Resumen de Conceptos Clave

## Introducción

Con la tabla única del Módulo 2 (33 acciones grandes de EE.UU., Europa e India, 2000-2026, ~300 features) planteamos el problema de predicción: **¿la acción va a estar más arriba dentro de 30 días hábiles?** Comparamos de menor a mayor complejidad: reglas a mano, modelos de series de tiempo (ARIMA) y árboles de decisión.

> Idea central: en finanzas la señal es muy débil. El objetivo no es acertar siempre sino tener **una ventaja chica pero estable** sobre el baseline, y medirla sin engañarse.

---

## 1. Formular hipótesis y elegir el target

| Target | Tipo | Ejemplo |
|---|---|---|
| `growth_future_30d` | Regresión | 1.05 = sube 5% en 30 días |
| `is_positive_growth_30d_future` | Clasificación binaria | 1 si `growth_future_30d > 1` |

- La clasificación binaria encaja con la decisión real: **comprar o no**.
- El horizonte (30 días) define qué features tienen sentido: macro y tendencia pesan más que patrones de velas de un día.

---

## 2. Series de tiempo: componentes y ARIMA

- **Descomposición:** tendencia + estacionalidad + resto (ruido).
- **ARIMA(p, d, q):** autorregresivo (p rezagos), integrado (d diferencias para volverla estacionaria) y media móvil (q errores pasados).
- En la clase: ARIMA(6,2,2) sobre AMZN con **walk-forward** (re-entrenar en cada paso con toda la historia). Es lento (un modelo por día y por acción) y solo usa el precio, así que no aprovecha el resto de las features.
- Conclusión práctica: para muchas acciones y muchas features conviene un modelo tabular (árboles) con el problema planteado como clasificación.

---

## 3. Split temporal (no aleatorio)

```python
train_end = min_date + (max_date - min_date) * 0.70
val_end   = train_end + (max_date - min_date) * 0.15
```

- En series de tiempo **nunca** se mezclan fechas: el modelo se entrena con el pasado y se evalúa en el futuro.
- Un split aleatorio filtra información: días vecinos son casi iguales (el target a 30 días de hoy y de mañana se solapa en 29 días), así que el modelo "vería" el test.
- Resultado: train 2000-2018, validation 2018-2022, test 2022-2026.

---

## 4. Variables categóricas y dummies

- `pd.get_dummies(df[['Month','Weekday','Ticker','ticker_type','month_wom']], dtype='int32')` → una columna 0/1 por valor.
- **Estacionalidad:** `month_wom` (mes × semana del mes) captura efectos tipo "fin de octubre". La correlación máxima fue 0.025: existe, pero es muy débil.
- Muchas dummies (60 de `month_wom` + 33 de `Ticker`) agregan ruido y riesgo de overfitting; en el Módulo 4 se prueba quitarlas.

---

## 5. Reglas "a mano" (heurísticas)

```python
new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)            # momentum
new_df["pred3_manual_dgs10_5"] = ((new_df.DGS10 <= 4.5) & (new_df.DGS5 <= 4)).astype(int)  # tasas bajas
```

- Son **baselines** interpretables: si un modelo no les gana, no sirve.
- Se pueden sacar de las primeras ramas de un árbol entrenado.
- Mejor regla: tasas de interés no muy altas (pred3) → precision 0.588.

---

## 6. Métrica: precision

**Precision = TP / (TP + FP):** de las veces que el modelo dice "comprar", ¿en cuántas subió?

- Es la métrica natural para una estrategia **long-only**: solo perdemos plata cuando compramos y baja (FP); no comprar algo que subió (FN) es una oportunidad perdida pero no una pérdida.
- **Baseline:** la proporción de positivos en TEST (0.570). Cualquier modelo útil tiene que superarla.
- Precision alta con muy pocas predicciones positivas no sirve: también importa **cuántas** operaciones genera (recall / cantidad de trades).

---

## 7. Árboles de decisión

```python
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
```

- Parten el espacio con preguntas tipo `cpi_core_yoy <= 0.02` eligiendo el corte que más reduce la impureza (Gini).
- **No necesitan escalar** features y manejan relaciones no lineales e interacciones.
- No aceptan `NaN` ni `±inf` (en la versión del curso): se reemplazan por 0.
- **`max_depth` controla la complejidad:** árboles profundos memorizan el train (en la tarea, precision en VALIDATION sube hasta 0.81 con depth 12, mientras que en TEST cae a 0.58).
- **Feature importance:** suma de la reducción de impureza por feature. Las más importantes fueron macro (`cpi_core_yoy`, `DGS10`, `FEDFUNDS`, `DGS1`) y crecimiento anual de petróleo y Bitcoin → el árbol aprende **regímenes de mercado**.
- `random_state` fija el desempate entre features con la misma ganancia; sin él cada corrida puede dar un árbol distinto.

---

## 8. Predicciones "únicas" (Q3)

Comparar qué casos acierta solo el modelo y ninguna regla responde: **¿aporta información nueva o repite las heurísticas?** El árbol de profundidad 10 encontró ~1.170 verdaderos positivos en TEST que ninguna regla detectaba. Esto justifica combinar modelos (ensambles) en vez de quedarse con uno.

---

## 9. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Dummies `month_wom` + correlación | 0.025 (`October_w4`) |
| Q2 | Reglas a mano macro | precision 0.588 (pred3) |
| Q3 | Verdaderos positivos únicos del árbol | 1171 (obtuve 1187) |
| Q4 | Tuning de `max_depth` | 4 (precision 0.629) |

---

## Preguntas típicas de entrevista

1. **¿Por qué no usar `train_test_split` con shuffle en series de tiempo?** Porque el modelo aprendería del futuro (fechas vecinas y targets solapados) y la métrica quedaría inflada.
2. **¿Qué métrica usarías para una estrategia que solo compra?** Precision (cuántas compras salen bien), junto con la cantidad de trades y el resultado financiero.
3. **¿Cómo detectás overfitting en un árbol?** La métrica en train/validation sigue subiendo con la profundidad pero en test se estanca o baja. Se controla con `max_depth`, `min_samples_leaf` o poda.
4. **¿Qué problema tiene elegir hiperparámetros mirando el test?** El test deja de ser una medición independiente (se sobreajusta al test). Hay que elegir con validación (o walk-forward CV) y usar el test una sola vez.
5. **¿Por qué los árboles no necesitan escalar features?** Porque solo comparan cada feature con un umbral; la escala no cambia el orden de los valores.
6. **¿Qué es ARIMA y cuándo lo usarías?** Modelo estadístico de una sola serie (autorregresión + diferencias + media móvil). Útil para pronosticar una serie con estructura temporal clara; poco práctico para cientos de acciones con muchas features.
7. **¿Qué es data leakage en este contexto?** Usar como feature algo calculado con datos futuros (`growth_future_*`, `bfill`, normalizar con estadísticas de todo el período).

## Recursos

- Notebook de la clase: `stock-markets-analytics-zoomcamp/03-modeling/[2026]_Module_03_Colab_Time_Series_Modeling.ipynb`
- Scikit-learn, árboles de decisión: https://scikit-learn.org/stable/modules/tree.html
- Código de la tarea: `hw3_questions.py`
