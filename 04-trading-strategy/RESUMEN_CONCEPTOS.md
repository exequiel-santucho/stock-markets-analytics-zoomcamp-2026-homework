# Módulo 4: Estrategia de trading y simulación - Resumen de Conceptos Clave

## Introducción

Un modelo con buena precision no garantiza ganar plata. Este módulo da el paso de **predicción → decisión → resultado financiero**: convertir probabilidades en reglas de compra, simular cuánto se habría ganado (con comisiones y capital necesario) y comparar estrategias por **CAGR** y por cantidad de operaciones.

> Idea central: se optimiza el resultado financiero, no la métrica de ML. Más precision con muchas menos operaciones puede dar menos ganancia total.

---

## 1. Cómo funciona una operación en la práctica

- Abrir cuenta en un broker, elegir el instrumento (acción, ETF), tipo de orden (market / limit) y tamaño.
- **Costos:** comisión por compra y por venta (en la clase 0.1% cada una → 0.2% ida y vuelta), spread, impuestos.
- **Riesgos:** pérdida máxima por operación (stop-loss), concentración, liquidez.
- **Timing:** cuándo entrar (cierre del día de la señal) y cuándo salir (horizonte fijo de 30 días, objetivo de ganancia, stop).

---

## 2. Más modelos sobre la misma tabla

| Modelo | Hiperparámetros elegidos | Comentario |
|---|---|---|
| Árbol de decisión | `max_depth` 4, 7, 10 | Rápido e interpretable; probabilidades "en escalones" (pocas hojas) |
| **Random Forest** | `n_estimators=100`, `max_depth=13` | Promedia 100 árboles con bootstrap + subconjunto de features: probabilidades más suaves y menos overfitting |
| Regresión logística | `C`, `max_iter` (requiere `StandardScaler`) | Precision ~0.55: las relaciones son no lineales y hay demasiadas dummies |
| Red neuronal (Keras) | 2 capas densas + dropout | Precision en test ~0.54: sobreajusta rápido; los datos tabulares ruidosos favorecen a los árboles |

Los hiperparámetros se buscaron con grillas (tardan 10-15 min) y en el notebook quedan los resultados precalculados.

---

## 3. De probabilidad a decisión: umbrales

```python
proba = rf_best.predict_proba(X_all)[:, 1]
new_df["pred10_rf_best_rule_55"] = (proba >= 0.55).astype(int)
new_df["pred11_rf_best_rule_65"] = (proba >= 0.65).astype(int)
```

- `predict` usa umbral 0.5 por defecto; elegir otro umbral es una **regla de decisión**.
- Tabla precision/recall por umbral (`tpr_fpr_dataframe`): al subir el umbral
  - sube la **precision** (las compras que hacemos son más seguras),
  - bajan el **recall** y la **cantidad de operaciones**.
- El árbol simple da pocas probabilidades distintas (una por hoja) → los umbrales saltan de golpe. El Random Forest da una distribución continua → permite ajustar fino.

---

## 4. Simulación vectorizada (sim1)

Regla: **invertir $100 en cada predicción positiva** y vender a los 30 días.

```python
gross = pred * 100 * (growth_future_30d - 1)
fees  = -pred * 100 * 0.002
net   = gross + fees
```

- **Capital necesario (aprox.):** `100 × 30 × Q75(inversiones por día)`: si cada día abrimos ~N posiciones y cada una dura 30 días, hay hasta 30·N posiciones abiertas a la vez.
- **CAGR (tasa anual compuesta):** `((capital + net) / capital) ** (1/4)` para los 4 años de test. CAGR 1.12 = 12% anual.
- **Inversiones promedio por día:** mide cuánto trabajo/atención requiere la estrategia.
- Gráfico de burbujas: CAGR vs. operaciones por día, tamaño = crecimiento del capital en 4 años. Se busca arriba a la izquierda (mucho retorno, pocas operaciones).

**Limitaciones de esta simulación:**
- El capital se estima con el percentil 75, no con la ocupación real día a día.
- No reinvierte ganancias (eso lo hace la simulación exacta iterando filas).
- Supone que siempre se puede comprar al cierre y no considera slippage ni impuestos.

---

## 5. Ideas de estrategias (de la clase)

| Estrategia | Idea |
|---|---|
| Buy & hold de una acción | Largo plazo; benchmark natural |
| Portafolio diversificado optimizado | Pesos que maximizan Sharpe (Markowitz) |
| Market-neutral (long/short) | Comprar las de mayor probabilidad y vender en corto las de menor |
| Reversión a la media | Comprar después de caídas fuertes o eventos (RSI bajo) |
| Pairs trading | Dos acciones correlacionadas: comprar la rezagada, vender la adelantada |
| Penny stocks / dividendos | Nichos con riesgo y dinámica propios |
| Opciones básicas | Coberturas o apuestas con riesgo acotado |

---

## 6. Cambios que probamos en la tarea

1. **Menos features (Q1):** quitar las dummies de `Ticker` y `month_wom` (301 → 208 features). Menos ruido → el Random Forest generaliza un poco mejor.
2. **Más reglas de umbral (Q2):** 0.55 a 0.65 de a 0.01 para el Random Forest, buscando la mejor relación CAGR / operaciones por día.
3. **Target más exigente (Q3):** positivo solo si sube más de 2.2% (≈ la mediana) en lugar de > 0%. Balancea las clases (60% → 50%) y hace que el modelo compre menos y mejor.

---

## 7. Lo que resolvimos en la tarea

| Q | Cambio | Resultado |
|---|---|---|
| Control | Notebook tal cual (301 features) | Mejor CAGR: `pred5_clf_10` = 1.1195 (igual que el enunciado); `pred10_rf_best_rule_55` = 1.111 |
| Q1 | Sin dummies de `Ticker` y `month_wom` (208 features) | `pred10_rf_best_rule_55`: CAGR **1.128** (6.1 operaciones/día) → pasa a ser el mejor predictor |
| Q2 | Umbrales del RF 0.56-0.64 | Con < 5 operaciones/día, el mejor CAGR es el umbral **0.57** (1.092, 4.85 operaciones/día) |
| Q3 | Target = `growth_future_30d > 1.022` | `pred5_clf_10` sigue siendo el mejor (CAGR 1.122) con **8** operaciones/día (antes 10.3) |

**Observaciones:**
- Sacar features **mejoró** el Random Forest: con 33 dummies de ticker y 60 de calendario, cada árbol gasta cortes en ruido.
- Subir el umbral reduce operaciones casi linealmente, pero el CAGR cae más rápido: con menos operaciones, la estimación del capital (`Q75 × 30 × $100`) no baja en la misma proporción.
- Con el target más exigente, el Random Forest casi no predice positivos (171 en 4 años con umbral 0.55): sus probabilidades se concentran cerca de 0.5 y los umbrales fijos dejan de servir. Los umbrales hay que recalibrarlos cada vez que cambia el target.

---

## Preguntas típicas de entrevista

1. **¿Por qué una mayor precision no implica más ganancia?** Porque la ganancia total depende de la cantidad de operaciones y del tamaño de cada acierto/error; un umbral muy alto opera poco y deja ganancias sobre la mesa.
2. **¿Qué es el CAGR y por qué se usa?** La tasa de crecimiento anual compuesta: permite comparar estrategias con distinta duración o capital.
3. **¿Cómo elegirías el umbral de probabilidad?** Barriendo umbrales en validación y eligiendo según el objetivo financiero (CAGR, Sharpe, drawdown) con restricciones operativas (máximo de operaciones por día, capital disponible).
4. **¿Por qué Random Forest suele ganarle a un árbol solo?** Promedia muchos árboles de alta varianza entrenados con datos y features distintos: reduce varianza sin aumentar mucho el sesgo.
5. **¿Qué sesgos tiene un backtest?** Look-ahead bias, sesgo de supervivencia, overfitting al período de test, ignorar costos/slippage/liquidez y no reinvertir capital.
6. **¿Qué benchmark usarías?** Comprar y mantener el índice (S&P 500 / ETF SPY) en el mismo período: si la estrategia no le gana ajustado por riesgo, no vale la pena.
7. **¿Por qué cambiar el target a "sube más de 2.2%"?** Para que el modelo aprenda a encontrar subas que superen comisiones y ruido, y para balancear las clases.

## Recursos

- Notebook de la clase: `stock-markets-analytics-zoomcamp/04-trading-strategy-and-simulation/[2026]_Module_04_Colab_Trading_Simulations.ipynb`
- Código de la tarea: `hw4_pipeline.py`
