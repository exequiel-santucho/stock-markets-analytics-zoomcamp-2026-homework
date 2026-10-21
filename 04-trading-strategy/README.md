# Módulo 4 - Estrategia de trading y simulación (Homework 4)

Solución del Homework 4 de Stock Markets Analytics Zoomcamp 2026.

## Archivos

- **hw4_pipeline.py**: versión en script del notebook `[2026]_Module_04_Colab_Trading_Simulations.ipynb`
  (dataset → dummies → split temporal → reglas manuales, árboles y Random Forest → reglas por umbral →
  simulación "$100 en cada predicción positiva"). Cada pregunta es un escenario que arranca de cero.
- **results/**: log de cada escenario (`run_*.log`) y la tabla `df_sim1_results` (`sim1_*.csv`).
- **data/**: se crea sola; baja el dataset del curso (~144 MB, Google Drive vía `gdown`).
- **pyproject.toml / uv.lock**: entorno reproducible (Python 3.13, pandas, scikit-learn).

## Ejecutar

```bash
uv sync
uv run python hw4_pipeline.py original   # control: reproduce el notebook (CAGR pred5_clf_10 = 1.1195)
uv run python hw4_pipeline.py q1         # Q1 y Q2: sin dummies de Ticker/month_wom + umbrales RF 0.56-0.64
uv run python hw4_pipeline.py q3         # Q3: target = growth_future_30d > 1.022
```

Cada escenario tarda ~2 minutos (3 árboles + un Random Forest de 100 árboles sobre 169k filas).

La red neuronal y la regresión logística del notebook no generan columnas `pred*` (el notebook usa resultados
precalculados), así que no afectan la simulación y no están en el script.

## Salida esperada (resumen)

```
original -> Mejor CAGR: pred5_clf_10 = 1.1195 (10.26 trades/día)
q1       -> Features NUMERICAL+DUMMIES: 208
            pred10_rf_best_rule_55  CAGR 1.1279                         -> Q1: opción 1.126
            pred13_rf_best_rule_57  4.85 trades/día, CAGR 1.0920        -> Q2: 0.57
q3       -> Mejor CAGR: pred5_clf_10 = 1.1218 (8.01 trades/día)         -> Q3: 8
```

## Qué practica

- Pasar de probabilidades (`predict_proba`) a reglas de decisión con distintos umbrales.
- Tabla precision/recall/cantidad de operaciones por umbral.
- Simulación vectorizada: ganancia bruta, comisiones, capital necesario y CAGR.
- Selección de features (menos dummies) y redefinición del target, y su efecto en el resultado financiero.
