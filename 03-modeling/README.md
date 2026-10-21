# Módulo 3 - Modelado (Homework 3)

Solución del Homework 3 de Stock Markets Analytics Zoomcamp 2026: dummies de calendario, reglas "a mano", árboles de decisión y tuning de `max_depth`.

## Archivos

- **hw3_questions.py**: reproduce el pipeline del notebook de la clase (mismos sets de features, mismo split temporal 70/15/15) y resuelve Q1-Q4.
- **data/**: se crea sola; baja el dataset del curso `stocks_df_combined_2026_09_18.parquet.brotli` (~144 MB, Google Drive vía `gdown`).
- **pyproject.toml / uv.lock**: entorno reproducible (Python 3.13, pandas, scikit-learn).

## Ejecutar

```bash
uv sync
uv run python hw3_questions.py
```

Tarda ~7 minutos en una CPU de laptop (entrena 13 árboles sobre 169k filas × 299 features).

## Salida esperada (resumen)

```
Q1 -> Mayor |corr|: month_wom_October_w4 = 0.025
Q2    pred3_manual_dgs10_5   predicciones=1 en TEST: 15910  precision: 0.5877
Q3 -> Positivos en TEST que solo detecta pred5_clf_10: 1187       -> opción 1171
Q4 -> Mejor max_depth en TEST: 4 (precision 0.6292)
```

## Qué practica

- `pd.get_dummies` para variables de calendario y categóricas.
- Split temporal (sin shuffle) para series de tiempo.
- `DecisionTreeClassifier`, `precision_score`, `export_text` para leer las reglas del árbol.
- Overfitting: precision en validación vs test al aumentar la profundidad.
