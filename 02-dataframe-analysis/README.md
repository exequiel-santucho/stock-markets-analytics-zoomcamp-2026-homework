# Módulo 2 - Trabajar con los datos en Pandas (Homework 2)

Solución del Homework 2 de Stock Markets Analytics Zoomcamp 2026.

## Archivos

- **hw2_questions.py**: resuelve Q1-Q4 e imprime tablas y resultados.
- **data/**: se crea sola. Cachea las tablas HTML de iposcoop, los precios de las IPOs (`stocks_df_ipos_2025.parquet`) y el parquet del curso con indicadores (`data.parquet`, ~130 MB, de Google Drive vía `gdown`).
- **pyproject.toml / uv.lock**: entorno reproducible (Python 3.13).

## Ejecutar

```bash
uv sync
uv run python hw2_questions.py
```

La primera corrida tarda unos minutos (baja ~150 tickers de Yahoo Finance y el parquet de 130 MB).

## Salida esperada (resumen)

```
Q1 -> Clase con más valor retirado: Acquisition Corp (485.0 M$)     -> opción 500
Q2 -> Sharpe mediano al 11/09/2026: 0.046                           -> opción 0.04
Q3 -> Mejor horizonte: future_growth_1_m con mediana 0.940          -> opción 1
Q4 -> Ganancia neta: $65,806 (~66 mil)                              -> opción 65
```

> iposcoop y Yahoo Finance cambian con el tiempo, así que los conteos pueden moverse una o dos filas
> (el enunciado lo advierte); las opciones están lo bastante separadas para que no cambie la respuesta.

## Qué practica

- Scraping con `pd.read_html`, limpieza de strings con `$` y comas, reglas de clasificación con regex (`\b` para palabra exacta).
- `groupby(...).shift()` / `transform(rolling)` para features por ticker sin mezclar series.
- Inner join contra la primera fecha de cada ticker para alinear retornos desde la IPO.
- Media vs mediana con distribuciones muy sesgadas.
