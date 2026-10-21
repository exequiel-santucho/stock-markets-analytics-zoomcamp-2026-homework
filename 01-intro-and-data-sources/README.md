# Módulo 1 - Introducción y fuentes de datos (Homework 1)

Solución del Homework 1 de Stock Markets Analytics Zoomcamp 2026.

## Archivos

- **hw1_questions.py**: resuelve Q1-Q4 (más las preguntas adicionales) e imprime cada resultado.
- **data/**: se crea sola; el script cachea lo que baja de Wikipedia y Yahoo Finance.
- **pyproject.toml / uv.lock**: entorno reproducible (Python 3.13, pandas, yfinance, lxml).

## Ejecutar

```bash
uv sync
uv run python hw1_questions.py
```

> Wikipedia y Yahoo Finance son fuentes vivas. Las fechas de corte están fijas en el código
> (las del enunciado), pero la lista del S&P 500 se baja tal como está hoy.
> Para recalcular desde cero, borrar `data/`.

## Salida esperada (resumen)

```
Q1 -> Año con más incorporaciones (2020-2025): 2025 (17 empresas)
Q2 -> Índices con mejor YTD que el S&P 500 (11.41%): 2 de 10
Q3 -> Mediana del drawdown: 8.0%   (74 correcciones)
Q4 -> Mediana del retorno 2 días tras sorpresa positiva: 0.35%
```

## Qué practica

- Scraping de tablas HTML con `requests` + `pd.read_html(io.StringIO(...))`.
- `yf.download` (columnas MultiIndex → `get_level_values(0)`), `yf.Ticker(...).get_earnings_dates()`.
- Series temporales en pandas: `cummax`, `shift`, ventanas entre fechas, `quantile`, `corr`.
