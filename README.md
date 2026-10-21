# Stock Markets Analytics Zoomcamp 2026 - Homework

Mis soluciones de las tareas de [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) de [DataTalksClub](https://datatalks.club/) y [PythonInvest](https://pythoninvest.com/), junto con un resumen de conceptos por módulo.

## Contenido

| Módulo | Tema | Herramientas |
|---|---|---|
| [01-intro-and-data-sources](01-intro-and-data-sources/) | Fuentes de datos: Wikipedia, Yahoo Finance, FRED; drawdowns del S&P 500, sorpresas de earnings | pandas, yfinance, requests |
| [02-dataframe-analysis](02-dataframe-analysis/) | Un único DataFrame: scraping de IPOs, features por ticker, Sharpe ratio, estrategia RSI | pandas, yfinance, pyarrow |
| [03-modeling](03-modeling/) | Dummies de calendario, reglas a mano, árboles de decisión y tuning | scikit-learn |
| [04-trading-strategy](04-trading-strategy/) | Random Forest, reglas por umbral de probabilidad y simulación de la estrategia (CAGR) | scikit-learn |

Cada carpeta tiene:

- `README.md`: cómo correr la solución y la salida esperada.
- `homework-es.md`: enunciado en español con las respuestas y el análisis.
- `RESUMEN_CONCEPTOS.md`: resumen del módulo para repasar.
- El script `hwN_*.py` y el entorno (`pyproject.toml` + `uv.lock`).

## Cómo correr

Python 3.13 + [uv](https://docs.astral.sh/uv/):

```bash
cd 02-dataframe-analysis
uv sync
uv run python hw2_questions.py
```

Los datos se descargan solos la primera vez (Wikipedia, iposcoop.com, Yahoo Finance y los datasets del curso en Google Drive) y quedan cacheados en `data/` (no se suben al repo).

> Wikipedia, iposcoop y Yahoo Finance son fuentes vivas: las fechas de corte están fijas en el código, pero algunos conteos pueden variar un poco respecto de los del enunciado.

#smazoomcamp
