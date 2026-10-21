# Módulo 5 - Deployment y automatización

Pipeline diario del curso (`stock-markets-analytics-zoomcamp/05-deployment-and-automation/`) corriendo localmente en Windows:
datos → transformaciones → modelo → predicciones para el último día disponible.

> El enunciado del Homework 5 (cohorte 2026) todavía no está publicado. Esta carpeta deja listo el sistema sobre el que probablemente se base.

## Archivos

- **main.py**: orquesta los 4 pasos. Agregué el flag `--fast` (antes había que comentar/descomentar variables).
- **scripts/data_repo.py** (`DataRepository`): baja OHLCV de 190 acciones de EE.UU. (> US$50B de capitalización), índices (S&P 500, Dow, DAX, VIX, oro, petróleo, Bitcoin) y series macro de FRED; `persist()` / `load()` en parquet.
- **scripts/transform.py** (`TransformData`): features por ticker (crecimientos, medias móviles, volatilidad), ~100 indicadores y patrones de TA-Lib, y el join con índices y macro.
- **scripts/train.py** (`TrainModel`): sets de features, dummies, split temporal 70/15/15, `RandomForestClassifier(max_depth=17, n_estimators=200)`, guardado con `joblib` e inferencia con ranking diario.
- **results/**: logs de la corrida completa y de la rápida.
- **local_data/**: se crea sola (≈300 MB: parquets + modelo). No se sube a GitHub.

## Ejecutar

```bash
uv sync                                   # Python 3.11 (pandas-datareader no funciona en 3.12+)
uv run python main.py                     # completo: descarga + transforma + entrena + predice (~10 min)
uv run python main.py --fast              # rápido: usa local_data/ (~3 min, casi todo es cargar el modelo)
```

TA-Lib: el paquete `ta-lib` 0.6+ trae wheels para Windows y Linux con la librería C incluida, así que ya no hace falta instalarla aparte (en el curso se instalaba con `brew install ta-lib`).

## Salida (corrida del 21/10/2026)

```
Saved 223557 ticker records / 1710 index records / 14818 macro records
X_train (156227, 389), X_validation (33480, 389), X_test (33850, 389)
Top 3 predictions (2026-10-21):  WELL 0.635 · REGN 0.635 · CMG 0.630
Bottom 3 predictions (2026-10-21): COST 0.442 · V 0.439 · MA 0.423
```

`pred_rf_30d_best` es la probabilidad de que la acción esté más alta dentro de 30 días hábiles; `_rank` la ordena dentro de cada día (1 = la más prometedora).

## Automatizar (correr todos los días)

Linux / macOS (cron, de lunes a viernes 22:30 UTC, después del cierre de Wall Street):

```cron
30 22 * * 1-5 cd /ruta && /ruta/uv run python main.py >> results/cron.log 2>&1
```

Windows (Programador de tareas):

```powershell
schtasks /Create /TN "smazoomcamp-daily" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:30 `
  /TR "cmd /c cd /d D:\ruta\hw5-automation && uv run python main.py >> results\cron.log 2>&1"
```

## Problemas encontrados

- 4 de los 190 tickers no devuelven datos en yfinance: `MMC`, `FI`, `BK` y `SNYS` (mal escrito en la lista; Synopsys es `SNPS`).
  El *fallback* a Stooq también falla (`pandas_datareader` no expone `get_data_stooq` en la versión actual),
  así que se saltean y el pipeline sigue con 186 acciones.
- El modelo pesa ~150 MB (200 árboles de profundidad 17): en modo rápido la mayor parte del tiempo es deserializarlo.
