# Módulo 5: Deployment y automatización - Resumen de Conceptos Clave

## Introducción

Hasta el Módulo 4 todo vivía en notebooks de Colab que se corren a mano. Un sistema de trading real tiene que correr **solo, todos los días**: bajar los datos nuevos, recalcular features, (re)entrenar o cargar el modelo, generar las predicciones y avisar qué operar. Este módulo hace ese salto: de notebook a scripts, de memoria a disco, y de "lo corro yo" a cron.

> Idea central: el mismo código de investigación, reorganizado en pasos independientes que guardan su resultado, se puede re-ejecutar, testear y programar.

---

## 1. De notebook a scripts (una clase por paso)

| Paso | Clase | Entrada | Salida en disco |
|---|---|---|---|
| 1. Datos | `DataRepository` | APIs (yfinance, FRED) | `tickers_df`, `indexes_df`, `macro_df` (.parquet) |
| 2. Transformación | `TransformData` | `DataRepository` | `transformed_df.parquet` |
| 3. Modelo | `TrainModel` | `TransformData` | `random_forest_model.joblib` |
| 4. Inferencia | `TrainModel.make_inference()` | modelo + último día | columnas `pred_*` y `pred_*_rank` |

- Cada clase tiene `fetch()/transform()/train...()` para calcular y `persist()/load()` para guardar o leer.
- `main.py` solo orquesta: decide en cada paso si **calcula** o **carga**. Eso da dos regímenes:
  - **Completo**: datos frescos + reentrenar (minutos, depende de las APIs).
  - **Rápido** (`--fast`): todo desde disco (reproducible, sin red, pero sin datos nuevos).
- Ventajas frente al notebook: sin estado oculto (celdas corridas fuera de orden), importable, versionable con git y testeable.

---

## 2. Persistencia

| Qué | Formato | Por qué |
|---|---|---|
| DataFrames | **Parquet** (+ compresión brotli) | Columnar, tipado, 5-10× más chico que CSV, lectura rápida de columnas |
| Modelo sklearn | **joblib** | Serializa objetos Python con arrays NumPy grandes de forma eficiente |
| Histórico incremental | **SQLite** / base de datos | Insertar solo los días nuevos en lugar de bajar todo de nuevo |

- Cuidado con joblib/pickle: el modelo solo se puede cargar con la **misma versión** de scikit-learn, y nunca hay que cargar archivos de origen desconocido (pickle ejecuta código).
- Un Random Forest de 200 árboles y profundidad 17 ocupa ~150 MB: el tamaño del modelo también es un criterio de diseño.

---

## 3. Fuentes de datos robustas

- Las APIs gratuitas fallan: rate limits, tickers deslistados, cambios de formato (yfinance pasó a devolver columnas MultiIndex).
- El curso usa una **cadena de *fallbacks***: yfinance → Stooq → FRED. Si ninguna responde, se salta ese ticker y el pipeline sigue.
- Validar lo que llega: cantidad de filas por ticker, fechas faltantes, valores 0/NaN. En la corrida de la tarea, `MMC`, `FI` y `BK` no devolvieron datos y `SNYS` está mal escrito en la lista: quedaron 186 de 190 acciones.

---

## 4. Automatización

| Herramienta | Cuándo |
|---|---|
| **cron** (Linux/macOS) / Programador de tareas (Windows) | Un script por día, una sola máquina |
| **Apache Airflow / Prefect / Dagster** | Varios pasos con dependencias, reintentos, monitoreo, *backfills* |
| GitHub Actions / cloud schedulers | Sin servidor propio; correr en la nube con un cron |

Formato cron: `minuto hora día-del-mes mes día-de-la-semana`. Ej.: `30 22 * * 1-5` = 22:30 de lunes a viernes.

Buenas prácticas:
- Correr **después del cierre** del mercado (datos del día completos).
- Mandar la salida a un log y avisar si falla (email, Telegram).
- Separar el **reentrenamiento** (semanal o mensual) de la **inferencia** (diaria): el modelo no necesita reentrenarse todos los días.

---

## 5. Del modelo a la operación

- La inferencia ordena las acciones de cada día por probabilidad (`rank`): el top-k es la lista de compras candidata.
- **Simulación avanzada** del notebook del módulo: invertir una fracción del capital cada día, reinvertir las ganancias, elegir el top-k por encima de un umbral, tamaño proporcional a la probabilidad, **stop-loss** y **take-profit** usando el mínimo/máximo de los próximos días. Es mucho más realista que "$100 por señal" del Módulo 4.
- Siguiente paso (opcional del curso): conectar con la API de un broker para ejecutar las órdenes y mandar un resumen diario por email o Telegram.

---

## Preguntas típicas de entrevista

1. **¿Por qué pasar un notebook a scripts antes de ponerlo en producción?** Estado reproducible, código modular y testeable, versionado y la posibilidad de programarlo.
2. **¿Parquet o CSV para guardar datos intermedios?** Parquet: es columnar, guarda tipos, comprime mucho mejor y lee solo las columnas necesarias.
3. **¿Cada cuánto reentrenarías el modelo?** Depende del *drift*: inferencia diaria, reentrenamiento periódico (semanal o mensual) o cuando caen las métricas en producción.
4. **¿Cron o Airflow?** Cron para un script simple; Airflow cuando hay DAGs de varios pasos, reintentos, dependencias entre tareas y necesidad de monitoreo.
5. **¿Qué puede salir mal en un pipeline que depende de APIs gratuitas?** Rate limits, datos faltantes o corregidos después, cambios de formato, tickers que cambian o se deslistan. Se mitiga con *fallbacks*, validaciones y alertas.
6. **¿Qué riesgo tiene cargar un modelo con pickle/joblib?** Incompatibilidad de versiones y ejecución de código arbitrario si el archivo no es de confianza.
7. **¿Cómo evitás *look-ahead bias* en la inferencia diaria?** Usar solo información disponible al cierre del día de la predicción. Las features "futuras" (`growth_future_*`) solo sirven como target al entrenar y quedan vacías en los últimos 30 días.

## Recursos

- Código del curso: `stock-markets-analytics-zoomcamp/05-deployment-and-automation/` (`main.py`, `scripts/`, `Module_05_Advanced_Strategies_And_Simulation.ipynb`)
- Código local: esta carpeta
