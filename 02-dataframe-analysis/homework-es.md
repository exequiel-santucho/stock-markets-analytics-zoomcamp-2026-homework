## Homework 2: Trabajar con los datos en Pandas (cohorte 2026)

> Enunciado original: `stock-markets-analytics-zoomcamp/cohorts/2026/homework2.md`
> Entrega: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw02 (vencía el 28/09/2026)
> Código: `hw2_questions.py`

Combinamos datos de varias fuentes (iposcoop.com, Yahoo Finance y el parquet del curso) para procesarlos en Pandas y generar campos nuevos.

> **Notas de entorno** (del propio enunciado, todas aplicadas en el código):
> - `yfinance` devuelve columnas MultiIndex aun para un solo ticker → `df.columns = df.columns.get_level_values(0)`.
> - En Windows, `pd.read_html(io.StringIO(resp.text))` en lugar de pasarle el texto directo.
> - Los conteos de filas pueden variar un poco: las tablas de iposcoop cambian con el tiempo y yfinance no siempre devuelve los mismos tickers.

---

## Q1. [IPO] IPOs retiradas por tipo de empresa

¿Cuál es el valor total (en millones de US$) de IPOs retiradas para el tipo de empresa con mayor valor retirado (antes del 11/09/2026)?

Pasos:
1. `pd.read_html` sobre [iposcoop.com/ipos-recently-filed](https://www.iposcoop.com/ipos-recently-filed/), filtrar `Expected To Trade == 'Withdrawn'` (el enunciado esperaba 32).
2. Columna **Company Type** por reglas, **en orden** (gana la primera que matchea, palabra exacta): Technologies → Acquisition Corp (`Acquisition Corp`, `Acquisition Corporation`, `Corp`) → Inc. → Group → Limited (`Ltd`, `Limited`) → Holdings → Other.
3. **Avg_price** = promedio de `Price Low` y `Price High` (`'$8.00'` → 8.0, `'-'` → NaN).
4. Pasar a número `Shares (millions)` y `Est $ Vol (millions)`.
5. **Shares_offered_value** = `Shares * Avg_price`, y si es nulo, `Est $ Vol`.
6. Sumar por **Company Type**.

- 200
- 300
- 400
- **500** ✓ (Acquisition Corp)

| Company Type | IPOs | Valor retirado (M$) |
|---|---|---|
| **Acquisition Corp** | 4 | **485.0** |
| Inc. | 1 | 351.0 |
| Holdings | 4 | 311.7 |
| Other | 8 | 290.4 |
| Limited | 6 | 191.2 |
| Technologies | 3 | 184.9 |
| Group | 3 | 32.5 |

> Hoy la tabla muestra solo 29 retiradas en el rango (la página lista las últimas 500 presentaciones y algunas viejas ya salieron). Por eso da 485 en lugar de ~500; la opción correcta sigue siendo la misma.

## Q2. [IPO] Sharpe ratio mediano de las IPOs de 2025 (primeros 8 meses)

¿Cuál es el Sharpe ratio mediano al 11/09/2026 de las empresas que salieron a bolsa antes del 1/09/2025?

Pasos:
1. Lista de 231 IPOs de 2025 de [iposcoop.com/2025-pricings](https://www.iposcoop.com/2025-pricings/).
2. `Offer Date` < 1/09/2025 y retorno distinto de 0% → 146 acciones (el enunciado: 148).
3. Bajar precios diarios con yfinance → 129 con datos (el enunciado: ~134; el resto está deslistado).
4. `growth_252d = Close / Close.shift(252)` y `volatility = Close.rolling(30).std() * sqrt(252)` (por ticker).
5. `Sharpe = (growth_252d - 0.05) / volatility` (tasa libre de riesgo 5%).
6. Filtrar el 11/09/2026 y `describe()`.

- -0.04
- **0.04** ✓ (mediana 0.046)
- 0.1
- 0.2

| | growth_252d | volatility | Sharpe |
|---|---|---|---|
| count | 126 | 128 | 126 |
| media | 1.023 | 28.4 | inf |
| mediana | **0.595** | 8.37 | **0.046** |

**Observaciones:**
- La **mediana** de `growth_252d` es 0.60 (la acción típica perdió 40% en un año) pero la **media** es 1.03: unos pocos casos que multiplicaron su precio (máx. ×33) inflan el promedio.
- 126 acciones llegaron a 252 días de cotización.
- Ojo con la fórmula del curso: la volatilidad está calculada sobre el **precio** (no sobre retornos), así que depende del nivel de precio y no es un Sharpe "de libro". Sirve para comparar dentro de la tarea, no como métrica real.

## Q3. [IPO] Estrategia de mantener N meses

¿Cuántos meses (1 a 12) conviene mantener una acción recién listada para maximizar la mediana del crecimiento (contra el cierre del primer día)?

Pasos: `future_growth_{m}_m = Close.shift(-21*m) / Close` por ticker, tomar el primer día de cada ticker (`min_date`), hacer inner join y `describe()`.

- **1** ✓ (mediana 0.940)
- 3
- 5
- 7

| Meses | Mediana | Media |
|---|---|---|
| 1 | **0.940** | 98.0 |
| 2 | 0.893 | 70.1 |
| 3 | 0.829 | 52.8 |
| 6 | 0.726 | 66.0 |
| 9 | 0.560 | 20.6 |
| 12 | 0.492 | 6.0 |

**Observaciones:**
- La mediana **cae casi todos los meses**: la IPO típica pierde valor después del primer día. Incluso el "mejor" horizonte da −6%.
- La media es enorme por un par de outliers (precios que van de centavos a dólares por splits inversos o errores de datos): la mediana describe mucho mejor a la acción típica.
- Para un inversor: comprar todas las IPOs al cierre del primer día y mantenerlas es una estrategia perdedora en promedio.

## Q4. [Estrategia] RSI < 30

¿Cuánto se habría ganado (en miles de US$) invirtiendo $1000 cada vez que una acción estaba sobrevendida (RSI < 30) entre 2000-01-01 y 2025-06-01, vendiendo a los 30 días?

`net_income = 1000 * (selected_df['growth_future_30d'] - 1).sum()`

- **65** ✓ ($65,806)
- 85
- 105
- 125

- 5206 operaciones, retorno medio a 30 días **+1.26%**, win rate **55.13%** (igual que el enunciado).

## Q5. [Exploratoria] ¿Cómo predecir una IPO con retorno positivo? (texto libre)

**Respuesta (para pegar en el formulario):**

> Since the typical IPO loses value after day 1, I would not buy every IPO but filter and time the entry:
> 1. **Wait for the lock-up expiry** (~180 days) and the post-IPO drift to play out, then buy only names that hold above their offer price; most of the decline happens in the first months.
> 2. **Quality filters at listing**: offer size (> $100M), reputable underwriters, profitable or fast-growing revenue, low float/insider selling, exclude SPACs/blank-check companies and micro-caps (they drive the negative median).
> 3. **First-day signals**: avoid IPOs with a huge first-day pop (over-hyped, mean-reverting); prefer those priced at or above the top of the range with moderate first-day returns.
> 4. **Market regime**: only trade when the IPO ETF (`IPO`) and the S&P 500 are above their 200-day moving average.
> 5. **Model**: train a classifier on these features (sector, size, underwriters, pricing vs range, first-day return, market momentum) to predict P(return over 6 months > 0), invest only above a high probability threshold, and add a stop-loss.

---

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 500 (Acquisition Corp) |
| Q2 | 0.04 |
| Q3 | 1 mes |
| Q4 | 65 |
| Q5 | texto libre |

> Verificado contra las respuestas correctas publicadas en la plataforma (Q1-Q4 coinciden).
