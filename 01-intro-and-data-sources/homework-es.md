## Homework 1: Introducción y fuentes de datos (cohorte 2026)

> Enunciado original: `stock-markets-analytics-zoomcamp/cohorts/2026/homework1.md`
> Entrega: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw01 (vencía el 14/09/2026)
> Código: `hw1_questions.py`

En esta tarea bajamos datos financieros de distintas fuentes (Wikipedia, Yahoo Finance) y hacemos cálculos simples.

---

## Q1. [Índice] Empresas agregadas al S&P 500

¿Qué año (completo, desde 2020) tuvo más incorporaciones al índice?

Pasos: bajar la tabla de [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) con `pandas.read_html` (hace falta un `User-Agent`), extraer el año de `Date added` y contar por año.

- **2025** ✓ (17 empresas)
- 2024 (16)
- 2023 (14)
- 2022 (14)

> 2026 tiene 18, pero no es un año completo todavía.

*Adicional:* 223 empresas actuales llevan más de 20 años en el índice.

> Contexto: cuando una acción entra al S&P 500 suele subir, porque los fondos indexados están obligados a comprarla.

## Q2. [Macro] Índices mundiales YTD (al 21/08/2026)

¿Cuántos de estos 10 índices tuvieron mejor retorno en el año que el S&P 500 (1/1 al 21/08/2026, precio de cierre)?

- 1
- **2** ✓ (Nikkei 225 +27.8% y S&P/TSX +14.1% vs S&P 500 +11.4%)
- 3
- 4

| Índice | YTD |
|---|---|
| Japón - Nikkei 225 | +27.75% |
| Canadá - S&P/TSX | +14.06% |
| **EE.UU. - S&P 500** | **+11.41%** |
| Reino Unido - FTSE 100 | +8.01% |
| Alemania - DAX | +5.88% |
| Brasil - Ibovespa | +4.60% |
| Australia - ASX 200 | +4.08% |
| México - IPC | +0.32% |
| Hong Kong - Hang Seng | -2.43% |
| China - Shanghai | -2.97% |
| India - Nifty 50 | -7.32% |

*Adicional (3/5/10 años):* solo 2, 2 y 1 índices le ganan al S&P 500 (Nikkei y TSX; a 10 años solo el Nikkei). La tendencia es la misma: EE.UU. es difícil de batir en plazos largos (sin considerar el tipo de cambio).

## Q3. [Índice] Correcciones del S&P 500

Mediana del drawdown (%) de las correcciones (caída ≥ 5% desde el último máximo histórico), con datos diarios desde 1950.

Pasos: marcar los máximos históricos (`Close >= cummax`), para cada par de máximos consecutivos buscar el mínimo entre ambos, calcular `(high - low) / high * 100`, quedarse con las ≥ 5% y calcular duración y percentiles.

- **8** ✓ (74 correcciones; mediana 8.0%)
- 16
- 24
- 32

| Percentil | Drawdown | Duración (días hasta el mínimo) |
|---|---|---|
| 25% | 6.2% | 22 |
| 50% | 8.0% | 40.5 |
| 75% | 14.0% | 86 |

El top 10 coincide exactamente con el del enunciado (2008: 56.8% en 517 días, 2000-2002: 49.1%, ...).

## Q4. [Acciones] Sorpresas de ganancias de Amazon (AMZN)

Mediana del retorno a 2 días después de una sorpresa positiva de ganancias (EPS reportado > estimado).

Pasos: `yf.Ticker('AMZN').get_earnings_dates()` (25 fechas desde 2020-10-29), precios históricos, retorno `Close_Día3 / Close_Día1 - 1` donde el Día 2 es el del anuncio, filtrar sorpresas positivas.

- 3.35
- 2.35
- 1.35
- **0.35** ✓ (20 sorpresas positivas, mediana 0.35%)

*Adicional:*
- Correlación entre el tamaño de la sorpresa y el retorno: **0.22** (débil). El mercado ya descuenta buena parte; importa más la guía a futuro que el número del trimestre.
- Régimen (precio vs media de 200 días): en tendencia alcista (16 casos) la mediana es +0.35%; en bajista (4 casos) -3.4%. Con tan pocos casos no es concluyente, pero sugiere que la misma buena noticia rinde menos en un mercado bajista.

## Q5. [Exploratoria] Idea para el proyecto final (texto libre)

**Respuesta (para pegar en el formulario):**

> I want to build a short-term (20 trading days) prediction model for the largest Latin American stocks and ADRs (Brazil's Ibovespa and Mexico's IPC top-30 constituents, plus Argentine ADRs listed in NYSE). The model will predict whether a stock will beat its local index over the next month. Features: OHLCV-based technical indicators (RSI, MACD, ATR, Bollinger width), relative strength vs the local index and vs the S&P 500, USD/BRL and USD/MXN FX moves, commodity prices (oil, iron ore, soybeans) and US rates (DGS10, FEDFUNDS). I plan to compare Decision Trees, Random Forest and XGBoost, and simulate a long-only top-5 strategy with fees, benchmarked against buying the index ETF (EWZ/EWW).

(En español: modelo de corto plazo para acciones grandes de Latinoamérica que prediga si le ganan a su índice local en el próximo mes, usando indicadores técnicos, tipo de cambio, commodities y tasas de EE.UU.; comparar árboles, Random Forest y XGBoost y simular una estrategia top-5 contra el ETF del índice.)

## Q6. [Exploratoria] Investigar métricas nuevas (texto libre)

**Respuesta (para pegar en el formulario):**

> 1. **FX rates (USD/BRL `BRL=X`, USD/MXN `MXN=X`, yfinance)**: local stocks priced in local currency are strongly affected by currency moves; a weakening currency usually hurts importers and helps exporters.
> 2. **Commodity futures (Brent `BZ=F`, iron ore proxy `VALE`, soybeans `ZS=F`, yfinance)**: a large share of Ibovespa is commodities (Petrobras, Vale); their returns are driven by these prices.
> 3. **US 10Y yield and Fed Funds (`DGS10`, `FEDFUNDS`, FRED via pandas_datareader)**: emerging markets suffer capital outflows when US rates rise.
> 4. **Country risk / EM volatility (`^VIX`, EM ETF `EEM` 30-day volatility)**: a proxy for global risk appetite.
> 5. **Earnings dates and surprises (`yf.Ticker(t).get_earnings_dates()`)**: to avoid or exploit holding through announcements, as in Q4.
>
> All of them can be downloaded daily with `yf.download(...)` or `pdr.DataReader(series, 'fred', start)` and merged by date with the stock OHLCV table.

---

## Resumen de respuestas

| Q | Respuesta |
|---|---|
| Q1 | 2025 |
| Q2 | 2 |
| Q3 | 8 |
| Q4 | 0.35 |
| Q5 | texto libre (idea de proyecto) |
| Q6 | texto libre (métricas nuevas) |

> Verificado contra las respuestas correctas publicadas en la plataforma (Q1-Q4 coinciden).
