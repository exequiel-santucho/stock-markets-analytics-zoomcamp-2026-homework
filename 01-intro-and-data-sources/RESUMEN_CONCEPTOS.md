# Módulo 1: Introducción y fuentes de datos - Resumen de Conceptos Clave

## Introducción

El curso enseña a pasar de "mirar gráficos" a **tomar decisiones de inversión con datos**: juntar datos de mercado y macroeconómicos, transformarlos en features, entrenar modelos que predigan retornos futuros, simular una estrategia y automatizarla. Este primer módulo arma la base: **de dónde salen los datos**, qué significan y cómo bajarlos con Python.

> Idea central: antes de modelar hay que entender **qué datos existen, con qué frecuencia se publican, cuánto cuestan y cuándo están disponibles** (para no usar información del futuro).

---

## 1. Decisiones basadas en datos

Preguntas que guían todo el curso:

- **¿En qué invertir?** Acciones, ETFs, índices, bonos, commodities, cripto. Cada uno tiene distinto riesgo, liquidez y horizonte.
- **Riesgo vs. retorno:** más retorno esperado implica más volatilidad. Un buen análisis compara ambos (por eso aparece el Sharpe ratio en el Módulo 2).
- **¿Dónde poner la atención?** No se puede seguir todo: se eligen universos (las acciones más grandes, un país, un sector) y horizontes (diario, semanal, mensual).

| Horizonte | Qué domina el precio | Ejemplo de datos útiles |
|---|---|---|
| Intradía / días | Flujo de órdenes, noticias, momentum | OHLCV, volumen, noticias |
| Semanas / meses | Resultados, sorpresas de ganancias, tendencia | Earnings, indicadores técnicos |
| Años | Macro, valuación, crecimiento de ganancias | PBI, inflación, tasas, ratios fundamentales |

---

## 2. Datos macroeconómicos (FRED)

[FRED](https://fred.stlouisfed.org/) (Reserva Federal de St. Louis) es gratis y se lee con `pandas_datareader`:

```python
import pandas_datareader as pdr
gdp = pdr.DataReader("GDPPOT", "fred", start="2000-01-01")
```

| Serie | Qué es | Por qué importa |
|---|---|---|
| `GDPPOT` | PBI potencial real de EE.UU. (trimestral) | Crecimiento de la economía → ganancias de las empresas |
| `CPILFESL` | CPI core (inflación sin alimentos ni energía) | Inflación alta → la Fed sube tasas |
| `FEDFUNDS` | Tasa de política monetaria | Costo del dinero; tasas altas bajan las valuaciones |
| `DGS1`, `DGS5`, `DGS10` | Rendimiento de bonos del Tesoro a 1, 5, 10 años | Tasa "libre de riesgo"; la curva anticipa recesiones |

**Transformaciones clave:**
- `yoy` (year over year): `x / x.shift(4) - 1` en datos trimestrales, `shift(12)` en mensuales.
- `qoq` / `mom`: contra el trimestre / mes anterior.
- **Desfase de publicación:** el PBI de un trimestre se publica semanas después. Si lo unimos por fecha del período, el modelo "ve el futuro" (**look-ahead bias**). Hay que unir por fecha de publicación o desplazarlo.

---

## 3. Datos de mercado (OHLCV) con yfinance

OHLCV = **O**pen, **H**igh, **L**ow, **C**lose, **V**olume por período.

```python
import yfinance as yf
df = yf.download("^GSPC", start="1950-01-01", auto_adjust=True)
df.columns = df.columns.get_level_values(0)   # las versiones nuevas devuelven MultiIndex
```

- **Tickers de índices** empiezan con `^` (`^GSPC`, `^N225`, `^GDAXI`); los de otras bolsas tienen sufijo (`.SS` Shanghai, `.NS` India, `.PA` París).
- **ETFs** (`SPY`, `EEM`, `GLD`) replican índices o activos y se pueden comprar como una acción.
- **Close vs Adj Close:** el ajustado corrige por dividendos y splits. Para calcular retornos hay que usar precios ajustados (`auto_adjust=True`).
- `end` en `yf.download` es **exclusivo** (no incluye ese día).

### Retorno y crecimiento

```python
growth_30d = close / close.shift(30)          # >1 subió, <1 bajó
ytd = close.iloc[-1] / close.iloc[0] - 1      # retorno en el año
```

---

## 4. Drawdowns y correcciones (Q3)

- **Máximo histórico (ATH):** día en que el precio supera todos los anteriores → `close >= close.cummax()`.
- **Drawdown:** caída desde el último máximo: `(high - low) / high`.
- **Corrección:** drawdown ≥ 5-10%; **bear market:** ≥ 20%.

Lo que encontramos en el S&P 500 (1950-2026): 74 correcciones ≥ 5%, mediana **8%**, y la mitad toca el mínimo en menos de **41 días**. Solo 9 superaron el 25%. Conclusión práctica: las caídas chicas son frecuentes y suelen recuperarse; vender por pánico en cada una es caro.

---

## 5. Fuentes pagas vs. gratis

| Fuente | Tipo | Gratis | Comentario |
|---|---|---|---|
| Yahoo Finance (`yfinance`) | Precios, dividendos, earnings, fundamentals | Sí | No oficial: puede cambiar o cortar; ideal para aprender |
| FRED | Macro EE.UU. | Sí | Muy confiable, con historial largo |
| Polygon.io | Precios, noticias, opciones | Plan limitado | Datos intradía y noticias con API estable |
| Alpha Vantage | Precios, indicadores, fundamentals | 25 req/día | Bueno para prototipos |
| EDGAR (SEC) | Reportes 10-K / 10-Q | Sí | Fuente primaria de los estados contables |

**¿Cuándo pagar?** Cuando necesitás datos intradía, sin sesgo de supervivencia (empresas deslistadas), con SLA, o para operar con dinero real. Para aprender y hacer backtests diarios, lo gratis alcanza.

---

## 6. Web scraping

```python
import io, requests, pandas as pd
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."}).text
tables = pd.read_html(io.StringIO(html))      # devuelve una lista de DataFrames
```

- Muchos sitios bloquean requests sin `User-Agent`.
- Envolver el HTML en `io.StringIO` evita el `OSError` de pandas/lxml en Windows.
- Ejemplo de la tarea: la lista del S&P 500 de Wikipedia, con la fecha en que entró cada empresa.

---

## 7. Eventos: earnings (Q4)

- **EPS** (earnings per share) estimado por analistas vs. reportado. **Sorpresa % = (reportado − estimado) / |estimado|**.
- `yf.Ticker("AMZN").get_earnings_dates(limit=N)` devuelve fechas pasadas y la próxima.
- Medimos el retorno con una **ventana de 3 días** (día anterior → día posterior al anuncio), porque las empresas suelen reportar después del cierre.

Resultado en AMZN: mediana +0.35% y correlación 0.22 con el tamaño de la sorpresa. La reacción depende más de la guía a futuro y de lo que ya estaba descontado que de "superar" el número.

---

## 8. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Scraping + `value_counts` por año | 2025 (17 incorporaciones) |
| Q2 | Retorno YTD con yfinance | 2 índices le ganan al S&P 500 |
| Q3 | Máximos históricos y drawdowns | mediana 8% |
| Q4 | Eventos de earnings + ventana de retorno | mediana 0.35% |

---

## Preguntas típicas de entrevista

1. **¿Qué es look-ahead bias y cómo aparece con datos macro?** Usar en una fecha un dato que todavía no se había publicado. Se evita uniendo por fecha de publicación o desplazando la serie.
2. **¿Por qué usar precio ajustado para retornos?** Porque los dividendos y splits generan saltos en el precio que no son ganancia o pérdida real.
3. **¿Qué es el sesgo de supervivencia?** Analizar solo empresas que siguen existiendo (por ejemplo, la lista actual del S&P 500) sobrestima los retornos históricos, porque las que quebraron o salieron no están.
4. **¿Diferencia entre drawdown, corrección y bear market?** Drawdown es cualquier caída desde un máximo; corrección suele ser ≥ 10% (en la tarea usamos 5%); bear market ≥ 20%.
5. **¿Por qué comparar índices de distintos países sin ajustar por moneda puede engañar?** Un índice en moneda local puede subir mucho mientras la moneda se devalúa; en dólares el retorno puede ser negativo.
6. **¿yfinance sirve para producción?** Para aprender y prototipar sí; para producción conviene una API oficial con SLA, porque yfinance hace scraping de Yahoo y puede cambiar sin aviso.

## Recursos

- Notebook de la clase: `stock-markets-analytics-zoomcamp/01-intro-and-data-sources/[2026]_Module_01_Colab_Introduction_and_Data_Sources.ipynb`
- FRED: https://fred.stlouisfed.org/
- Código de la tarea: `hw1_questions.py`
