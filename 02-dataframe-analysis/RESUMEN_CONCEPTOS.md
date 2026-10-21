# Módulo 2: Trabajar con los datos en Pandas - Resumen de Conceptos Clave

## Introducción

El objetivo del módulo es construir **un único DataFrame** listo para modelar: una fila por (fecha, ticker) con el precio, decenas de features derivadas (crecimientos, volatilidad, indicadores técnicos, macro) y lo que queremos predecir (el crecimiento futuro). Todo con Pandas.

> Idea central: el modelo solo es tan bueno como la tabla que le damos. La mayor parte del trabajo es **juntar, limpiar y transformar** sin filtrar información del futuro.

---

## 1. Librerías base

| Librería | Para qué |
|---|---|
| NumPy | Arrays y operaciones vectorizadas (`np.log`, `np.sqrt`, `np.where`) |
| Pandas | DataFrames: lectura, joins, groupby, ventanas temporales |
| Matplotlib / Seaborn | Gráficos estáticos |
| Plotly Express | Gráficos interactivos (velas, líneas) |
| TA-Lib | 150+ indicadores técnicos y patrones de velas |

---

## 2. Tipos de datos y limpieza

- **Numéricos** que llegan como texto: `'$8.00'`, `'1,234'`, `'-'`. Se limpian con `str.replace` y `pd.to_numeric(errors='coerce')` → lo que no se puede convertir queda `NaN`.
- **Fechas:** `pd.to_datetime(col, format='%m/%d/%Y')`. Con fechas se derivan `Year`, `Month`, `Weekday`, `Quarter`, semana del mes.
- **Strings:** clasificar por patrones con regex. `\b` marca borde de palabra: `\bCorp\b` no matchea "Corporate".
- **Stacking (UNION ALL):** `pd.concat([df_2024, df_2025])` apila tablas con las mismas columnas (por ejemplo, IPOs de varios años).
- **Dummies:** `pd.get_dummies(df[['Month','Weekday']], dtype='int32')` → una columna 0/1 por categoría (se usan en el Módulo 3).

---

## 3. Features a partir del OHLCV

Todas se calculan **por ticker** (si no, el `shift` mezcla la última fila de una acción con la primera de otra):

```python
g = df.groupby("Ticker")["Close"]
df["growth_30d"] = df["Close"] / g.shift(30)                  # pasado: se puede usar como feature
df["growth_future_30d"] = g.shift(-30) / df["Close"]          # futuro: SOLO como target
df["is_positive_growth_30d_future"] = (df["growth_future_30d"] > 1).astype(int)
df["SMA10"] = g.transform(lambda s: s.rolling(10).mean())
df["volatility"] = g.transform(lambda s: s.rolling(30).std()) * np.sqrt(252)
df["high_minus_low_relative"] = (df.High - df.Low) / df.Close
df["ln_volume"] = np.log(df.Volume + 1e-6)
```

- `shift(n)` mira hacia atrás (feature legítima); `shift(-n)` mira hacia adelante (target). Mezclarlos es **data leakage**.
- 252 ≈ días hábiles de un año; 21 ≈ un mes.
- `log(Volume)` comprime una variable con escala enorme y cola larga.

---

## 4. Riesgo y retorno: Sharpe ratio

**Sharpe = (retorno − tasa libre de riesgo) / volatilidad.** Mide el retorno por unidad de riesgo.

- Definición "de libro": retornos diarios → `mean * 252` y `std * sqrt(252)`.
- En la tarea se usó una versión simplificada (`growth_252d` y desvío del **precio**), útil para comparar pero no para reportar.
- Un Sharpe > 1 sostenido es muy bueno; negativo significa que rindió menos que el bono.

---

## 5. Macro y otros activos

| Serie | Fuente | Por qué |
|---|---|---|
| VIX (`^VIX`) | yfinance | "Índice del miedo": volatilidad implícita del S&P 500 |
| Oro (`GC=F`) | yfinance | Refugio en crisis |
| WTI / Brent (`CL=F`, `BZ=F`) | yfinance | Inflación, energía |
| Bitcoin (`BTC-USD`) | yfinance | Apetito de riesgo |
| DAX, S&P 500, Dow Jones | yfinance | Contexto de mercado (growth_snp500_30d, etc.) |
| Eurostat | API Eurostat | Macro europea |

Se unen por fecha con `merge(how='left')` y luego se rellenan los huecos (feriados distintos en cada mercado) con `ffill`, que solo usa valores pasados.

---

## 6. Indicadores técnicos (TA-Lib)

| Grupo | Ejemplos | Qué mide |
|---|---|---|
| Momentum | RSI, MACD, CCI, ADX, MOM, ROC | Fuerza y dirección de la tendencia |
| Volumen | OBV, AD, ADOSC | Si el volumen confirma el movimiento |
| Volatilidad | ATR, NATR | Amplitud de los movimientos |
| Ciclos | HT_DCPERIOD, HT_SINE | Componentes cíclicos (transformada de Hilbert) |
| Transformación de precio | AVGPRICE, TYPPRICE | Precios "promedio" del día |
| Patrones de velas | CDLDOJI, CDLHAMMER, ... (61) | Formaciones de 1-3 velas (valores −100/0/100) |

**RSI (Relative Strength Index):** 0-100. Por debajo de 30 = "sobrevendido"; por encima de 70 = "sobrecomprado". La Q4 lo usó como señal de compra: +1.26% medio a 30 días, 55% de aciertos.

---

## 7. IPOs: lo que mostraron los datos

- Las IPOs de 2025 tuvieron **mediana negativa** en todos los horizontes (−6% a 1 mes, −51% a 12 meses desde el primer cierre).
- La **media** es positiva por un puñado de casos extremos → con datos financieros sesgados hay que mirar mediana y percentiles.
- Muchas IPOs chicas (micro-caps, SPACs) desaparecen o se deslistan: yfinance no devuelve datos (sesgo de supervivencia en sentido contrario).

---

## 8. Lo que resolvimos en la tarea

| Q | Concepto | Resultado |
|---|---|---|
| Q1 | Scraping + reglas de clasificación + `groupby().sum()` | Acquisition Corp, ~500 M$ |
| Q2 | Features por ticker + Sharpe | mediana 0.04 |
| Q3 | Retornos futuros alineados desde la IPO | 1 mes (mediana 0.935) |
| Q4 | Estrategia por regla (RSI < 30) | ~$65 mil en 5206 trades |

---

## Preguntas típicas de entrevista

1. **¿Por qué hay que hacer `groupby('Ticker')` antes de `shift` o `rolling`?** Para no mezclar series de distintas acciones en la misma ventana.
2. **¿Cómo evitás data leakage al crear features temporales?** Solo usar información disponible al cierre de cada día: `shift(+n)`, `ffill`, nunca `bfill` ni `shift(-n)` en features.
3. **¿Media o mediana para retornos de IPOs?** Mediana (y percentiles): la distribución tiene cola derecha larga y la media queda dominada por outliers.
4. **¿Qué es el Sharpe ratio y qué limitaciones tiene?** Retorno en exceso por unidad de volatilidad. Asume que la volatilidad es todo el riesgo y penaliza igual las subas que las bajas (el Sortino solo penaliza las bajas).
5. **¿Qué significa RSI < 30?** Que en las últimas 14 ruedas dominaron las bajas: acción "sobrevendida", posible rebote (estrategia de reversión a la media).
6. **¿Inner, left u outer join para agregar datos macro a precios diarios?** Left join sobre la tabla de precios (no perder días de trading) y después `ffill` de los datos macro.

## Recursos

- Notebook de la clase: `stock-markets-analytics-zoomcamp/02-dataframe-analysis/[2026]_Module_02_Colab_Working_with_the_data.ipynb`
- TA-Lib: https://github.com/TA-Lib/ta-lib-python
- Código de la tarea: `hw2_questions.py`
