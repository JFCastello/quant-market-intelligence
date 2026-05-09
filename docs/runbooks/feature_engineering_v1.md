# Feature Engineering v1

## 1. Propósito

Este documento congela la semántica del **feature set v1** del proyecto **Quant Market Intelligence** para la etapa de **features base** del pipeline.

Su función es dejar explícito:

- qué features se calculan;
- con qué fórmulas se calculan;
- qué columnas deben salir en la capa `data/features/`;
- qué decisiones de diseño se tomaron;
- qué queda fuera en esta etapa.

Este documento corresponde al **paso 16** del SDD: construcción de **features base**. No incluye todavía features de contexto de mercado cruzado; eso queda para el paso siguiente.

---

## 2. Alcance de esta versión

La versión `v1` incluye únicamente **features univariadas base** construidas a partir de la serie OHLCV normalizada de cada instrumento.

### Sí incluye

- retornos logarítmicos;
- volatilidad rolling;
- rango intradía;
- ATR normalizado;
- momentum simple;
- medias móviles;
- ratios entre medias móviles;
- drawdown rolling.

### No incluye todavía

- features cross-asset;
- contexto de mercado (`SPY`, `TLT`, `GLD`, `HYG` usados como variables entre sí);
- correlaciones rolling entre activos;
- spreads entre activos;
- calendario;
- targets;
- escalado para modelado.

---

## 3. Convenciones generales

## 3.1 Columna base de precio

La columna base de precio es:

```text
close
```

Se usa como referencia principal para:

- retornos;
- momentum;
- medias móviles;
- drawdown;
- normalización de ATR.

---

## 3.2 Factor de anualización

Para anualizar volatilidades rolling se usa:

```text
252
```

Interpretación: número aproximado de días bursátiles por año.

---

## 3.3 Política de warm-up

No se eliminan filas iniciales por falta de historial.

```text
drop_warmup_rows = false
```

Por tanto:

- las primeras filas pueden contener `NaN` estructurales;
- esos `NaN` no se consideran error;
- la poda eventual de filas quedará para una etapa posterior si el pipeline de modelado así lo requiere.

---

## 3.4 Columnas intermedias

No se persisten columnas auxiliares de cálculo en el output final.

```text
include_intermediate_columns = false
```

Ejemplos de columnas internas que pueden existir temporalmente durante el cálculo, pero **no deben persistirse**:

- `prev_close`
- `true_range`
- `rolling_max_20`
- `rolling_max_60`

---

## 3.5 Convención de nombres

Se adopta la siguiente convención:

- `log_ret_kd` para retornos logarítmicos a `k` días;
- `vol_kd` para volatilidad rolling anualizada con ventana `k`;
- `ma_k` para media móvil simple de ventana `k`;
- `ma_ratio_a_b` para la razón entre medias móviles `a` y `b`;
- `drawdown_k` para drawdown rolling sobre ventana `k`.

---

## 4. Input esperado

Cada instrumento entra al bloque de features como un DataFrame proveniente de `data/normalized/`, con al menos estas columnas:

- `instrument_id`
- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `provider`
- `ingested_at`

Además, el DataFrame debe venir:

- ordenado ascendentemente por fecha;
- sin duplicados por `instrument_id, date`.

---

## 5. Output esperado

Cada fila de output corresponde a una fecha de un instrumento y debe seguir el contrato `FeatureRow`.

### Columnas finales del feature set v1

```text
instrument_id
date
feature_version
log_ret_1d
log_ret_5d
vol_5d
vol_10d
vol_20d
vol_60d
hl_range
co_range
atr_14
mom_10d
ma_5
ma_20
ma_60
ma_ratio_5_20
ma_ratio_20_60
drawdown_20
drawdown_60
```

---

## 6. Definición formal de cada feature

# 6.1 Retornos logarítmicos

## `log_ret_1d`

### Definición

\[
\text{log\_ret\_1d}_t = \ln\left(\frac{close_t}{close_{t-1}}\right)
\]

### Interpretación
Cambio logarítmico de un día en el precio de cierre.

### Uso
- base para volatilidad rolling;
- señal de movimiento reciente.

### Warm-up
La primera fila no tiene valor válido.

---

## `log_ret_5d`

### Definición

\[
\text{log\_ret\_5d}_t = \ln\left(\frac{close_t}{close_{t-5}}\right)
\]

### Interpretación
Cambio logarítmico acumulado sobre 5 días.

### Uso
Señal de movimiento de corto plazo ampliado.

### Warm-up
Las primeras 5 filas no tienen valor válido.

---

# 6.2 Volatilidad rolling anualizada

La volatilidad rolling se calcula sobre `log_ret_1d`.

## Definición general

Para una ventana `k`:

\[
\text{vol}_{k,t} = \operatorname{std}\left(\text{log\_ret\_1d}_{t-k+1}, \dots, \text{log\_ret\_1d}_t\right) \cdot \sqrt{252}
\]

Donde `std` es la desviación estándar rolling sobre la ventana `k`.

---

## `vol_5d`

Volatilidad rolling anualizada de 5 días.

### Rol
Captura agitación muy reciente del activo.

---

## `vol_10d`

Volatilidad rolling anualizada de 10 días.

### Rol
Proporciona una memoria algo más estable que `vol_5d`.

---

## `vol_20d`

Volatilidad rolling anualizada de 20 días.

### Rol
Aproxima un horizonte tipo mensual bursátil.

---

## `vol_60d`

Volatilidad rolling anualizada de 60 días.

### Rol
Captura un estado de varianza más persistente.

---

# 6.3 Rango intradía

## `hl_range`

### Definición

\[
\text{hl\_range}_t = \frac{high_t - low_t}{close_t}
\]

### Interpretación
Amplitud relativa del rango diario usando el cierre como normalizador.

### Rol
Captura expansión o compresión intradía.

---

## `co_range`

### Definición

\[
\text{co\_range}_t = \frac{close_t - open_t}{open_t}
\]

### Interpretación
Movimiento relativo entre apertura y cierre.

### Rol
Señal simple de direccionalidad intradía.

---

# 6.4 ATR normalizado

## Paso intermedio: `prev_close`

\[
prev\_close_t = close_{t-1}
\]

No se persiste en el output final.

---

## Paso intermedio: `true_range`

\[
TR_t = \max\left(
high_t - low_t,
|high_t - prev\_close_t|,
|low_t - prev\_close_t|
\right)
\]

No se persiste en el output final.

---

## `atr_14`

### Definición

Primero se calcula el promedio rolling de 14 días del `true_range`:

\[
ATR^{raw}_{14,t} = \operatorname{mean}(TR_{t-13}, \dots, TR_t)
\]

Luego se normaliza por el cierre:

\[
atr\_14_t = \frac{ATR^{raw}_{14,t}}{close_t}
\]

### Interpretación
Medida suavizada del rango efectivo del precio, ajustada por escala del activo.

### Rol
Complementa a la volatilidad rolling con una métrica basada en rangos y gaps.

---

# 6.5 Momentum

## `mom_10d`

### Definición

\[
mom\_10d_t = \ln\left(\frac{close_t}{close_{t-10}}\right)
\]

### Interpretación
Momentum logarítmico a 10 días.

### Rol
Captura persistencia simple de movimiento direccional.

### Nota
En esta versión se usa una sola ventana de momentum para evitar redundancia excesiva.

---

# 6.6 Medias móviles

## `ma_5`

### Definición

\[
ma\_5_t = \operatorname{mean}(close_{t-4}, \dots, close_t)
\]

### Rol
Representa el nivel suavizado de corto plazo.

---

## `ma_20`

### Definición

\[
ma\_20_t = \operatorname{mean}(close_{t-19}, \dots, close_t)
\]

### Rol
Representa un nivel suavizado intermedio, aproximadamente mensual.

---

## `ma_60`

### Definición

\[
ma\_60_t = \operatorname{mean}(close_{t-59}, \dots, close_t)
\]

### Rol
Representa una tendencia más persistente.

---

# 6.7 Ratios entre medias móviles

## `ma_ratio_5_20`

### Definición

\[
ma\_ratio\_5\_20_t = \frac{ma\_5_t}{ma\_20_t}
\]

### Interpretación
Relación entre tendencia muy reciente y tendencia intermedia.

### Rol
Detecta aceleración, compresión o desacople entre corto plazo y horizonte mensual aproximado.

---

## `ma_ratio_20_60`

### Definición

\[
ma\_ratio\_20\_60_t = \frac{ma\_20_t}{ma\_60_t}
\]

### Interpretación
Relación entre tendencia media y tendencia más persistente.

### Rol
Señal de estructura de tendencia a mayor escala.

---

# 6.8 Drawdown rolling

## Definición general

Para una ventana `k`:

\[
rolling\_max_{k,t} = \max(close_{t-k+1}, \dots, close_t)
\]

\[
drawdown_{k,t} = \frac{close_t}{rolling\_max_{k,t}} - 1
\]

Por construcción, esta magnitud suele ser menor o igual a 0.

---

## `drawdown_20`

### Interpretación
Distancia relativa al máximo rolling de 20 días.

### Rol
Captura deterioro reciente frente a máximos cercanos.

---

## `drawdown_60`

### Interpretación
Distancia relativa al máximo rolling de 60 días.

### Rol
Captura deterioro más persistente frente a máximos de ventana más larga.

---

## 7. Features desactivadas en esta versión

## `context_market.enabled = false`

La versión `v1` **no** calcula todavía features contextuales de mercado.

### Ejemplos fuera de alcance

- retornos de otros activos usados como regresores;
- spreads entre ETFs;
- correlaciones rolling cross-asset;
- ratios entre activos;
- variables de calendario.

### Razón
Mantener el paso 16 enfocado en **features base univariadas** y dejar el bloque de contexto para el paso siguiente.

---

## 8. Consideraciones de implementación

## 8.1 Orden de cálculo recomendado

1. ordenar por fecha;
2. validar columnas mínimas;
3. calcular `prev_close`;
4. calcular retornos;
5. calcular volatilidades rolling;
6. calcular rangos intradía;
7. calcular `true_range`;
8. calcular `atr_14`;
9. calcular `mom_10d`;
10. calcular medias móviles;
11. calcular ratios de medias;
12. calcular drawdowns;
13. agregar `feature_version = "v1"`;
14. seleccionar columnas finales.

---

## 8.2 Validaciones mínimas esperadas

El output final debería cumplir al menos:

- columnas esperadas presentes;
- `instrument_id` no nulo;
- `date` no nula;
- `feature_version == "v1"`;
- sin duplicados por `instrument_id, date`;
- fechas ordenadas ascendentemente;
- sin `inf` ni `-inf`;
- `vol_* >= 0`;
- `atr_14 >= 0`;
- `drawdown_* <= 0` salvo tolerancias numéricas mínimas.

---

## 9. Resumen final

El feature set `v1` queda compuesto por las siguientes columnas de modelado:

```text
log_ret_1d
log_ret_5d
vol_5d
vol_10d
vol_20d
vol_60d
hl_range
co_range
atr_14
mom_10d
ma_5
ma_20
ma_60
ma_ratio_5_20
ma_ratio_20_60
drawdown_20
drawdown_60
```

Este documento debe considerarse la **fuente de verdad semántica** para implementar:

- `src/quant_platform/features/builders.py`
- `scripts/build_features.py`
- `scripts/check_features_quality.py`
- `tests/unit/test_feature_builders.py`

