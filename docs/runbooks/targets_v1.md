# Targets v1

## 1. Propósito

Este documento congela la semántica del bloque de **targets v1** del proyecto **Quant Market Intelligence**.

Su función es dejar explícito:

- qué target continuo se construye en esta etapa;
- con qué fórmula exacta se calcula;
- qué convención temporal utiliza;
- qué filas quedan sin valor válido por falta de futuro;
- qué target discreto queda reservado para el paso siguiente.

En esta versión, el foco es exclusivamente el target continuo:

```text
future_rv_5d
```

El target discreto `future_regime_5d` queda definido a nivel de contrato y configuración, pero **todavía no se construye** en este paso.

---

## 2. Alcance de esta versión

### Sí incluye

- definición formal de `future_rv_5d`;
- uso de retornos logarítmicos diarios como base de cálculo;
- ventana estrictamente futura de 5 días;
- anualización del target;
- política de `NaN` en filas sin suficiente horizonte futuro.

### No incluye todavía

- discretización en regímenes;
- cálculo de cuantiles de régimen;
- umbrales `calm / normal / stress`;
- lógica de train/test para congelar thresholds;
- etiquetas discretas finales.

Eso corresponde al **paso 19**.

---

## 3. Definición del target continuo principal

## `future_rv_5d`

### Qué representa
La volatilidad realizada anualizada observada en los próximos 5 días de mercado para un activo dado.

### Qué pregunta responde
Para una fila fechada en `t`, responde:

> “¿Qué tan volátil fue realmente el activo entre `t+1` y `t+5`?”

### Intuición
No es una feature contemporánea ni pasada.  
Es una **etiqueta futura** que se usará para entrenar y evaluar modelos.

---

## 4. Columna base utilizada

El target se calcula a partir de:

```text
log_ret_1d
```

### Por qué se usa `log_ret_1d`
Porque esa columna ya forma parte del pipeline de features base, está alineada con la convención matemática del proyecto y evita recomputar retornos desde `close` en cada etapa.

### Implicación
El target depende de la serie de retornos logarítmicos diarios futuros del activo objetivo.

---

## 5. Convención temporal exacta

Para una fecha `t`, el target usa los cinco retornos diarios futuros:

- `log_ret_1d(t+1)`
- `log_ret_1d(t+2)`
- `log_ret_1d(t+3)`
- `log_ret_1d(t+4)`
- `log_ret_1d(t+5)`

### Importante
**No** usa `log_ret_1d(t)`.

### Razón
La etiqueta debe vivir estrictamente en el futuro respecto de la fila actual para evitar fuga de información.

---

## 6. Definición matemática formal

Sea \( r_{t+i} \) el retorno logarítmico diario del activo en el día \( t+i \). Entonces:

\[
future\_rv\_5d(t)
=
\sqrt{252}\;
\operatorname{std}\left(
r_{t+1}, r_{t+2}, r_{t+3}, r_{t+4}, r_{t+5}
\right)
\]

donde:

- `std` es la desviación estándar sobre esa ventana futura fija de 5 observaciones;
- `252` es el factor de anualización adoptado por el proyecto.

---

## 7. Factor de anualización

Se adopta:

```text
252
```

### Interpretación
Número aproximado de días bursátiles por año.

### Propósito
Hacer que `future_rv_5d` viva en una escala comparable con las volatilidades anualizadas del resto del sistema, por ejemplo:

- `vol_5d`
- `vol_10d`
- `vol_20d`
- `vol_60d`

---

## 8. Tipo de ventana

La ventana usada para el target es:

```text
fixed_forward
```

### Qué significa
No es una rolling window hacia atrás.  
Es una ventana **fija hacia adelante** a partir de la fecha actual.

### Implicación operativa
Cada fila usa una etiqueta que pertenece al futuro del activo.

---

## 9. Política de observaciones incompletas

## `allow_partial_window = false`

### Qué significa
El target solo se calcula si existen **los 5 retornos futuros completos**.

### Consecuencia
Las últimas 5 filas de cada instrumento deben quedar como:

```text
future_rv_5d = NaN
```

### Por qué
Porque no existe suficiente futuro observable para construir la etiqueta completa sin inventar datos o mezclar horizontes.

---

## 10. Política de `NaN`

En esta etapa, los `NaN` del target final son **esperados** al final de cada serie.

### Deben aparecer
- en las últimas 5 filas por instrumento.

### No deberían aparecer
- de forma arbitraria en el medio de la serie, salvo que haya problemas heredados en la columna base `log_ret_1d`.

### Interpretación
Un `NaN` al final es estructural; no indica error.

---

## 11. Contrato de salida esperado

Cada fila de target debe seguir el contrato `TargetRow`.

### Columnas esperadas

```text
instrument_id
date
target_version
future_rv_5d
future_regime_5d
```

### Nota
En esta etapa:

- `future_rv_5d` se construye;
- `future_regime_5d` puede permanecer en `None` / `NaN`, porque todavía no se activa.

---

## 12. Fuente de entrada operativa

El job de targets puede leer desde la capa:

```text
features_context_v1
```

pero el cálculo de `future_rv_5d` debe depender únicamente de:

- `instrument_id`
- `date`
- `log_ret_1d`

### Por qué
Porque el target es una propiedad futura del activo objetivo, no una transformación del contexto de mercado.

### Beneficio
Si más adelante cambias o amplías las context features, el target continuo no necesita redefinirse.

---

## 13. Relación con el paso 19

En el paso 19, el sistema transformará:

```text
future_rv_5d
```

en:

```text
future_regime_5d
```

usando cuantiles calculados **solo en train**.

### Razón
Eso evita fuga de información y hace comparable la clasificación entre benchmark y modelo ML.

### Etiquetas previstas
- `calm`
- `normal`
- `stress`

Pero esa lógica todavía no se ejecuta aquí.

---

## 14. Resumen operativo

En `targets v1`, el sistema debe:

1. tomar cada serie ordenada por `instrument_id` y `date`;
2. leer `log_ret_1d`;
3. para cada fecha `t`, mirar los 5 retornos futuros;
4. calcular su desviación estándar;
5. anualizar con `sqrt(252)`;
6. guardar el resultado como `future_rv_5d`;
7. dejar las últimas 5 filas como `NaN`;
8. no construir todavía `future_regime_5d`.

---

## 15. Resumen final

El bloque de targets v1 queda congelado así:

- target continuo principal: `future_rv_5d`
- columna base: `log_ret_1d`
- horizonte: 5 días futuros
- tipo de ventana: `fixed_forward`
- anualización: `252`
- ventanas parciales: no permitidas
- target discreto: reservado para el paso 19

Este documento debe considerarse la **fuente de verdad semántica** para implementar:

- `src/quant_platform/targets/builders.py`
- `scripts/build_targets.py`
- `scripts/check_targets_quality.py`
- `tests/unit/test_target_builders.py`
