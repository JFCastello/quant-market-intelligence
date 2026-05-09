# Regime Targets by Split v1

## 1. Propósito

Este documento congela la semántica de la capa **regime_targets_by_split_v1** del proyecto Quant Market Intelligence.

Su objetivo es dejar explícito cómo debe construirse correctamente `future_regime_5d` cuando el proyecto ya opera con:

- target continuo `future_rv_5d`;
- splits walk-forward;
- separación estricta entre `train`, `validation` y `test`.

Este documento existe para evitar una discretización global incorrecta del régimen y para asegurar que la definición del target discreto respete la metodología temporal del proyecto.

---

## 2. Problema que resuelve

El target continuo principal del proyecto es:

```text
future_rv_5d
```

El target discreto derivado es:

```text
future_regime_5d
```

Pero `future_regime_5d` no debe calcularse usando cuantiles de toda la historia completa, porque eso introduciría fuga de información.

En su lugar, los umbrales de discretización deben calcularse únicamente con el bloque `train` de cada fold.

---

## 3. Regla central

La regla congelada para esta capa es:

```text
regime_thresholds_source = train_only
```

Esto significa:

- los thresholds del régimen se estiman solo con observaciones del bloque `train`;
- `validation` y `test` no participan en la estimación de thresholds;
- esos thresholds luego se aplican al resto de filas del fold.

---

## 4. Relación con `future_rv_5d`

`future_regime_5d` no es un target independiente.

Es una discretización de:

```text
future_rv_5d
```

Por tanto, esta capa depende completamente de que el target continuo ya exista y esté correctamente calculado.

La capa de régimen no redefine el target continuo; solo lo transforma en clases discretas.

---

## 5. Relación con los splits walk-forward

Esta capa depende de la existencia de folds temporales válidos.

Cada fold debe aportar, como mínimo:

- un bloque `train`;
- un bloque `validation` si aplica;
- un bloque `test`.

La discretización del régimen se hace fold por fold, no de forma global.

---

## 6. Qué significa “por split”

Construir el régimen “por split” significa que para cada `split_id`:

1. se seleccionan las filas pertenecientes a ese fold;
2. se toma solo el subconjunto `train`;
3. se calculan los cuantiles definidos en configuración;
4. se construyen los thresholds;
5. se aplican las etiquetas a todas las filas del fold.

Por tanto, cada `split_id` puede tener thresholds distintos.

---

## 7. Etiquetas esperadas

La capa `v1` usa tres clases:

```text
calm
normal
stress
```

La interpretación es:

- `calm`: baja volatilidad futura relativa al train del fold;
- `normal`: zona intermedia;
- `stress`: alta volatilidad futura relativa al train del fold.

Estas clases no son absolutas a nivel histórico completo; son relativas al contexto estadístico del fold.

---

## 8. Por qué no debe usarse una discretización global

Una discretización global sería incorrecta por varias razones:

- usaría información del futuro para etiquetar el pasado;
- contaminaría la evaluación out-of-sample;
- rompería la coherencia con el benchmark;
- haría inválida la comparación temporal entre modelos.

Por eso este proyecto adopta discretización por fold y con thresholds estimados solo en `train`.

---

## 9. Salida esperada de esta capa

La capa `regime_targets_by_split_v1` debería producir una tabla con, como mínimo, las siguientes columnas:

- `split_id`
- `instrument_id`
- `date`
- `dataset_role`
- `future_rv_5d`
- `future_regime_5d`
- `threshold_low`
- `threshold_high`
- `target_version`
- `regime_thresholds_source`

Columnas opcionales razonables:

- `n_classes`
- `labels`
- `quantile_low`
- `quantile_high`

---

## 10. Qué representa `dataset_role`

Cada fila del output debe indicar a qué bloque del fold pertenece:

- `train`
- `validation`
- `test`

Esto es importante porque:

- permite trazabilidad;
- permite revisar qué thresholds se usaron;
- permite evaluar benchmark y ML con la misma semántica;
- evita ambigüedad en análisis posteriores.

---

## 11. Qué debe quedar congelado en configuración

La configuración del régimen debe dejar explícitos, como mínimo:

- nombre del target discreto;
- source target continuo;
- método de discretización;
- labels;
- cuantiles;
- cantidad de clases;
- política de inclusión en bordes;
- `thresholds_source = train_only`.

Nada de esto debe quedar hardcodeado de forma silenciosa dentro de scripts.

---

## 12. Comportamiento esperado ante valores faltantes

Si `future_rv_5d` es nulo para una fila, entonces `future_regime_5d` también debe quedar nulo.

No se deben inventar etiquetas para filas donde el target continuo no existe.

Esto suele ocurrir naturalmente al final de la serie por falta de ventana futura.

---

## 13. Consistencia esperada con benchmark y ML

Tanto benchmark como ML deben compararse contra exactamente la misma definición de régimen.

Eso implica que:

- ambos usan el mismo `split_id`;
- ambos usan los mismos thresholds del train del fold;
- ambos se evalúan contra la misma columna `future_regime_5d`.

Sin esta consistencia, la comparación entre modelos pierde validez.

---

## 14. Restricciones mínimas de validez

Una construcción válida de `regime_targets_by_split_v1` debe cumplir simultáneamente:

1. los thresholds se calculan solo con `train`;
2. `validation` y `test` no participan en la estimación de thresholds;
3. las filas mantienen su `split_id`;
4. las filas mantienen su `dataset_role`;
5. `future_regime_5d` es nulo cuando `future_rv_5d` es nulo;
6. las etiquetas generadas pertenecen únicamente al conjunto permitido;
7. el proceso es reproducible para una misma configuración y mismos splits.

---

## 15. Resumen operativo

La lógica congelada de esta capa es:

1. tomar `future_rv_5d`;
2. tomar los folds walk-forward ya construidos;
3. para cada fold, calcular thresholds solo con `train`;
4. aplicar esos thresholds a `train`, `validation` y `test`;
5. persistir una tabla de targets discretos por split;
6. dejar trazabilidad explícita de thresholds y roles de dataset.

---

## 16. Resumen final

La capa `regime_targets_by_split_v1` queda congelada así:

- depende de `future_rv_5d`;
- depende de los splits walk-forward;
- discretiza por fold;
- calcula thresholds solo con `train`;
- produce `future_regime_5d` con trazabilidad por `split_id`;
- sirve como target discreto oficial para benchmark y ML.

Este documento debe considerarse la fuente de verdad semántica para implementar:

- `src/quant_platform/targets/regime_split_builders.py`
- `scripts/build_regime_targets_by_split.py`
- `scripts/check_regime_targets_by_split_quality.py`
- `tests/unit/test_regime_split_builders.py`
