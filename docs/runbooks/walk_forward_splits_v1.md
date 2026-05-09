# Walk-Forward Splits v1

## 1. Propósito

Este documento congela la semántica del bloque de **walk-forward splits v1** del proyecto **Quant Market Intelligence**.

Su función es dejar explícito:

- qué tipo de partición temporal usa el proyecto;
- cómo se separan `train`, `validation` y `test`;
- por qué se usa una ventana expansiva;
- por qué el bloque de test debe permanecer estrictamente out-of-sample;
- cómo se conecta este esquema con `future_rv_5d` y `future_regime_5d`;
- qué propiedades debe cumplir cualquier implementación del builder de splits.

Este documento es la fuente de verdad semántica del paso 20 del SSD.

---

## 2. Alcance de esta versión

### Sí incluye

- definición formal del esquema walk-forward;
- separación temporal entre `train`, `validation` y `test`;
- política de ventana expansiva;
- política de avance entre folds;
- criterios mínimos de validez de cada fold;
- relación entre splits y cálculo correcto de thresholds de régimen;
- requisitos del builder reusable de splits.

### No incluye todavía

- entrenamiento de benchmark;
- entrenamiento de XGBoost;
- tuning de hiperparámetros;
- scoring final por fold;
- evaluación comparativa de modelos.

Eso corresponde a los pasos posteriores del pipeline.

---

## 3. Qué problema resuelve este bloque

En series temporales financieras no se puede usar validación aleatoria estándar.

La razón es simple: mezclar observaciones del futuro dentro de train, validation o selección de umbrales produce resultados artificialmente optimistas y rompe la interpretación out-of-sample.

Por eso el proyecto adopta un esquema **walk-forward temporal estricto**.

---

## 4. Esquema adoptado

El método congelado para `v1` es:

```text
walk_forward_expanding
```

### Qué significa

Cada fold contiene tres bloques ordenados en el tiempo:

1. `train`
2. `validation`
3. `test`

y debe cumplirse siempre:

```text
train_end < validation_start < validation_end < test_start <= test_end
```

Además, el bloque de entrenamiento **crece** de fold en fold.

---

## 5. Estructura conceptual de cada fold

Cada fold debe tener la siguiente forma:

```text
|--------- train ---------|--- validation ---|--- test ---|
```

### Interpretación

- `train`: usado para ajustar modelos y calcular thresholds de régimen.
- `validation`: usado para decisiones intermedias, selección de configuraciones o comparación interna.
- `test`: usado únicamente como bloque final out-of-sample.

---

## 6. Política de ventana expansiva

La política adoptada es:

```text
expanding train window
```

### Qué significa

El conjunto de entrenamiento del fold siguiente contiene todo el entrenamiento anterior más un bloque temporal adicional.

### Ejemplo conceptual

- Fold 1: train = primer tramo histórico
- Fold 2: train = train del fold 1 + tramo adicional
- Fold 3: train = train del fold 2 + tramo adicional

### Por qué se elige esta política

Porque en datos financieros diarios suele ser razonable aprovechar todo el pasado disponible, manteniendo siempre la dirección temporal correcta.

---

## 7. Política de validation y test

En `v1`, ambos bloques existen de forma explícita.

### `validation`
Existe para:

- selección de decisiones intermedias;
- tuning futuro;
- chequeos de promoción del modelo;
- fijación indirecta de configuraciones sin tocar test.

### `test`
Debe permanecer congelado y no puede participar en:

- cálculo de thresholds de régimen;
- tuning;
- selección de modelo;
- normalización aprendida;
- decisión final de promoción.

---

## 8. Relación con el target continuo

El target continuo principal del proyecto es:

```text
future_rv_5d
```

Cada fold debe construirse sobre filas que ya tengan disponible esta etiqueta según la definición del paso 18.

### Implicación
Los splits no redefinen el target; solo organizan temporalmente las filas válidas para entrenamiento y evaluación.

---

## 9. Relación con el target discreto de régimen

El target discreto derivado es:

```text
future_regime_5d
```

pero sus thresholds deben calcularse exclusivamente con el bloque `train` de cada fold.

### Regla congelada

```text
regime_thresholds_source = train_only
```

### Consecuencia
Cada fold puede tener thresholds distintos, porque cada bloque de entrenamiento puede tener una distribución distinta de `future_rv_5d`.

Esto no es un problema; es precisamente lo correcto desde el punto de vista estadístico.

---

## 10. Política de avance entre folds

El esquema v1 avanza con un stride temporal fijo.

### Concepto
Después de construir un fold, el sistema mueve la frontera temporal hacia adelante una cantidad predefinida y genera el fold siguiente.

### Beneficio
Esto produce una familia de evaluaciones comparables, reproducibles y cronológicamente consistentes.

---

## 11. Parámetros esperados en configuración

La configuración del bloque `splits` debe congelar, como mínimo:

- `split_version`
- `method`
- `date_column`
- `group_key`
- `train_years`
- `validation_months`
- `test_months`
- `step_months`
- `min_train_observations`
- `allow_partial_last_fold`
- `require_validation`
- `regime_thresholds_source`
- `output_format`

Estos parámetros deben vivir en el YAML y no en valores hardcodeados dentro de scripts.

---

## 12. Restricciones mínimas de un fold válido

Un fold válido debe cumplir simultáneamente:

1. todas las observaciones de `train` ocurren antes que `validation`;
2. todas las observaciones de `validation` ocurren antes que `test`;
3. no existen filas repetidas dentro del mismo fold;
4. el bloque `train` cumple un tamaño mínimo;
5. `validation` y `test` no están vacíos si la configuración los requiere;
6. las fronteras temporales son trazables y explícitas;
7. la construcción del fold no usa información futura para redefinir el pasado.

---

## 13. Qué debe devolver el builder reusable

El builder de splits debería poder devolver una representación tabular por fold, con metadatos como:

- `split_version`
- `split_id`
- `instrument_id`
- `train_start`
- `train_end`
- `validation_start`
- `validation_end`
- `test_start`
- `test_end`
- `train_rows`
- `validation_rows`
- `test_rows`
- `regime_thresholds_source`

Esto permite trazabilidad, debugging y validación automática.

---

## 14. Por qué el test debe permanecer intacto

El test no es un bloque decorativo.

Es el único lugar donde el sistema puede estimar, de forma creíble, cómo se habría comportado fuera de muestra.

Si el test participa en:

- selección de thresholds,
- tuning,
- elección de modelo,
- o normalización aprendida,

entonces deja de ser test y la comparación con benchmark pierde validez.

---

## 15. Por qué este paso es crítico antes del benchmark y ML

El SDD deja claro que antes de entrenar GARCH y XGBoost debe existir una base seria de evaluación temporal. El roadmap incluso ubica los walk-forward splits antes del benchmark GARCH y del pipeline formal de evaluación.

La razón es estructural:

- sin splits rigurosos no hay evaluación seria;
- sin evaluación seria no puede saberse si ML aporta valor;
- sin eso, el producto pierde su pregunta central.

---

## 16. Errores que este diseño busca evitar

Este bloque existe precisamente para evitar errores como:

- mezclar fechas futuras dentro de train;
- usar validación aleatoria;
- fijar thresholds de régimen con todo el histórico;
- usar test para decidir hiperparámetros;
- comparar benchmark y ML con folds distintos;
- generar evaluación irreproducible entre corridas.

---

## 17. Resumen operativo

El bloque `walk_forward_splits_v1` debe funcionar así:

1. tomar una serie temporal ordenada por `instrument_id` y `date`;
2. definir una ventana inicial de `train`;
3. anexar un bloque de `validation`;
4. anexar un bloque de `test`;
5. registrar las fronteras temporales exactas;
6. avanzar el reloj una cantidad fija;
7. repetir hasta agotar el histórico disponible;
8. garantizar que los thresholds de régimen se calculen solo con train dentro de cada fold.

---

## 18. Resumen final

El bloque de splits v1 queda congelado así:

- método: `walk_forward_expanding`
- bloques: `train`, `validation`, `test`
- train: expansivo
- test: estrictamente out-of-sample
- thresholds de régimen: `train_only`
- configuración: centralizada en YAML
- salida: tabla de folds trazable y validable

Este documento debe considerarse la fuente de verdad semántica para implementar:

- `src/quant_platform/evaluation/split_builders.py`
- `scripts/build_walk_forward_splits.py`
- `scripts/check_splits_quality.py`
- `tests/unit/test_split_builders.py`
