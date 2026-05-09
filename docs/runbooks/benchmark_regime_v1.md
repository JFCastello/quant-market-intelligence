# Benchmark regime v1

## Propósito

Transformar el forecast continuo del benchmark GARCH (`yhat_future_rv_5d`) en una predicción discreta de régimen:

- `calm`
- `normal`
- `stress`

usando exactamente los thresholds `train_only` ya materializados en la capa `regime_targets_by_split`.

---

## Lugar del bloque en el proyecto

Este bloque corresponde al **paso 22** del SDD:

- el paso 21 construye el benchmark continuo;
- el paso 22 traduce ese benchmark a régimen discreto usando los mismos umbrales que definen el target discreto del proyecto.

No se recalculan cuantiles nuevos aquí.  
La semántica del régimen se hereda de la capa ya materializada de `regime_targets`.

---

## Inputs

### 1. Benchmark forecasts
Ruta base:

`artifacts/evaluations/benchmark_forecasts/{symbol}/`

Columnas relevantes:

- `date`
- `split_id`
- `dataset_role`
- `yhat_future_rv_5d`

### 2. Regime targets by split
Ruta base:

`artifacts/evaluations/regime_targets/{symbol}/`

Columnas relevantes:

- `date`
- `split_id`
- `dataset_role`
- `future_rv_5d`
- `future_regime_5d`
- `threshold_low`
- `threshold_high`
- `regime_thresholds_source`

---

## Configuración congelada v1

Sección usada en `configs/base.yaml`:

- `benchmark_regime`

Parámetros principales:

- `forecast_value_column = yhat_future_rv_5d`
- `output_regime_column = yhat_future_regime_5d`
- `target_continuous_column = future_rv_5d`
- `target_regime_column = future_regime_5d`
- `threshold_low_column = threshold_low`
- `threshold_high_column = threshold_high`
- `threshold_source_column = regime_thresholds_source`
- `expected_threshold_source = train_only`
- labels:
  - `calm`
  - `normal`
  - `stress`

---

## Metodología

### 1. Fuente de thresholds
Los thresholds no se recalculan desde benchmark forecasts.  
Se extraen desde `regime_targets_by_split`, donde ya quedaron congelados por split bajo política `train_only`.

### 2. Validación de thresholds
Antes del merge, se valida que por cada `split_id`:

- `threshold_low` sea constante;
- `threshold_high` sea constante;
- `regime_thresholds_source` sea constante;
- `threshold_low < threshold_high`.

### 3. Join con targets reales
El builder une benchmark forecasts con regime targets por:

- `split_id`
- `date`
- `dataset_role`

Esto permite dejar en una sola capa:

- forecast continuo benchmark;
- target continuo real;
- target discreto real;
- thresholds aplicados;
- predicción discreta del benchmark.

### 4. Regla de clasificación
La asignación de régimen se hace así:

- `calm` si `yhat_future_rv_5d < threshold_low`
- `normal` si `threshold_low <= yhat_future_rv_5d <= threshold_high`
- `stress` si `yhat_future_rv_5d > threshold_high`

---

## Output

Ruta base:

`artifacts/evaluations/benchmark_regimes/{symbol}/`

Archivo esperado por símbolo:

`{symbol}_{start}_{end}_benchmark_regimes_v1.parquet`

Columnas principales:

- `symbol`
- `date`
- `split_id`
- `dataset_role`
- `model_name`
- `benchmark_version`
- `yhat_future_rv_5d`
- `future_rv_5d`
- `future_regime_5d`
- `threshold_low`
- `threshold_high`
- `regime_thresholds_source`
- `yhat_future_regime_5d`

---

## Script operativo

Construcción completa:

```bash
python scripts/build_benchmark_regime_predictions.py
```

Salida esperada:

- un parquet por símbolo en `benchmark_regimes/`
- cobertura de todos los `split_id`
- filas en `validation` y `test`
- targets reales y thresholds unidos al output

---

## Checker de calidad

Validación:

```bash
python scripts/check_benchmark_regime_quality.py
```

El checker verifica:

- existencia de archivos;
- columnas requeridas;
- ausencia de duplicados por `(split_id, date)`;
- roles válidos;
- thresholds válidos;
- `regime_thresholds_source = train_only`;
- labels esperados;
- constancia de thresholds y metadatos dentro de cada split.

---

## Tests unitarios

Archivo:

`tests/unit/test_benchmark_regime.py`

Ejecución:

```bash
pytest tests/unit/test_benchmark_regime.py -q
```

Cobertura actual:

- asignación correcta de labels;
- extracción de thresholds constantes por split;
- error cuando thresholds varían dentro del split;
- merge correcto con targets reales;
- validación de forecasts finitos.

---

## Archivos involucrados

### Configuración
- `configs/base.yaml`

### Código de dominio
- `src/quant_platform/models/benchmark_regime.py`
- `src/quant_platform/models/__init__.py`

### Scripts
- `scripts/build_benchmark_regime_predictions.py`
- `scripts/check_benchmark_regime_quality.py`

### Tests
- `tests/unit/test_benchmark_regime.py`

### Documentación
- `docs/runbooks/benchmark_regime_v1.md`

---

## Criterio de done de 22

El paso 22 se considera cerrado cuando:

- el benchmark continuo se transforma a régimen discreto sin recalcular thresholds;
- la nueva capa queda materializada por símbolo;
- los outputs pasan checker de calidad;
- los tests unitarios pasan;
- el bloque queda documentado en runbook.

---

## Siguiente paso natural

El siguiente bloque lógico ya es pasar a evaluación formal / comparación,
o al siguiente paso definido en el SDD para explotar estas salidas en métricas y reporting.
