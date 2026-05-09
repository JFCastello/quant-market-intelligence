# Model Comparison v1

## Propósito
Materializar y validar la comparación formal entre el benchmark clásico `garch_11_student_t` y el modelo `xgboost_regressor` sobre el mismo problema de forecast de volatilidad futura a 5 días (`future_rv_5d`) y su discretización a régimen (`calm`, `normal`, `stress`).

## Objetivo del bloque
Construir una capa de evaluación reusable que:
- compare benchmark vs ML sobre exactamente las mismas filas out-of-sample;
- produzca métricas continuas y discretas por `split_id`, `dataset_role` y `model_name`;
- genere matrices de confusión persistidas;
- deje un panel fila-a-fila reutilizable para análisis posterior y gráficos por fold.

## Inputs reales usados
### Benchmark
- `artifacts/evaluations/benchmark_regimes/{symbol}/*_benchmark_regimes_v1.parquet`

Se usa `benchmark_regimes` y no `benchmark_forecasts`, porque aquí ya vienen:
- `future_rv_5d`
- `future_regime_5d`
- `yhat_future_rv_5d`
- `yhat_future_regime_5d`
- `threshold_low`
- `threshold_high`
- `regime_thresholds_source`

### ML
- `artifacts/evaluations/ml_forecasts/{symbol}/*_xgboost_regressor_v1.parquet`

### Régimen por split
- `artifacts/evaluations/regime_targets/{symbol}/*_regime_by_split_v1.parquet`

El lado ML se construye como:
- `ml_forecasts`
- más `regime_targets`

porque `ml_forecasts` trae el forecast continuo, pero no trae ni thresholds ni régimen predicho.

## Clave de alineación OOS
La clave efectiva de alineación entre benchmark y ML es:
- `date`
- `split_id`
- `dataset_role`

La evaluación se corre por archivo de símbolo, por lo que `instrument_id` se infiere desde `regime_targets` y `symbol` desde los archivos de forecast.

## Lógica del builder
Archivo principal:
- `src/quant_platform/evaluation/model_comparison.py`

Función pública:
- `build_model_comparison_artifacts(...)`

### Qué produce
Devuelve tres artefactos:

1. `evaluation_panel_df`
   - panel fila-a-fila con benchmark y ML ya alineados;
   - incluye errores continuos y término `qlike_term`.

2. `metrics_df`
   - tabla larga de métricas por:
     - `instrument_id`
     - `symbol`
     - `split_id`
     - `dataset_role`
     - `model_name`

3. `confusion_df`
   - matriz de confusión en formato largo por grupo.

## Métricas implementadas
### Continuas
- `qlike`
- `rmse`
- `mae`

### Discretas
- `macro_f1`
- `balanced_accuracy`

## Nota importante sobre QLIKE
El target almacenado por el proyecto es volatilidad (`future_rv_5d`), no varianza.

Por eso, el `QLIKE` se calcula en dominio de varianza:
- `y_true_var = future_rv_5d ** 2`
- `y_pred_var = yhat_future_rv_5d ** 2`

Esto evita mezclar una métrica propia de varianza con una serie persistida como volatilidad.

## Mapeo de régimen para ML
Como el bloque actual materializa un `XGBoost Regressor` y no una cabeza probabilística de clasificación, el régimen predicho para ML se obtiene aplicando los thresholds del split a `yhat_future_rv_5d`:
- `< threshold_low` -> `calm`
- `>= threshold_low` y `<= threshold_high` -> `normal`
- `> threshold_high` -> `stress`

## Brier score
Queda explícitamente desactivado en esta versión:
- `compute_brier: false`

Razón:
- no existen todavía probabilidades de régimen materializadas ni para benchmark ni para ML;
- solo existen etiquetas duras.

## Script operativo
Archivo:
- `scripts/evaluate_models.py`

### Qué hace
Por cada símbolo:
1. lee `benchmark_regimes`;
2. lee `ml_forecasts`;
3. lee `regime_targets`;
4. llama al builder reusable;
5. persiste:
   - `panel`
   - `metrics`
   - `confusion`

## Outputs persistidos
### Panels y métricas
Raíz:
- `artifacts/evaluations/model_comparison/{symbol}/`

Archivos:
- `*_model_comparison_panel_v1.parquet`
- `*_model_comparison_metrics_v1.parquet`

### Matrices de confusión
Raíz:
- `artifacts/evaluations/model_comparison/confusion_matrices/{symbol}/`

Archivo:
- `*_model_comparison_confusion_v1.parquet`

## Checker de calidad
Archivo:
- `scripts/check_model_comparison_quality.py`

### Qué valida
- columnas obligatorias;
- ausencia de duplicados indebidos;
- presencia de ambos modelos;
- roles OOS válidos;
- labels válidos;
- cinco métricas por grupo;
- nueve celdas por matriz de confusión;
- consistencia cruzada entre:
  - `panel`
  - `metrics`
  - `confusion`

## Tests unitarios
Archivo:
- `tests/unit/test_model_comparison.py`

Cobertura actual:
- construcción de artefactos esperados;
- mapeo de régimen para ML;
- error si no alinean las filas OOS;
- error si faltan columnas obligatorias.

## Comandos operativos del bloque
### Materializar comparación
```bash
python scripts/evaluate_models.py
```

### Validar calidad de la capa
```bash
python scripts/check_model_comparison_quality.py
```

### Ejecutar tests unitarios del builder
```bash
pytest tests/unit/test_model_comparison.py -q
```

## Estado actual del bloque
Paso 24 funcionalmente resuelto en versión `v1` para:
- `SPY`
- `TLT`
- `GLD`
- `HYG`

con artefactos persistidos, checker en PASS y tests unitarios en PASS.

## Limitación actual
Todavía no hay:
- probabilidades de régimen,
- Brier score,
- ni comparación formal tipo Diebold-Mariano / bootstrap pareado.

Eso queda como extensión natural del cierre analítico y del paso siguiente de decisión de promoción del modelo.
