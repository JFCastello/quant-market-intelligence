# Model Decision v1

## Propósito
Aplicar una regla formal de promoción de modelo sobre los resultados del Paso 24 para decidir, de forma explícita y auditable, si el modelo ML `xgboost_regressor` supera lo suficiente al benchmark `garch_11_student_t`.

## Objetivo del bloque
Construir una capa de decisión reusable que:
- lea métricas ya materializadas de benchmark vs ML;
- compare ambos modelos con una regla de promoción fija;
- produzca un panel de decisión, un resumen por activo y una tabla de razones;
- deje una conclusión clara de promoción o no promoción del modelo ML.

## Inputs reales usados
### Métricas comparativas
- `artifacts/evaluations/model_comparison/{symbol}/*_model_comparison_metrics_v1.parquet`

Estas métricas vienen del Paso 24 y ya contienen, por:
- `instrument_id`
- `symbol`
- `split_id`
- `dataset_role`
- `model_name`
- `metric_name`

los valores de:
- `qlike`
- `rmse`
- `mae`
- `macro_f1`
- `balanced_accuracy`

## Configuración de decisión
Archivo:
- `configs/base.yaml`

Sección:
- `decision`

### Regla congelada en v1
- métrica principal: `qlike`
- guardrail discreto principal: `macro_f1`
- guardrail discreto secundario: `balanced_accuracy`

### Umbrales
- mejora relativa media mínima en `qlike`: `0.10`
- caída máxima permitida en `macro_f1`: `0.00`
- caída máxima permitida en `balanced_accuracy`: `0.00`

## Nota importante sobre calibración
El SDD propone incluir calibración probabilística y `brier_score` cuando existan probabilidades de régimen.

En la versión actual:
- no hay probabilidades de régimen materializadas;
- solo existen etiquetas duras.

Por eso, en `v1`:
- `calibration_status = not_evaluable_yet`

y la política configurada es:
- `mark_as_not_evaluable`

## Builder reusable
Archivo:
- `src/quant_platform/evaluation/model_decision.py`

Función pública:
- `build_model_decision_artifacts(...)`

### Qué produce
Devuelve tres artefactos:

1. `decision_panel_df`
   - benchmark y ML alineados por:
     - `instrument_id`
     - `symbol`
     - `split_id`
     - `dataset_role`
     - `metric_name`

2. `decision_summary_df`
   - una fila por activo con:
     - medias benchmark
     - medias ML
     - mejora relativa media en `qlike`
     - deltas en `macro_f1`
     - deltas en `balanced_accuracy`
     - flags de regla
     - decisión final

3. `decision_reasons_df`
   - tabla larga de razones por activo para trazabilidad.

## Lógica de decisión
### Para métricas lower-is-better
- `qlike`
- `rmse`
- `mae`

la mejora relativa se calcula como:

`(benchmark - ml) / abs(benchmark)`

### Para métricas higher-is-better
- `macro_f1`
- `balanced_accuracy`

la mejora relativa se calcula como:

`(ml - benchmark) / abs(benchmark)`

## Regla final de promoción
`promote_ml` solo ocurre si:

- `qlike_pass = True`
- `macro_f1_pass = True`
- `balanced_accuracy_pass = True`

En cualquier otro caso:
- `decision = do_not_promote_ml`

## Script operativo
Archivo:
- `scripts/apply_model_decision.py`

### Qué hace
1. descubre todos los `model_comparison_metrics`;
2. concatena los archivos;
3. llama al builder reusable;
4. persiste:
   - `decision_panel`
   - `decision_summary`
   - `decision_reasons`

## Outputs persistidos
Raíz:
- `artifacts/evaluations/decision/`

Archivos:
- `all_symbols_decision_panel_v1.parquet`
- `all_symbols_decision_summary_v1.parquet`
- `all_symbols_decision_reasons_v1.parquet`

## Resultado obtenido en v1
El resultado formal quedó en:

- `GLD` -> `do_not_promote_ml`
- `HYG` -> `do_not_promote_ml`
- `SPY` -> `do_not_promote_ml`
- `TLT` -> `do_not_promote_ml`

### Resumen ejecutivo
- en los 4 activos, `qlike_pass = False`;
- por tanto, ML no alcanza el umbral mínimo de mejora relativa media de `10%` en la métrica principal;
- en `SPY`, ML sí iguala o mejora los guardrails discretos, pero falla la condición principal;
- en `GLD`, `HYG` y `TLT`, además de fallar `qlike`, también falla al menos un guardrail discreto.

## Checker de calidad
Archivo:
- `scripts/check_model_decision_quality.py`

### Qué valida
- existencia de `panel`, `summary` y `reasons`;
- columnas obligatorias;
- ausencia de duplicados indebidos;
- una fila por activo en `summary`;
- cuatro reglas por activo en `reasons`;
- consistencia entre flags de `summary` y flags de `reasons`;
- consistencia de símbolos y conteos entre artefactos.

## Tests unitarios
Archivo:
- `tests/unit/test_model_decision.py`

Cobertura actual:
- creación de artefactos esperados;
- caso que promueve ML;
- caso que no promueve ML;
- error por duplicados;
- error por columnas faltantes.

## Comandos operativos del bloque
### Materializar decisión
```bash
python scripts/apply_model_decision.py
```

### Validar calidad
```bash
python scripts/check_model_decision_quality.py
```

### Ejecutar tests unitarios
```bash
pytest tests/unit/test_model_decision.py -q
```

## Estado actual del bloque
Paso 25 resuelto localmente en versión `v1`, con:
- artefactos persistidos;
- checker en PASS;
- tests unitarios en PASS;
- y conclusión formal de no promoción de ML en el universo actual.

## Limitación actual
Todavía no hay:
- probabilidades de régimen;
- `brier_score`;
- ni comparación formal tipo Diebold-Mariano o bootstrap pareado.

Eso queda como extensión analítica posterior.
