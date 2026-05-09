# XGBoost regressor v1

## Propósito

Entrenar el primer modelo ML del proyecto para predecir el target continuo principal:

- `future_rv_5d`

usando la capa validada de `features_context`, los splits walk-forward ya congelados y un `XGBoost Regressor`.

---

## Lugar del bloque en el proyecto

Este bloque corresponde al **paso 23** del SDD:

- el benchmark clásico ya quedó construido en pasos anteriores;
- aquí se construye el primer pipeline ML serio para forecast continuo;
- la comparación formal benchmark vs ML queda habilitada para el paso siguiente.

En esta versión se implementa solo el **regressor**, no el classifier.

---

## Inputs

### 1. Features de entrada
Ruta base:

`data/features_context/{symbol}/`

Columnas relevantes:

- `instrument_id`
- `date`
- `feature_version`
- features base
- features de contexto `ctx_*`

### 2. Targets continuos
Ruta base:

`data/targets/{symbol}/`

Columna principal:

- `future_rv_5d`

### 3. Splits walk-forward
Ruta base:

`artifacts/evaluations/splits/{symbol}/`

Los parquets de splits están en formato interval-level:
una fila por `split_id`, con ventanas:

- `train_start`
- `train_end`
- `validation_start`
- `validation_end`
- `test_start`
- `test_end`

El trainer expande internamente estas ventanas a daily-level.

---

## Configuración congelada v1

Sección usada en `configs/base.yaml`:

- `ml_regressor`

Parámetros principales:

- `model_name = xgboost_regressor`
- `model_version = v1`
- `feature_source = features_context`
- `target_column = future_rv_5d`
- `score_roles = validation, test`
- `objective = reg:squarederror`
- `eval_metric = rmse`

Hiperparámetros base:

- `n_estimators = 500`
- `learning_rate = 0.05`
- `max_depth = 4`
- `min_child_weight = 5`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_alpha = 0.0`
- `reg_lambda = 1.0`
- `gamma = 0.0`
- `random_state = 42`
- `tree_method = hist`
- `n_jobs = -1`
- `early_stopping_rounds = 50`

---

## Metodología

### 1. Construcción del panel de modelado
Se unen:

- `features_context`
- `targets`

por:

- `instrument_id`
- `date`

El panel resultante contiene:

- columnas metadata
- target continuo
- lista explícita de `feature_columns`

### 2. Manejo de NaN
No se hace imputación manual en esta versión.

Se aprovecha la capacidad nativa de XGBoost para manejar valores faltantes, lo cual permite conservar las filas warm-up sin introducir reglas adicionales.

### 3. Expansión de splits
Como los splits vienen en formato interval-level, el trainer los expande a daily-level antes de asignar:

- `train`
- `validation`
- `test`

### 4. Entrenamiento por split
Para cada `split_id`:

- se entrena con `train`;
- se usa `validation` para early stopping;
- se generan predicciones out-of-sample sobre `validation` y `test`.

### 5. Persistencia
Se persisten:

- forecasts por símbolo
- modelos XGBoost por split
- lista de feature columns por split
- metadata por split

---

## Outputs

### Forecasts ML
Ruta base:

`artifacts/evaluations/ml_forecasts/{symbol}/`

Archivo esperado por símbolo:

`{symbol}_{start}_{end}_xgboost_regressor_v1.parquet`

Columnas principales:

- `symbol`
- `date`
- `split_id`
- `dataset_role`
- `model_name`
- `model_version`
- `target_column`
- `yhat_future_rv_5d`
- `future_rv_5d`
- `train_start_date`
- `train_end_date`
- `n_train`
- `n_validation`
- `n_score`
- `feature_count`
- `best_iteration`
- `best_score`

### Modelos serializados
Ruta base:

`artifacts/models/xgboost_regressor/{symbol}/`

Por cada split se guardan:

- modelo `.json`
- `feature_columns` en `.json`
- metadata en `.json`

---

## Script operativo

Construcción completa:

```bash
python scripts/build_xgboost_regressor_forecasts.py
```

Salida esperada:

- un parquet de forecasts por símbolo;
- modelos serializados por split;
- cobertura de `validation` y `test` en todos los folds.

---

## Checker de calidad

Validación:

```bash
python scripts/check_xgboost_regressor_quality.py
```

El checker verifica:

- existencia de archivos;
- columnas requeridas;
- ausencia de duplicados por `(split_id, date)`;
- roles válidos;
- predicciones y targets positivos;
- consistencia de metadata por split;
- constancia de `feature_count`, `best_iteration`, `best_score`, etc. dentro de cada split.

---

## Tests unitarios

Archivo:

`tests/unit/test_xgboost_regressor.py`

Ejecución:

```bash
pytest tests/unit/test_xgboost_regressor.py -q
```

Cobertura actual:

- construcción del panel features + target;
- generación de forecasts OOS por split;
- metadata del modelo entrenado;
- manejo de missing values;
- error cuando validation no contiene targets utilizables.

---

## Archivos involucrados

### Configuración
- `configs/base.yaml`

### Código de dominio
- `src/quant_platform/models/xgboost_regressor.py`
- `src/quant_platform/models/__init__.py`

### Scripts
- `scripts/build_xgboost_regressor_forecasts.py`
- `scripts/check_xgboost_regressor_quality.py`

### Tests
- `tests/unit/test_xgboost_regressor.py`

### Documentación
- `docs/runbooks/xgboost_regressor_v1.md`

---

## Criterio de done de 23

El paso 23 se considera cerrado cuando:

- el regressor entrena correctamente por split;
- produce forecasts OOS sobre `validation` y `test`;
- la nueva capa `ml_forecasts` queda materializada;
- los modelos quedan persistidos;
- los outputs pasan checker de calidad;
- los tests unitarios pasan;
- el bloque queda documentado en runbook.

---

## Siguiente paso natural

El siguiente bloque lógico ya es usar benchmark + ML para comparación formal y evaluación:
métricas, tablas por split, comparación por activo y decisión real sobre si ML aporta valor.
