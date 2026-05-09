# GARCH benchmark v1

## Propósito

Materializar el benchmark clásico del proyecto para forecast de volatilidad a 5 días:

- modelo: `GARCH(1,1)`
- innovaciones: `Student-t`
- horizonte: `5d`
- target de salida: `future_rv_5d`

Este benchmark existe para servir como línea base seria frente al modelo ML posterior.

---

## Lugar del bloque en el proyecto

Este bloque corresponde al **paso 21** del SDD:

- primero se construye el benchmark continuo de volatilidad;
- después, en el paso 22, ese forecast se mapea a regímenes usando umbrales train-only;
- más adelante se compara formalmente contra ML.

---

## Inputs

### 1. Datos normalizados
Ruta base:

`data/normalized/{symbol}/`

Contienen la serie diaria de precios usada para construir retornos del benchmark.

### 2. Splits walk-forward
Ruta base:

`artifacts/evaluations/splits/{symbol}/`

Los parquets de splits están en formato **interval-level**:
una fila por `split_id`, con ventanas explícitas:

- `train_start`
- `train_end`
- `validation_start`
- `validation_end`
- `test_start`
- `test_end`

El builder del benchmark expande internamente esas ventanas a un esquema daily-level con `dataset_role`.

### 3. Configuración
Sección usada en `configs/base.yaml`:

- `benchmark`

---

## Configuración congelada v1

- `benchmark_name`: `garch_11_student_t`
- `benchmark_version`: `v1`
- `mean_model`: `zero`
- `vol_model`: `garch`
- `p=1, o=0, q=1`
- `distribution`: `studentst`
- `input_price_column`: `close`
- `return_type`: `log`
- `return_column_name`: `ret_1d`
- `fit_scale`: `100.0`
- `annualization_factor`: `252`
- `forecast_horizon_days`: `5`
- `output_target_name`: `future_rv_5d`
- `min_train_points`: `252`
- `score_roles`: `validation`, `test`

---

## Metodología

## 1. Serie base
A partir de `close`, se calculan retornos diarios logarítmicos:

- `ret_1d = log(close_t / close_{t-1}) * fit_scale`

El `fit_scale=100` se usa para mejorar estabilidad numérica en el ajuste.

## 2. Ajuste por split
Para cada `split_id`:

- el modelo se ajusta **solo con train**;
- los parámetros quedan congelados dentro del split.

No se reentrena en validation ni test.

## 3. Scoring secuencial out-of-sample
Para cada fecha en `validation` y `test`:

- se usa el retorno ya observado para actualizar el estado condicional;
- se genera una trayectoria forecast de varianzas diarias a horizonte fijo;
- esa trayectoria se agrega y se transforma a volatilidad anualizada compatible con `future_rv_5d`.

## 4. Forecast final
La salida principal es:

- `yhat_future_rv_5d`

Este benchmark todavía **no** produce régimen discreto.
Eso queda para el paso 22.

---

## Outputs

Ruta base:

`artifacts/evaluations/benchmark_forecasts/{symbol}/`

Archivo esperado por símbolo:

`{symbol}_{start}_{end}_garch_11_student_t_v1.parquet`

Columnas principales:

- `symbol`
- `date`
- `split_id`
- `dataset_role`
- `model_name`
- `benchmark_version`
- `forecast_horizon_days`
- `output_target_name`
- `yhat_future_rv_5d`
- `train_start_date`
- `train_end_date`
- `n_train`
- `fit_status`
- `omega`
- `alpha_1`
- `beta_1`
- `nu`

---

## Script operativo

Construcción completa:

```bash
python scripts/build_garch_benchmark_forecasts.py