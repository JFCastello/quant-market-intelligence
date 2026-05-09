# Structural Breaks v1

## Propósito
Detectar cambios estructurales en cada activo del universo usando una señal conjunta de retorno diario logarítmico y volatilidad futura realizada a 5 días.

## Objetivo del bloque
Construir una capa reusable que:
- prepare una señal conjunta por activo;
- aplique detección de quiebres estructurales con `ruptures`;
- materialice la señal preparada y los eventos detectados;
- deje eventos listos para inspección, validación visual y futuro uso en dashboard.

## Configuración congelada en v1
Archivo:
- `configs/base.yaml`

Sección:
- `structural_breaks`

### Señales usadas
- `log_ret_1d`
- `future_rv_5d`

### Método
- librería: `ruptures`
- algoritmo: `pelt`
- costo: `rbf`

### Preprocesamiento
- `dropna: true`
- `min_required_rows: 252`
- `standardize_joint_signal: true`

### Segmentación
- `penalty: 8.0`
- `min_size: 20`
- `jump: 1`

## Inputs reales usados
### Features
- `data/features/{symbol}/*_features_v1.parquet`

### Targets
- `data/targets/{symbol}/*_targets_v1.parquet`

## Hallazgo importante de contrato
El retorno disponible en `features` no era `ret_1d`, sino `log_ret_1d`.

Por eso la configuración final del detector quedó basada en:
- `log_ret_1d`
- `future_rv_5d`

## Builder reusable
Archivo:
- `src/quant_platform/evaluation/structural_breaks.py`

Función pública:
- `build_structural_break_artifacts(...)`

### Qué produce
Devuelve dos artefactos:

1. `signal_df`
   - señal conjunta ya preparada y ordenada por fecha;
   - incluye columnas estandarizadas:
     - `z_log_ret_1d`
     - `z_future_rv_5d`

2. `events_df`
   - eventos de quiebre estructural detectados;
   - incluye:
     - `break_date`
     - segmentos previo y siguiente
     - configuración de método
     - metadatos de detección

## Lógica del builder
1. valida columnas base;
2. hace join `features + targets` por:
   - `instrument_id`
   - `date`
3. elimina filas con `NaN` en la señal configurada;
4. exige un mínimo de filas útiles;
5. estandariza la señal conjunta;
6. corre `ruptures.Pelt(model="rbf")`;
7. transforma breakpoints en eventos interpretables por fecha y segmentos.

## Script operativo
Archivo:
- `scripts/detect_structural_breaks.py`

### Qué hace
Por cada símbolo:
1. lee `features`;
2. lee `targets`;
3. llama al builder reusable;
4. persiste:
   - `signal`
   - `events`

## Outputs persistidos
Raíz:
- `artifacts/evaluations/structural_breaks/{symbol}/`

Archivos:
- `*_structural_break_signal_v1.parquet`
- `*_structural_break_events_v1.parquet`

## Resultado obtenido en v1
### Número de eventos detectados
- `GLD` -> 7
- `HYG` -> 6
- `SPY` -> 9
- `TLT` -> 4

### Ejemplos de fechas detectadas
- `SPY`: 2020-02-14, 2020-06-30, 2023-03-23
- `HYG`: 2020-02-20, 2022-02-02, 2023-11-15
- `TLT`: 2019-07-31, 2021-10-29, 2023-12-15

## Interpretación operativa
La salida actual parece razonable como primera versión:
- no se ve vacía;
- no se ve absurdamente ruidosa;
- y genera cortes amplios que pueden servir para visualización e interpretación.

El `penalty=8.0` se deja congelado en `v1` y podrá recalibrarse después si se quiere una segmentación más conservadora o más sensible.

## Checker de calidad
Archivo:
- `scripts/check_structural_breaks_quality.py`

### Qué valida
- existencia de `signal` y `events`;
- columnas obligatorias;
- ausencia de duplicados indebidos;
- fechas ordenadas;
- consistencia de segmentos;
- coherencia entre `signal_df` y `events_df`.

## Tests unitarios
Archivo:
- `tests/unit/test_structural_breaks.py`

Cobertura actual:
- creación de artefactos esperados;
- detección de quiebre en señal sintética;
- error por columna faltante;
- error por filas insuficientes;
- error por `instrument_id` inconsistente.

## Comandos operativos del bloque
### Materializar cambios estructurales
```bash
python scripts/detect_structural_breaks.py
```

### Validar calidad
```bash
python scripts/check_structural_breaks_quality.py
```

### Ejecutar tests unitarios
```bash
pytest tests/unit/test_structural_breaks.py -q
```

## Estado actual del bloque
Paso 26 resuelto localmente en versión `v1`, con:
- señal y eventos persistidos para los 4 activos;
- checker en PASS;
- tests unitarios en PASS.

## Limitación actual
Todavía no existe:
- scoring formal de calidad del detector;
- comparación entre distintos penalties/modelos de costo;
- ni integración visual en dashboard.

Eso queda para la iteración siguiente.
