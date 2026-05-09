# Services Layer v1

## Propósito
Crear una capa de servicios Python puros, reusable y desacoplada de Streamlit, para que el dashboard local y una futura API consuman la misma lógica de acceso y armado de datos.

## Objetivo del bloque
Construir una service layer que:
- centralice carga de artefactos y datasets del proyecto;
- exponga funciones puras para `overview`, `model_comparison` y `structural_changes`;
- no dependa de Streamlit;
- devuelva objetos pandas y contratos estables para reutilización futura.

## Configuración congelada en v1
Archivo:
- `configs/base.yaml`

Sección:
- `services_layer`

### Inputs declarados
- `data/features`
- `data/targets`
- `artifacts/evaluations/benchmark_regimes`
- `artifacts/evaluations/ml_forecasts`
- `artifacts/evaluations/model_comparison`
- `artifacts/evaluations/decision`
- `artifacts/evaluations/structural_breaks`

### Defaults
- `overview_symbol: SPY`
- `latest_break_events_limit: 10`
- `preferred_decision_summary_file: all_symbols_decision_summary_v1.parquet`
- `preferred_decision_reasons_file: all_symbols_decision_reasons_v1.parquet`

### Contratos
- `return_pandas_objects: true`
- `service_functions_must_not_import_streamlit: true`
- `one_source_of_truth: true`

## Artefactos reales consumidos
### Globales
- `artifacts/evaluations/decision/all_symbols_decision_summary_v1.parquet`
- `artifacts/evaluations/decision/all_symbols_decision_reasons_v1.parquet`

### Por símbolo
- `data/features/{symbol}/*.parquet`
- `data/targets/{symbol}/*.parquet`
- `artifacts/evaluations/benchmark_regimes/{symbol}/*.parquet`
- `artifacts/evaluations/ml_forecasts/{symbol}/*.parquet`
- `artifacts/evaluations/model_comparison/{symbol}/*_model_comparison_metrics_*.parquet`
- `artifacts/evaluations/model_comparison/{symbol}/*_model_comparison_panel_*.parquet`
- `artifacts/evaluations/structural_breaks/{symbol}/*_structural_break_signal_*.parquet`
- `artifacts/evaluations/structural_breaks/{symbol}/*_structural_break_events_*.parquet`

## Hallazgos importantes de contrato
1. `decision_summary` y `decision_reasons` ya existen como artefactos globales reutilizables.
2. `model_comparison_metrics` es el input correcto para comparación agregada, y `model_comparison_panel` sirve para detalle fila a fila.
3. `structural_break_signal` y `structural_break_events` bastan para la vista de cambios estructurales.
4. En `overview`, la última fecha de `features` no necesariamente coincide con la última fecha con `future_rv_5d` válida en `targets`, así que la service layer debe manejar explícitamente ese desfase.

## Arquitectura implementada
### 1. Loaders reutilizables
Archivo:
- `src/quant_platform/services/artifact_loaders.py`

### Qué expone
- lista de símbolos disponibles;
- carga de `features`, `targets`, `benchmark_regimes`, `ml_forecasts`;
- carga de `model_comparison_metrics`, `model_comparison_panel`;
- carga de `structural_break_signal`, `structural_break_events`;
- carga de `decision_summary`, `decision_reasons`.

## Servicio de overview
Archivo:
- `src/quant_platform/services/overview_service.py`

### Qué expone
- `get_symbol_overview_timeseries(symbol)`
- `get_symbol_decision_summary(symbol)`
- `get_symbol_recent_break_events(symbol, limit=None)`
- `build_symbol_overview_snapshot(symbol)`
- `get_symbol_overview_bundle(symbol)`

### Contrato principal
El snapshot sintetiza:
- símbolo e instrumento;
- última fecha con features;
- última fecha con target válido;
- últimos valores relevantes de mercado;
- decisión del modelo;
- últimos eventos de cambio estructural.

## Servicio de model comparison
Archivo:
- `src/quant_platform/services/model_comparison_service.py`

### Qué expone
- `get_symbol_model_comparison_metrics(symbol)`
- `get_symbol_model_comparison_panel(symbol)`
- `get_symbol_model_comparison_pivot(symbol)`
- `build_symbol_model_comparison_summary(symbol)`
- `get_symbol_model_comparison_bundle(symbol)`

### Contrato principal
Permite:
- ver métricas largas;
- pivotear benchmark vs ML;
- obtener resumen agregado por métrica;
- exponer panel fila a fila para drill-down.

## Servicio de structural changes
Archivo:
- `src/quant_platform/services/structural_changes_service.py`

### Qué expone
- `get_symbol_structural_break_signal(symbol)`
- `get_symbol_structural_break_events(symbol)`
- `get_symbol_recent_structural_break_events(symbol, limit=None)`
- `build_symbol_structural_break_summary(symbol)`
- `get_symbol_structural_changes_bundle(symbol)`

### Contrato principal
Permite:
- cargar la señal preparada;
- cargar eventos detectados;
- generar un resumen corto del detector;
- y devolver bundle reusable para UI o API.

## Exportación del paquete
Archivo:
- `src/quant_platform/services/__init__.py`

Se consolidaron exports de:
- loaders;
- overview service;
- model comparison service;
- structural changes service;
- settings.

## Smoke checker
Archivo:
- `scripts/check_services_layer_smoke.py`

### Qué valida
- descubrimiento de símbolos;
- consistencia básica de overview bundle;
- consistencia básica de model comparison bundle;
- consistencia básica de structural changes bundle;
- operatividad completa de la capa sobre los 4 activos.

## Tests unitarios
Archivo:
- `tests/unit/test_services_layer.py`

Cobertura actual:
- universo disponible;
- carga de `decision_summary`;
- snapshot y bundle de overview;
- resumen y bundle de model comparison;
- resumen y bundle de structural changes.

## Resultado validado en v1
### Símbolos disponibles
- `GLD`
- `HYG`
- `SPY`
- `TLT`

### Smoke checker
Salida validada:
- `overview_rows=2074/2075`
- `comparison_rows=5`
- `break_events` coherentes por símbolo
- `SERVICES LAYER SMOKE CHECKS: PASS`

### Tests unitarios
- `8 passed`

## Comandos operativos del bloque
### Smoke check
```bash
python scripts/check_services_layer_smoke.py
```

### Tests unitarios
```bash
pytest tests/unit/test_services_layer.py -q
```

## Estado actual del bloque
Paso 27 resuelto localmente en versión `v1`, con:
- loaders reutilizables;
- servicios de overview;
- servicios de model comparison;
- servicios de structural changes;
- smoke checker en PASS;
- tests unitarios en PASS.

## Limitación actual
Todavía no existe:
- integración visual directa con Streamlit;
- capa HTTP / FastAPI;
- caching o memoization de servicios;
- contratos tipados más estrictos mediante dataclasses o pydantic.

Eso queda para los pasos siguientes.
