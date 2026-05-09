# Bitácora ejecutiva — Paso 27: services layer

## Propósito
Crear una capa de servicios Python puros que desacople la lógica reusable del dashboard y deje una base común para consumo posterior desde Streamlit o API.

## Resultado útil logrado
Quedó implementada una nueva service layer en:

- `src/quant_platform/services/artifact_loaders.py`
- `src/quant_platform/services/overview_service.py`
- `src/quant_platform/services/model_comparison_service.py`
- `src/quant_platform/services/structural_changes_service.py`

Además:
- el smoke checker quedó en PASS;
- y el test unitario específico de la capa quedó en PASS.

---

## Secuencia efectiva ejecutada

### 1. Congelar alcance de la capa de servicios
**Acción efectiva**
Editar `configs/base.yaml` y agregar la sección `services_layer`.

**Resultado / efecto útil**
Quedaron fijados:
- inputs de artefactos;
- defaults de vista;
- vistas objetivo de `v1`;
- y contrato explícito de no depender de Streamlit.

---

### 2. Inspeccionar artefactos reales que consumirá la capa
**Comando efectivo**
Inspección con Python de:
- `features`
- `targets`
- `benchmark_regimes`
- `ml_forecasts`
- `model_comparison`
- `decision`
- `structural_breaks`

**Resultado / efecto útil**
Se confirmó que:
- `decision_summary` y `decision_reasons` existen como artefactos globales;
- por símbolo ya existen métricas, paneles y eventos estructurales;
- `features` y `targets` son suficientes para `overview`;
- la última fecha con features puede ser posterior a la última fecha con target válido.

---

### 3. Crear loaders reusables
**Archivo efectivo**
- `src/quant_platform/services/artifact_loaders.py`

**Resultado / efecto útil**
Se centralizó la lógica de carga de artefactos del proyecto y se eliminaron rutas hardcodeadas dispersas.

---

### 4. Exportar loaders en el paquete
**Acción efectiva**
Editar:
- `src/quant_platform/services/__init__.py`

**Resultado / efecto útil**
Los loaders quedaron disponibles desde `quant_platform.services`.

---

### 5. Corregir exports del paquete
**Acción efectiva**
Ajustar `src/quant_platform/services/__init__.py` cuando los imports iniciales fallaron.

**Resultado / efecto útil**
La service layer quedó correctamente exportada y consumible desde scripts y tests.

---

### 6. Crear servicio de overview
**Archivo efectivo**
- `src/quant_platform/services/overview_service.py`

**Resultado / efecto útil**
Quedó implementado un servicio reusable para:
- armar serie temporal merged;
- construir snapshot del símbolo;
- obtener decisión del modelo;
- y recuperar quiebres recientes.

---

### 7. Validar servicio de overview
**Comando efectivo**
Prueba local sobre `SPY`.

**Resultado / efecto útil**
Se verificó que:
- `snapshot_symbol = SPY`
- `snapshot_decision = do_not_promote_ml`
- `timeseries_rows = 2075`
- la última fecha con features y la última fecha con target válido son distintas y el servicio lo maneja bien.

---

### 8. Crear servicio de model comparison
**Archivo efectivo**
- `src/quant_platform/services/model_comparison_service.py`

**Resultado / efecto útil**
Quedó implementada una capa reusable para:
- métricas largas;
- pivot benchmark vs ML;
- resumen agregado por métrica;
- y panel fila a fila.

---

### 9. Validar servicio de model comparison
**Comando efectivo**
Prueba local sobre `SPY`.

**Resultado / efecto útil**
Se verificó que:
- `metrics_rows = 140`
- `pivot_rows = 70`
- `summary_rows = 5`
- `panel_rows = 3510`

---

### 10. Crear servicio de structural changes
**Archivo efectivo**
- `src/quant_platform/services/structural_changes_service.py`

**Resultado / efecto útil**
Quedó implementada una capa reusable para:
- señal preparada;
- eventos detectados;
- eventos recientes;
- y resumen corto del detector.

---

### 11. Validar servicio de structural changes
**Comando efectivo**
Prueba local sobre `SPY`.

**Resultado / efecto útil**
Se verificó que:
- `summary_event_count = 9`
- `signal_rows = 2069`
- `events_rows = 9`
- `recent_rows = 5`

---

### 12. Crear smoke checker de la service layer
**Archivo efectivo**
- `scripts/check_services_layer_smoke.py`

**Resultado / efecto útil**
Quedó definida una validación operativa mínima sobre:
- overview;
- model comparison;
- structural changes;
- símbolos disponibles.

---

### 13. Ejecutar smoke checker
**Comando efectivo**
```bash
python scripts/check_services_layer_smoke.py
```

**Resultado / efecto útil**
La service layer quedó validada en:
```text
SERVICES LAYER SMOKE CHECKS: PASS
```

con líneas OK para:
- `GLD`
- `HYG`
- `SPY`
- `TLT`

---

### 14. Crear tests unitarios de la capa
**Archivo efectivo**
- `tests/unit/test_services_layer.py`

**Resultado / efecto útil**
Se cubrieron:
- universo disponible;
- carga de `decision_summary`;
- snapshot y bundle de overview;
- resumen y bundle de model comparison;
- resumen y bundle de structural changes.

---

### 15. Ejecutar tests unitarios
**Comando efectivo**
```bash
pytest tests/unit/test_services_layer.py -q
```

**Resultado / efecto útil**
Cierre unitario del bloque con:
```text
8 passed
```

---

## Estado de cierre del paso
El Paso 27 quedó resuelto localmente en su versión `v1` con:
- loaders reutilizables;
- servicios puros desacoplados de Streamlit;
- smoke checker en PASS;
- tests unitarios en PASS.

## Siguiente paso natural
El siguiente paso razonable es usar esta service layer como base del:
- dashboard local en Streamlit,
- o una futura capa de API,
sin reescribir la lógica central del producto.
