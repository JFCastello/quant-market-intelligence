# Bitácora ejecutiva — Paso 26: structural breaks

## Propósito
Detectar cambios estructurales por activo y dejar eventos interpretables persistidos para futura visualización y análisis.

## Resultado útil logrado
Quedó materializada una nueva capa en:

- `artifacts/evaluations/structural_breaks/{symbol}/*_structural_break_signal_v1.parquet`
- `artifacts/evaluations/structural_breaks/{symbol}/*_structural_break_events_v1.parquet`

para:
- `SPY`
- `TLT`
- `GLD`
- `HYG`

Además:
- el checker de calidad quedó en PASS;
- y el test unitario específico del builder quedó en PASS.

---

## Secuencia efectiva ejecutada

### 1. Congelar configuración del bloque
**Acción efectiva**
Editar `configs/base.yaml` y agregar la sección `structural_breaks`.

**Resultado / efecto útil**
Quedaron fijados:
- señales de entrada;
- algoritmo;
- costo;
- parámetros de segmentación;
- rutas de salida.

---

### 2. Verificar dependencia real
**Comando efectivo**
```bash
pip install ruptures
pip freeze > requirements.txt
```

**Resultado / efecto útil**
La librería `ruptures` quedó disponible en el entorno.

---

### 3. Inspeccionar contratos reales de entrada
**Comando efectivo**
Inspección con Python de:
- `data/features/{symbol}/*.parquet`
- `data/targets/{symbol}/*.parquet`

**Resultado / efecto útil**
Se confirmó que:
- la señal real disponible era `log_ret_1d`, no `ret_1d`;
- `future_rv_5d` estaba en `targets`;
- el join por `instrument_id + date` era limpio;
- quedaban más de 2068 filas útiles por activo tras `dropna`.

---

### 4. Corregir configuración del retorno
**Acción efectiva**
Actualizar `structural_breaks.signals.return_signal_column` a `log_ret_1d`.

**Resultado / efecto útil**
La configuración quedó alineada con el estado real de los parquets de features.

---

### 5. Crear builder reusable
**Archivo efectivo**
- `src/quant_platform/evaluation/structural_breaks.py`

**Resultado / efecto útil**
Se implementó:
- `StructuralBreakArtifacts`
- `build_structural_break_artifacts(...)`

con producción de:
- `signal_df`
- `events_df`

---

### 6. Exportar el builder en el paquete
**Acción efectiva**
Editar:
- `src/quant_platform/evaluation/__init__.py`

**Resultado / efecto útil**
El builder quedó importable desde:
```python
from quant_platform.evaluation import build_structural_break_artifacts
```

---

### 7. Validar el builder sobre SPY real
**Comando efectivo**
Ejecución local sobre `SPY`.

**Resultado / efecto útil**
Se verificó que:
- la señal tenía 2069 filas útiles;
- las columnas estandarizadas estaban presentes;
- se detectaron 9 eventos de quiebre estructural.

---

### 8. Crear script operativo
**Archivo efectivo**
- `scripts/detect_structural_breaks.py`

**Resultado / efecto útil**
Quedó implementada la materialización del detector para todo el universo.

---

### 9. Ejecutar la detección completa
**Comando efectivo**
```bash
python scripts/detect_structural_breaks.py
```

**Resultado / efecto útil**
Se materializaron señales y eventos para:
- `GLD`
- `HYG`
- `SPY`
- `TLT`

Conteo detectado:
- `GLD`: 7
- `HYG`: 6
- `SPY`: 9
- `TLT`: 4

---

### 10. Crear checker de calidad
**Archivo efectivo**
- `scripts/check_structural_breaks_quality.py`

**Resultado / efecto útil**
Quedó definida la validación estructural y cruzada entre `signal_df` y `events_df`.

---

### 11. Corregir checker
**Acción efectiva**
Ajustar el patrón de descubrimiento de archivos del checker para que coincidiera con los nombres reales materializados.

**Resultado / efecto útil**
El checker pasó a descubrir correctamente los artefactos generados.

---

### 12. Ejecutar checker de calidad
**Comando efectivo**
```bash
python scripts/check_structural_breaks_quality.py
```

**Resultado / efecto útil**
La nueva capa quedó validada en:

```text
STRUCTURAL BREAK QUALITY CHECKS: PASS
```

---

### 13. Crear tests unitarios del builder
**Archivo efectivo**
- `tests/unit/test_structural_breaks.py`

**Resultado / efecto útil**
Se cubrieron:
- artefactos esperados;
- detección de quiebre sintético;
- error por columna faltante;
- error por filas insuficientes;
- error por `instrument_id` inconsistente.

---

### 14. Ejecutar tests unitarios
**Comando efectivo**
```bash
pytest tests/unit/test_structural_breaks.py -q
```

**Resultado / efecto útil**
Cierre unitario del bloque con:

```text
5 passed
```

---

## Estado de cierre del paso
El Paso 26 quedó resuelto localmente en su versión `v1` con:
- artefactos persistidos;
- checker en PASS;
- tests unitarios en PASS;
- eventos detectados para todo el universo activo.

## Siguiente paso natural
El siguiente paso razonable es integrar esta salida en una capa de:
- servicios Python reutilizables,
- o dashboard / visualización,

para que los cambios estructurales se vuelvan visibles dentro del producto.
