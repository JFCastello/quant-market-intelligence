# Regime Targets v1

## 1. Propósito

Este documento congela la semántica del bloque de **regime targets v1** del proyecto **Quant Market Intelligence**.

Su función es dejar explícito:

- qué significa `future_regime_5d`;
- a partir de qué target continuo se construye;
- cómo se calculan los thresholds;
- por qué esos thresholds deben salir solo de train;
- cómo se asignan las etiquetas discretas;
- cómo se manejan los valores faltantes;
- qué parte de esta lógica se implementa ahora y qué parte se reserva para la etapa de splits temporales.

En esta etapa, el objetivo es dejar lista la **lógica reusable de discretización**, no materializar todavía un dataset global definitivo de regímenes.

---

## 2. Alcance de esta versión

### Sí incluye

- definición formal de `future_regime_5d`;
- vínculo con el target continuo `future_rv_5d`;
- definición de thresholds por cuantiles;
- política de cálculo `train_only`;
- política de asignación de labels;
- política de `NaN`;
- criterios semánticos que deberán respetar los builders del bloque de régimen.

### No incluye todavía

- construcción global de `future_regime_5d` sobre toda la historia;
- uso de thresholds calculados con toda la serie;
- materialización de targets discretos finales sin conocer los splits;
- aplicación completa por fold de walk-forward.

Eso se hará cuando el pipeline temporal de train/validation/test esté formalmente construido.

---

## 3. Definición del target discreto principal

## `future_regime_5d`

### Qué representa
Una etiqueta discreta que resume el estado futuro de riesgo del activo en el horizonte de 5 días.

### Fuente de origen
Se construye a partir del target continuo:

```text
future_rv_5d
```

### Qué pregunta responde
Para una fila fechada en `t`, responde algo del estilo:

> “¿El entorno de riesgo futuro del activo, medido por su volatilidad realizada a 5 días, debe considerarse calm, normal o stress?”

### Intuición
No es una medición primaria; es una **discretización controlada** del target continuo.

---

## 4. Fuente continua de discretización

El target discreto nace de:

```text
future_rv_5d
```

### Por qué
Porque el benchmark clásico y el modelo ML deben competir sobre un mismo objetivo continuo, y luego compararse también sobre una versión discreta interpretable.

### Ventaja
Esto evita que los regímenes sean una etiqueta arbitraria o difícil de validar.  
Primero se define una magnitud continua bien fundada; luego se discretiza.

---

## 5. Método de discretización

El método congelado para `v1` es:

```text
quantile_bins
```

### Qué significa
Se divide la distribución del target continuo en cuantiles y se asignan etiquetas según el rango en el que cae cada observación.

### Configuración adoptada
Se usarán tres clases:

- `calm`
- `normal`
- `stress`

y dos cortes cuantílicos:

```text
[0.33, 0.66]
```

---

## 6. Thresholds de régimen

Sea la serie continua de entrenamiento:

```text
future_rv_5d_train
```

Se calculan:

- `q1 = quantile(train, 0.33)`
- `q2 = quantile(train, 0.66)`

Estos dos valores actúan como umbrales del target discreto.

### Interpretación
- `q1`: separa el régimen bajo del intermedio;
- `q2`: separa el régimen intermedio del alto.

---

## 7. Regla de asignación de clases

Con los thresholds `q1` y `q2`, la asignación recomendada en `v1` es:

- si `future_rv_5d <= q1` → `calm`
- si `q1 < future_rv_5d <= q2` → `normal`
- si `future_rv_5d > q2` → `stress`

### Política de inclusividad congelada

```text
lower_bin_inclusive = true
upper_bin_inclusive = false
```

### Interpretación operativa
Esto hace explícito cómo tratar valores exactamente iguales a los thresholds y evita ambigüedades en la implementación.

---

## 8. Política `train_only`

La fuente oficial de thresholds es:

```text
train_only
```

### Qué significa
Los cuantiles deben calcularse exclusivamente usando la porción de entrenamiento del split activo.

### Por qué es obligatorio
Porque si se usan observaciones de validation o test para calcular los umbrales, se introduce fuga de información.

### Regla conceptual
Cada fold temporal tendrá sus propios thresholds, derivados solo de su train.

---

## 9. Política de `NaN`

Si la columna fuente:

```text
future_rv_5d
```

tiene `NaN`, entonces:

```text
future_regime_5d = NaN
```

### Razón
No tiene sentido asignar una clase discreta si el target continuo no existe.

### Ejemplo típico
Las últimas 5 filas de cada activo, que en el paso 18 quedan sin `future_rv_5d` por falta de futuro, también deben quedar sin `future_regime_5d`.

---

## 10. Contrato de salida esperado

El contrato `TargetRow` ya queda preparado para aceptar:

```text
instrument_id
date
target_version
future_rv_5d
future_regime_5d
```

### Nota importante
En esta etapa, el bloque de targets discretos se diseña y se prepara, pero no se materializa todavía de forma global y definitiva sobre toda la historia.

---

## 11. Por qué no se debe materializar globalmente todavía

Aunque el YAML ya deje activado conceptualmente el target discreto, hay una diferencia importante entre:

- **definir la lógica**
- **aplicarla correctamente en un pipeline temporal**

### Problema de una materialización global
Si discretizas todo el histórico usando cuantiles de toda la serie, entonces los thresholds habrán visto información futura respecto de muchos folds.

### Consecuencia
Eso rompería la disciplina temporal del proyecto y contaminaría la evaluación posterior.

### Decisión correcta
En `v1`, la lógica reusable del régimen se construye ahora, pero su aplicación final debe ocurrir cuando el sistema ya conozca los splits temporales.

---

## 12. Qué sí debe construirse en esta etapa

En esta fase deben quedar listos:

- el bloque semántico en YAML;
- este documento de semántica;
- un builder reusable que calcule thresholds y asigne clases;
- tests unitarios de discretización.

### Qué no se hace todavía
No se genera aún un parquet final “global” de `future_regime_5d` como si los thresholds fueran universales.

---

## 13. Metadatos mínimos esperados del builder reusable

Cuando se implemente el builder de régimen, debería poder devolver, además de la serie clasificada, metadata como:

- nombre del target continuo fuente;
- labels utilizadas;
- cuantiles configurados;
- thresholds numéricos resultantes;
- política de inclusividad;
- política `train_only`.

Esto será útil para trazabilidad y debugging por fold.

---

## 14. Resumen operativo

El bloque de régimen v1 debe funcionar así:

1. tomar una serie continua `future_rv_5d` en train;
2. calcular `q1` y `q2` con cuantiles configurados;
3. congelar esos thresholds;
4. aplicar esos thresholds a cualquier subconjunto compatible;
5. asignar una de tres etiquetas:
   - `calm`
   - `normal`
   - `stress`
6. preservar `NaN` cuando el target continuo sea `NaN`;
7. no recalcular thresholds usando validation o test.

---

## 15. Resumen final

El bloque de régimen v1 queda congelado así:

- target discreto: `future_regime_5d`
- fuente: `future_rv_5d`
- método: `quantile_bins`
- clases: `calm`, `normal`, `stress`
- cuantiles: `0.33`, `0.66`
- thresholds calculados con: `train_only`
- política de bins:
  - `x <= q1` → `calm`
  - `q1 < x <= q2` → `normal`
  - `x > q2` → `stress`
- política de faltantes:
  - si `future_rv_5d` es `NaN`, entonces `future_regime_5d` también es `NaN`

Este documento debe tomarse como la **fuente de verdad semántica** para implementar:

- `src/quant_platform/targets/regime_builders.py`
- `tests/unit/test_regime_builders.py`

y más adelante, cuando existan splits temporales formales, para aplicar la discretización de manera correcta por fold.
