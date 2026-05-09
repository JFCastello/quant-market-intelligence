# Explicación de la sección `features` en `configs/base.yaml`

## Propósito de esta sección

La sección `features` de `configs/base.yaml` define **qué familias de features se activan**, **con qué parámetros se calculan** y **qué decisiones de diseño gobiernan el feature engineering v1**.

No es solamente una lista de columnas: es la **especificación operativa** de cómo debe construirse la capa de features base del proyecto.

---

## Bloque recomendado

```yaml
features:
  feature_version: v1
  price_column: close
  annualization_factor: 252
  drop_warmup_rows: false
  include_intermediate_columns: false

  returns:
    enabled: true
    method: log
    windows: [1, 5]

  volatility:
    enabled: true
    base_return_column: log_ret_1d
    windows: [5, 10, 20, 60]
    annualize: true

  intraday_range:
    enabled: true
    include_hl_range: true
    include_co_range: true

  atr:
    enabled: true
    window: 14
    normalize_by: close

  momentum:
    enabled: true
    method: log
    windows: [10]

  moving_averages:
    enabled: true
    windows: [5, 20, 60]
    ratios:
      - [5, 20]
      - [20, 60]

  drawdown:
    enabled: true
    windows: [20, 60]

  context_market:
    enabled: false
```

---

## Explicación fila por fila

## `features:`

### Qué representa
Es la clave raíz que agrupa toda la configuración relacionada con feature engineering.

### Para qué sirve
Permite separar claramente la lógica de features del resto de la configuración del proyecto, como datos, forecast, rutas o modelado.

### Qué se busca lograr
Tener un solo bloque central donde viva la especificación completa del feature set v1.

---

## `feature_version: v1`

### Qué representa
La versión oficial del feature set.

### Para qué sirve
Permite versionar el diseño de features. Si más adelante cambian ventanas, fórmulas, nombres o familias de features, se podría pasar a `v2` sin mezclar datasets incompatibles.

### Qué se busca lograr
Trazabilidad. Cada archivo de features debe poder identificarse como construido bajo una especificación concreta.

---

## `price_column: close`

### Qué representa
La columna base de precio que se toma como referencia principal para varios cálculos.

### Para qué sirve
Indica al pipeline qué columna usar como precio “canónico” para retornos, medias móviles, momentum, drawdown y normalizaciones relacionadas.

### Por qué `close`
Porque en series OHLCV diarias el cierre es la referencia más estándar para modelado cuantitativo de features de horizonte diario.

### Qué se busca lograr
Consistencia matemática en todo el pipeline.

---

## `annualization_factor: 252`

### Qué representa
El factor usado para anualizar medidas de volatilidad.

### Para qué sirve
Cuando calculas una desviación estándar rolling de retornos diarios, la vuelves comparable en escala anual multiplicando por `sqrt(252)`.

### Por qué `252`
Porque es una aproximación estándar al número de días bursátiles al año.

### Qué se busca lograr
Que las volatilidades calculadas tengan una escala interpretable y estándar.

---

## `drop_warmup_rows: false`

### Qué representa
La decisión sobre si deben eliminarse las filas iniciales que quedan con `NaN` por falta de historial suficiente.

### Para qué sirve
Controla si el pipeline persiste toda la serie o si recorta automáticamente las primeras filas afectadas por ventanas rolling.

### Por qué está en `false`
Porque en esta etapa conviene **persistir la serie completa** y dejar explícitos los `NaN` estructurales. Eso preserva trazabilidad y evita esconder el efecto de las ventanas.

### Qué se busca lograr
Separar claramente dos cosas:
- `NaN` normales por warm-up
- `NaN` anómalos por errores del pipeline

---

## `include_intermediate_columns: false`

### Qué representa
La decisión sobre si se guardan o no columnas auxiliares de cálculo.

### Ejemplos de columnas intermedias
- `prev_close`
- `true_range`
- `rolling_max_20`
- `rolling_max_60`

### Para qué sirve
Evita que el output final de features quede contaminado con columnas técnicas internas que no pertenecen al contrato del dominio.

### Por qué está en `false`
Porque el dataset final debe contener solo las features canónicas, no todos los pasos internos del cálculo.

### Qué se busca lograr
Un output limpio, estable y fácil de usar en modelado.

---

# Bloque `returns`

## `returns:`

### Qué representa
La familia de features de retornos.

### Para qué sirve
Agrupa la configuración de todas las variables que describen cambio de precio entre fechas.

### Qué se busca lograr
Construir una base sólida sobre la cual luego puedan definirse volatilidades, momentum y otros indicadores derivados.

---

## `enabled: true` dentro de `returns`

### Qué representa
Activa esta familia de features.

### Para qué sirve
Permite que el pipeline sepa si debe calcular o no los retornos.

### Qué se busca lograr
Hacer que la generación de features sea modular y controlable desde configuración.

---

## `method: log`

### Qué representa
La convención matemática usada para calcular los retornos.

### Para qué sirve
Define si el retorno será aritmético o logarítmico.

### Por qué `log`
Porque los retornos logarítmicos son muy usados en finanzas cuantitativas: son más cómodos para agregación temporal y muy naturales para cálculos de volatilidad.

### Qué se busca lograr
Consistencia con el resto del pipeline de volatilidad.

---

## `windows: [1, 5]`

### Qué representa
Los horizontes de retorno que se calcularán.

### Qué columnas genera
- `log_ret_1d`
- `log_ret_5d`

### Para qué sirve
Captura movimiento de precio de corto plazo y de horizonte algo más extendido.

### Qué se busca lograr
Dar al modelo una señal base de dirección y magnitud reciente.

---

# Bloque `volatility`

## `volatility:`

### Qué representa
La familia de features de volatilidad rolling.

### Para qué sirve
Define cómo medir la variabilidad reciente de los retornos.

### Qué se busca lograr
Aportar al modelo memoria estadística del comportamiento reciente del activo.

---

## `enabled: true` dentro de `volatility`

### Qué representa
Activa el cálculo de volatilidades rolling.

### Qué se busca lograr
Incluir explícitamente una familia central para un producto cuyo problema principal es forecast de volatilidad.

---

## `base_return_column: log_ret_1d`

### Qué representa
La columna de retornos a partir de la cual se calcularán las volatilidades rolling.

### Para qué sirve
Deja explícito que la volatilidad no se calcula directamente desde precios, sino desde retornos diarios.

### Qué se busca lograr
Evitar ambigüedad metodológica.

---

## `windows: [5, 10, 20, 60]`

### Qué representa
Las ventanas temporales usadas para calcular desviaciones estándar rolling.

### Qué columnas genera
- `vol_5d`
- `vol_10d`
- `vol_20d`
- `vol_60d`

### Para qué sirve
Capturar distintos horizontes de memoria:
- 5 días: reacción rápida
- 10 días: corto plazo ampliado
- 20 días: ventana mensual aproximada
- 60 días: estado más persistente

### Qué se busca lograr
Que el modelo vea simultáneamente volatilidad muy reciente y volatilidad más estable.

---

## `annualize: true`

### Qué representa
Indica que la volatilidad rolling debe anualizarse.

### Para qué sirve
Convierte la volatilidad diaria rolling en una escala anual estándar.

### Qué se busca lograr
Comparabilidad e interpretabilidad financiera.

---

# Bloque `intraday_range`

## `intraday_range:`

### Qué representa
La familia de features que capturan amplitud y movimiento intradía.

### Para qué sirve
Agregar información que no aparece mirando solo el cierre contra el cierre anterior.

### Qué se busca lograr
Capturar expansión/compresión del rango diario.

---

## `enabled: true` dentro de `intraday_range`

### Qué representa
Activa esta familia.

### Qué se busca lograr
Usar información OHLC adicional más allá del simple close-to-close.

---

## `include_hl_range: true`

### Qué representa
Activa la feature basada en la amplitud `high-low`.

### Qué columna genera
- `hl_range`

### Interpretación sugerida
Mide qué tan amplio fue el rango del día en relación con el precio.

### Qué se busca lograr
Capturar jornadas comprimidas frente a jornadas expansivas.

---

## `include_co_range: true`

### Qué representa
Activa la feature basada en el movimiento entre apertura y cierre.

### Qué columna genera
- `co_range`

### Interpretación sugerida
Resume si el día cerró por encima o por debajo de la apertura y con qué magnitud relativa.

### Qué se busca lograr
Aportar una señal simple de direccionalidad intradía.

---

# Bloque `atr`

## `atr:`

### Qué representa
La familia de Average True Range.

### Para qué sirve
Medir el “rango efectivo” del precio, incorporando gaps respecto al cierre previo.

### Qué se busca lograr
Complementar la volatilidad rolling con una medida robusta basada en rangos.

---

## `enabled: true` dentro de `atr`

### Qué representa
Activa el cálculo de ATR.

### Qué se busca lograr
Añadir una medida clásica e interpretable de agitación del mercado.

---

## `window: 14`

### Qué representa
La longitud de la ventana usada para promediar el true range.

### Qué columna genera
- `atr_14`

### Por qué `14`
Porque es una ventana estándar y suficientemente corta como para reaccionar, pero no tan ruidosa como una muy pequeña.

### Qué se busca lograr
Tener una medida de rango suavizada y útil para un v1 estable.

---

## `normalize_by: close`

### Qué representa
La convención usada para normalizar ATR y volverlo comparable entre activos con distintos niveles de precio.

### Para qué sirve
Evita que ATR dependa excesivamente de la escala nominal del activo.

### Qué se busca lograr
Comparabilidad entre instrumentos como SPY, TLT, GLD y HYG.

---

# Bloque `momentum`

## `momentum:`

### Qué representa
La familia de features de persistencia de movimiento.

### Para qué sirve
Agregar una señal simple de tendencia reciente.

### Qué se busca lograr
Capturar inercia o continuidad del movimiento del precio.

---

## `enabled: true` dentro de `momentum`

### Qué representa
Activa el cálculo de momentum.

### Qué se busca lograr
Incluir una señal direccional sencilla sin sobrecargar el v1.

---

## `method: log`

### Qué representa
La convención usada para calcular momentum.

### Para qué sirve
Hace consistente el cálculo de momentum con la convención usada en retornos.

### Qué se busca lograr
Uniformidad matemática dentro del pipeline.

---

## `windows: [10]`

### Qué representa
La ventana temporal usada para medir momentum.

### Qué columna genera
- `mom_10d`

### Por qué solo `10`
Porque para un v1 conviene comenzar con una señal simple y estable, no con demasiadas variantes redundantes.

### Qué se busca lograr
Añadir una primera señal de tendencia sin complejidad innecesaria.

---

# Bloque `moving_averages`

## `moving_averages:`

### Qué representa
La familia de medias móviles y relaciones entre medias.

### Para qué sirve
Aportar información de suavizado y tendencia a distintos horizontes.

### Qué se busca lograr
Que el modelo vea la estructura del precio a corto, medio y algo más largo plazo.

---

## `enabled: true` dentro de `moving_averages`

### Qué representa
Activa esta familia.

### Qué se busca lograr
Añadir contexto de tendencia de manera interpretable.

---

## `windows: [5, 20, 60]`

### Qué representa
Las ventanas de medias móviles que se calcularán.

### Qué columnas genera
- `ma_5`
- `ma_20`
- `ma_60`

### Para qué sirve
- `ma_5`: corto plazo
- `ma_20`: referencia tipo mensual bursátil
- `ma_60`: tendencia más persistente

### Qué se busca lograr
Dar al modelo una noción gradual de tendencia y nivel suavizado del precio.

---

## `ratios:`

### Qué representa
La lista de pares de ventanas cuya relación debe calcularse.

### Qué columnas genera
- `ma_ratio_5_20`
- `ma_ratio_20_60`

### Para qué sirve
Las ratios entre medias suelen ser más informativas que las medias absolutas para identificar fase de tendencia, compresión o desacople.

### Qué se busca lograr
Obtener señales relativas más robustas que el simple nivel de una media móvil aislada.

---

## `- [5, 20]`

### Qué representa
Define el ratio entre media móvil de 5 días y media móvil de 20 días.

### Qué columna genera
- `ma_ratio_5_20`

### Qué se busca lograr
Capturar relación entre corto plazo y horizonte mensual aproximado.

---

## `- [20, 60]`

### Qué representa
Define el ratio entre media móvil de 20 días y media móvil de 60 días.

### Qué columna genera
- `ma_ratio_20_60`

### Qué se busca lograr
Capturar relación entre tendencia media y tendencia más persistente.

---

# Bloque `drawdown`

## `drawdown:`

### Qué representa
La familia de features de deterioro relativo respecto a máximos rolling.

### Para qué sirve
Mide cuánto se ha alejado el precio de su máximo reciente.

### Qué se busca lograr
Añadir contexto de fragilidad, recuperación o presión acumulada del precio.

---

## `enabled: true` dentro de `drawdown`

### Qué representa
Activa esta familia.

### Qué se busca lograr
Incluir una señal relevante para distinguir periodos de tensión y deterioro persistente.

---

## `windows: [20, 60]`

### Qué representa
Los horizontes sobre los cuales se calculará drawdown.

### Qué columnas genera
- `drawdown_20`
- `drawdown_60`

### Para qué sirve
- `drawdown_20`: deterioro reciente
- `drawdown_60`: deterioro más extendido

### Qué se busca lograr
Que el modelo vea si el activo está apenas corrigiendo o si viene arrastrando una caída más estructural.

---

# Bloque `context_market`

## `context_market:`

### Qué representa
La familia de features cruzadas o contextuales del mercado.

### Para qué sirve
Reservar desde ya un espacio claro para features cross-asset o macro de contexto, pero sin activarlas todavía.

### Qué se busca lograr
Mantener separación limpia entre:
- paso 16: features base
- paso 17: features de contexto

---

## `enabled: false` dentro de `context_market`

### Qué representa
Desactiva explícitamente las features contextuales en esta etapa.

### Por qué está en `false`
Porque en este bloque del proyecto todavía no corresponde mezclar señales cruzadas como:
- retornos de SPY usados como contexto de otro activo
- spreads entre activos
- correlaciones rolling cross-asset

### Qué se busca lograr
Mantener el alcance del paso 16 bien controlado y metodológicamente limpio.

---

# Resumen operativo

## Qué controla esta sección del YAML
Esta sección controla cinco decisiones grandes del pipeline:

1. **versión del feature set**
2. **convenciones matemáticas generales**
3. **familias de features activas**
4. **ventanas temporales por familia**
5. **qué queda dentro o fuera del output final**

---

## Qué columnas finales produce este diseño

```text
log_ret_1d
log_ret_5d
vol_5d
vol_10d
vol_20d
vol_60d
hl_range
co_range
atr_14
mom_10d
ma_5
ma_20
ma_60
ma_ratio_5_20
ma_ratio_20_60
drawdown_20
drawdown_60
```

---

## Qué no pertenece todavía a esta sección
No deberían entrar todavía aquí:

- targets como `future_rv_5d`
- umbrales de régimen
- features cross-asset activas
- splits de train/test
- parámetros de benchmark o ML

Eso pertenece a bloques posteriores del proyecto.

---

## Idea central

La sección `features` no existe solo para “encender columnas”.
Su función real es **congelar la semántica del feature engineering v1** para que:

- el cálculo sea reproducible,
- el dominio sea estable,
- los scripts sean consistentes,
- y el equipo trabaje con una sola fuente de verdad.
