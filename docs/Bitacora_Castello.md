# Paso 16 del SSD ----------------------------------------------------------------------

Comienzo formal del bloque de **feature engineering base**.  
Se crea una rama dedicada para aislar este bloque de trabajo:

```bash
git checkout -b feat/feature-engineering-v1
```

La intención del paso 16 es construir la primera capa seria de variables explicativas del proyecto, de manera reproducible, versionada y validable, a partir de los parquet normalizados ya consolidados.

## 16.1 Modificar `configs/base.yaml`

Se agrega una sección nueva llamada `features`.  
Aquí se congela la definición oficial del **feature set v1**, es decir, qué variables se calculan, con qué ventanas, con qué método y con qué convenciones.

Esto convierte al YAML en la **fuente de verdad de configuración** del bloque de features.

Para consultar el significado exacto de cada feature, ver:

```text
docs/runbooks/feature\_engineering\_v1.md
```

\---

## 16.2 Modificar `src/quant\_platform/schemas/feature\_row.py`

Se amplía el schema `FeatureRow` para que el contrato del dominio quede alineado con las nuevas columnas definidas en `features:` dentro del YAML.

La idea es que el contrato de datos acepte explícitamente las variables que el pipeline va a generar, evitando divergencias entre:

* configuración,
* implementación,
* persistencia,
* validación.

\---

## 16.3 Crear `docs/runbooks/feature\_engineering\_v1.md`

Se documenta formalmente la semántica del bloque de features base.

Este documento deja congelado:

* listado completo de features;
* fórmula matemática de cada una;
* ventana temporal utilizada;
* interpretación operativa breve;
* política de `NaN` por warm-up;
* decisión de qué queda dentro y fuera del `feature set v1`.

Este archivo actúa como referencia conceptual antes de escribir o modificar código.

\---

## 16.4 Crear `src/quant\_platform/features/builders.py`

Este módulo implementa el motor central del paso 16.

Su responsabilidad es:

* recibir un DataFrame normalized ya limpio;
* validar que tenga las columnas mínimas necesarias;
* ordenarlo por `instrument\_id` y `date`;
* calcular todas las features base;
* adaptar la salida al contrato `FeatureRow`;
* validar que el resultado final tenga sentido antes de persistirlo.

Incluye el cálculo de:

* retornos logarítmicos;
* volatilidades rolling;
* rango intradía;
* ATR;
* momentum;
* medias móviles;
* ratios entre medias;
* drawdowns;
* `feature\_version`.

En otras palabras, aquí vive la lógica analítica reusable del paso 16.

\---

## 16.5 Crear `scripts/build\_features.py`

Este script es el **orquestador operativo** del pipeline de features base.

Para cada símbolo del universo definido en settings:

* descubre el parquet normalized más reciente;
* lo carga;
* ejecuta el builder de features;
* adapta la salida al contrato;
* valida el resultado;
* guarda el dataset final en `data/features/` en formato Parquet.

Es el job que materializa la capa de features base del proyecto.

\---

## 16.6 Crear `scripts/check\_features\_quality.py`

Este script ejecuta controles de calidad sobre los archivos de features generados.

Para cada archivo de `data/features/`, verifica:

* existencia del archivo;
* columnas esperadas;
* ausencia de nulos en claves críticas;
* ausencia de duplicados por `instrument\_id/date`;
* orden temporal correcto;
* valores numéricos válidos;
* ausencia de `inf` y `-inf`;
* restricciones específicas por familia:

  * volatilidades `>= 0`
  * ATR `>= 0`
  * drawdowns `<= 0`
  * `hl\_range >= 0`

Si detecta problemas, aborta con error y reporta exactamente qué falló.

\---

## 16.7 Crear `tests/unit/test\_feature\_builders.py`

Suite de tests unitarios para el bloque de features base.

Verifica, al menos:

1. que `get\_enabled\_feature\_columns(...)` devuelva exactamente el contrato esperado;
2. que `build\_base\_features(...)` produzca las columnas correctas;
3. que el cálculo de `log\_ret\_1d` coincida con una cuenta manual;
4. que `validate\_feature\_output(...)` acepte un dataset válido;
5. que el orden temporal se preserve incluso si el input entra desordenado.

Este archivo protege el corazón analítico del paso 16 frente a regresiones futuras.

# \--------------------------------------------------------------------------------------

# Paso 17 del SSD ----------------------------------------------------------------------

Construcción del bloque de **features de contexto de mercado**, usando el resto del universo como fuente de información adicional.

La lógica deja de ser puramente univariada y pasa a incorporar relaciones entre activos:

* retornos de proxies de mercado,
* spreads simples risk-on / risk-off,
* volatilidad relativa,
* correlación rolling contra el activo objetivo.

\---

## 17.1 Modificar `configs/base.yaml`

Se agrega una nueva sección llamada `context\_features`.

Aquí se congela la definición oficial del **context feature set v1**, incluyendo:

* roles de mercado;
* retornos directos contextuales;
* volatilidades directas;
* spreads;
* volatilidad relativa;
* correlación rolling.

Esta configuración se diseña con una lógica **role-based**, para que el sistema sea extensible cuando el universo crezca.

Para consultar la semántica económica y matemática de estas columnas, ver:

```text
docs/runbooks/context\_features\_v1.md
```

\---

## 17.2 Extender `src/quant\_platform/schemas/feature\_row.py`

Se amplía `FeatureRow` para incluir las nuevas columnas `ctx\_\*`.

Esto asegura que el contrato del dominio acepte no solo las features base del paso 16, sino también las features de contexto del paso 17.

\---

## 17.3 Crear `docs/runbooks/context\_features\_v1.md`

Documento semántico del bloque de contexto.

Deja congelado:

* qué roles de mercado existen;
* qué representa cada uno;
* qué context features se calculan;
* cómo se interpreta cada una;
* qué fórmulas y ventanas usan;
* qué políticas especiales aplican, por ejemplo la autocorrelación del equity proxy consigo mismo.

Sirve como referencia conceptual para implementación, validación y futuras expansiones del universo.

\---

## 17.4 Crear `src/quant\_platform/features/context\_builders.py`

Módulo analítico del paso 17.

Su trabajo es construir la capa de contexto a partir de un universo de features base ya calculadas.

Entre sus responsabilidades están:

* inferir el mapeo rol → `instrument\_id`;
* construir un panel de contexto a nivel de fecha;
* generar retornos y volatilidades directas por rol;
* construir spreads entre roles;
* calcular volatilidad relativa;
* calcular correlación rolling con el rol de referencia;
* aplicar políticas especiales, por ejemplo `self\_reference\_policy = nan`.

El resultado es un DataFrame enriquecido con columnas `ctx\_\*` que añaden al activo objetivo una representación compacta del entorno de mercado.

\---

## 17.5 Crear `scripts/build\_context\_features.py`

Script operativo que materializa la capa contextual.

Su flujo general es:

* leer las features base del universo;
* concatenarlas en un solo `universe\_df`;
* aplicar `build\_context\_enriched\_features(...)`;
* adaptar la salida al contrato `FeatureRow`;
* separar nuevamente por símbolo;
* guardar los resultados en `data/features\_context/`.

Es la pieza que convierte la capa base del paso 16 en una capa enriquecida de contexto de mercado.

\---

## 17.6 Crear `scripts/check\_context\_features\_quality.py`

Checker específico para la nueva capa `data/features\_context/`.

Valida, por archivo:

1. columnas base + contexto;
2. no nulos en claves críticas;
3. no duplicados por `instrument\_id/date`;
4. fechas parseables y ordenadas;
5. columnas numéricas válidas;
6. ausencia de `inf` / `-inf`;
7. volatilidades no negativas;
8. correlaciones dentro de `\[-1, 1]`;
9. spreads numéricamente sanos;
10. exactamente un `instrument\_id` por archivo;
11. autocorrelación del `equity\_proxy` como `NaN` cuando corresponde.

Es el último filtro antes de consumir esta capa en modelado.

\---

## 17.7 Crear `tests/unit/test\_context\_builders.py`

Suite de tests unitarios del bloque contextual.

Verifica al menos:

1. columnas esperadas de `get\_enabled\_context\_feature\_columns(...)`;
2. inferencia correcta de roles;
3. construcción del panel date-level;
4. cálculo correcto de spreads;
5. enriquecimiento completo sin pérdida de filas;
6. política de autocorrelación `NaN` para el equity proxy.

Usa datos sintéticos controlados para detectar regresiones de lógica contextual.

# \--------------------------------------------------------------------------------------

# Paso 18 del SSD ----------------------------------------------------------------------

Construcción del **target continuo principal del proyecto**:

```text
future\_rv\_5d
```

Este paso ya no genera features, sino la **etiqueta futura** que utilizarán el benchmark clásico y el modelo ML para competir sobre el mismo objetivo.

La lógica del paso 18 consiste en definir, calcular, persistir y validar la volatilidad realizada anualizada de los próximos 5 días de mercado usando únicamente información futura.

\---

## 18.1 Modificar `configs/base.yaml`

Se agrega una nueva sección llamada `targets`.

Aquí se congela la definición oficial del bloque de targets v1, incluyendo:

* `target\_version`
* target continuo principal: `future\_rv\_5d`
* horizonte: 5 días
* columna base de retornos: `log\_ret\_1d`
* anualización: `252`
* tipo de ventana: `fixed\_forward`
* política de ventanas parciales: desactivadas

También se deja definido, pero apagado, el target discreto `future\_regime\_5d`, que se activará recién en el paso 19.

Para consultar la semántica completa del target, ver:

```text
docs/runbooks/targets\_v1.md
```

\---

## 18.2 Revisar / extender `src/quant\_platform/schemas/target\_row.py`

Se confirma y alinea el contrato `TargetRow` con la nueva sección `targets:` del YAML.

El contrato queda preparado para aceptar:

* `instrument\_id`
* `date`
* `target\_version`
* `future\_rv\_5d`
* `future\_regime\_5d`

Aunque en este paso solo se construye `future\_rv\_5d`, se deja el contrato listo para el paso siguiente.

\---

## 18.3 Crear `docs/runbooks/targets\_v1.md`

Documento semántico del bloque de targets v1.

Deja congelado:

* qué representa `future\_rv\_5d`;
* qué retorno base utiliza;
* qué ventana temporal exacta usa (`t+1` a `t+5`);
* cómo se anualiza;
* qué filas quedan en `NaN` por falta de futuro;
* por qué el target discreto queda reservado para el paso 19.

Este archivo es la referencia conceptual del paso 18.

\---

## 18.4 Crear `src/quant\_platform/targets/builders.py`

Módulo central del paso 18.

Sus responsabilidades son:

* validar el DataFrame fuente;
* ordenar por `instrument\_id`, `date`;
* calcular `future\_rv\_5d` usando solo retornos futuros;
* añadir `target\_version`;
* adaptar la salida al contrato `TargetRow`;
* validar el output final.

La fórmula implementada es, conceptualmente:

* tomar `log\_ret\_1d(t+1:t+5)`
* calcular su desviación estándar
* anualizar con `sqrt(252)`

Además, respeta la política de no permitir ventanas parciales, por lo que las últimas 5 filas de cada instrumento quedan en `NaN`.

\---

## 18.5 Crear `scripts/build\_targets.py`

Script operativo que materializa la capa `data/targets/`.

Para cada símbolo:

* descubre el parquet contextual más reciente en `data/features\_context/`;
* lo carga;
* construye el target continuo con `build\_continuous\_targets(...)`;
* adapta la salida al contrato `TargetRow`;
* valida el resultado;
* guarda el parquet final en `data/targets/`.

Es el job que convierte la capa contextual en la capa oficial de targets.

\---

## 18.6 Crear `scripts/check\_targets\_quality.py`

Checker específico para la nueva capa `data/targets/`.

Valida, por archivo:

* existencia;
* columnas esperadas;
* no vacío;
* no nulos en `instrument\_id`, `date`, `target\_version`;
* `target\_version` correcta;
* ausencia de duplicados por `instrument\_id/date`;
* fechas ordenadas;
* `future\_rv\_5d` numérico;
* ausencia de `inf` / `-inf`;
* ausencia de valores negativos;
* exactamente un `instrument\_id` por archivo.

Su función es asegurar que la capa de targets sea segura antes de pasar al target discreto y al modelado.

\---

## 18.7 Crear `tests/unit/test\_target\_builders.py`

Suite de tests unitarios del builder de targets.

Verifica, al menos:

1. que `get\_enabled\_target\_columns(...)` devuelva el contrato correcto;
2. que `build\_continuous\_targets(...)` añada `future\_rv\_5d`;
3. que el número de filas se preserve;
4. que una fila válida coincida con una cuenta manual;
5. que las últimas 5 filas queden en `NaN`;
6. que `adapt\_targets\_to\_contract(...)` devuelva el contrato correcto;
7. que `validate\_target\_output(...)` acepte un dataset válido.

Con esto, el paso 18 queda protegido frente a regresiones y deja listo el terreno para el paso 19.

# \-----------------------------------------------------------------------------------------------

# Paso 19A del SSD ----------------------------------------------------------------------

Construcción de la lógica reusable del target discreto de régimen, sin materializar todavía un dataset
global definitivo de regímenes.

La intención de este bloque es hacer las cosas correctamente desde el punto de vista estadístico:
dejar lista la semántica, la configuración, la documentación y el builder reusable de `future\_regime\_5d`,
pero sin calcular los thresholds usando toda la historia, ya que eso introduciría fuga de información.

El target discreto `future\_regime\_5d` nace a partir de:

future\_rv\_5d

y su discretización debe depender de thresholds calculados solo sobre train en cada split temporal.

## 19A.1 Modificar `configs/base.yaml`

Se refina la sección `targets.classification\_target` para congelar la semántica oficial del target discreto
de régimen.

En este bloque se deja explícito:

* nombre del target discreto: `future\_regime\_5d`
* columna continua fuente: `future\_rv\_5d`
* método de discretización: `quantile\_bins`
* labels oficiales:

  * `calm`
  * `normal`
  * `stress`
* cuantiles:

  * `0.33`
  * `0.66`
* política de thresholds:

  * `train\_only`
* política de inclusividad de bins:

  * inferior inclusivo
  * superior exclusivo
* política de faltantes:

  * si la fuente continua es `NaN`, el target discreto también puede ser `NaN`

Con esto, la semántica de régimen queda congelada antes de la implementación.

## 19A.2 Crear `docs/runbooks/regime\_targets\_v1.md`

Documento semántico del bloque de régimen.

Este archivo deja por escrito:

* qué representa `future\_regime\_5d`
* por qué nace de `future\_rv\_5d`
* cómo se calculan los thresholds
* por qué los thresholds deben salir solo de train
* cómo se asignan las etiquetas:

  * `calm`
  * `normal`
  * `stress`
* qué política se usa para empates y valores en los bordes
* qué ocurre con `NaN`
* por qué no se debe materializar todavía un target discreto global sin conocer primero los splits temporales

Este documento actúa como la referencia conceptual del paso 19A.

## 19A.3 Crear `src/quant\_platform/targets/regime\_builders.py`

Módulo reusable de discretización de regímenes.

Su objetivo no es todavía construir un parquet final de targets discretos, sino implementar correctamente
la lógica reusable del bloque de régimen.

Incluye funciones para:

* validar la serie continua fuente
* calcular thresholds cuantílicos desde una serie de train
* devolver metadata completa de thresholds
* aplicar labels discretos a cualquier serie compatible
* preservar `NaN` cuando la fuente sea `NaN`
* adjuntar el target discreto a un DataFrame existente
* validar que la salida contenga únicamente labels válidos

La regla de clasificación adoptada en `v1` es:

* si `x <= q1` -> `calm`
* si `q1 < x <= q2` -> `normal`
* si `x > q2` -> `stress`

donde `q1` y `q2` se calculan únicamente a partir de train.

## 19A.4 Modificar `src/quant\_platform/targets/\_\_init\_\_.py`

Se exportan las nuevas funciones del bloque de régimen para que puedan importarse desde:

quant\_platform.targets

Esto deja integrado el nuevo builder reusable dentro del paquete del dominio, sin necesidad de importar
rutas internas manualmente.

## 19A.5 Crear `tests/unit/test\_regime\_builders.py`

Suite de tests unitarios para el bloque de discretización de régimen.

Verifica, al menos:

1. que `compute\_quantile\_thresholds(...)` calcule metadata y thresholds válidos
2. que `apply\_regime\_labels(...)` asigne correctamente:

   * `calm`
   * `normal`
   * `stress`
3. que los `NaN` de la fuente se preserven
4. que `build\_regime\_target\_series(...)` devuelva serie + metadata consistente
5. que `attach\_regime\_target\_to\_dataframe(...)` agregue correctamente `future\_regime\_5d`
6. que `validate\_regime\_output\_series(...)` rechace labels inválidos

Estos tests blindan la lógica del target discreto antes de integrarla al pipeline temporal real.

## 19A.6 Qué se logra al cerrar este bloque

Al finalizar 19A, el proyecto ya tiene:

* semántica congelada del target discreto
* configuración formal en YAML
* documentación conceptual del régimen
* builder reusable estadísticamente correcto
* tests unitarios pasando

Lo que todavía no se hace es construir un parquet global definitivo de `future\_regime\_5d`, porque esa
discretización debe aplicarse correctamente una vez existan los splits temporales del paso 20.

En otras palabras:

future\_rv\_5d
-> thresholds desde train
-> future\_regime\_5d

queda ya implementado a nivel reusable, pero su aplicación final por fold se reserva para la siguiente
etapa del pipeline.

# \--------------------------------------------------------------------------------------

Paso 20 del SSD ----------------------------------------------------------------------
Construcción del bloque de splits temporales walk-forward del proyecto.
La intención de este paso es convertir el pipeline de features y targets en un sistema realmente evaluable de forma seria, temporal y reproducible. Hasta el paso 19 ya existían:
features base;
features de contexto;
target continuo `future\_rv\_5d`;
lógica reusable para discretizar `future\_regime\_5d`.
Sin embargo, todavía faltaba la pieza que vuelve estadísticamente válido el entrenamiento y la comparación entre benchmark y ML: una política formal de separación entre `train`, `validation` y `test`, implementada bajo un esquema walk-forward con ventana expansiva y con test estrictamente out-of-sample.
Este paso cierra ese vacío.
---

20.1 Modificar `configs/base.yaml`
Se agrega una nueva sección llamada `splits`.
Aquí se congela la política oficial del esquema temporal de evaluación, incluyendo:
versión del esquema de split;
método: `walk\_forward\_expanding`;
columna temporal oficial;
clave de agrupación por activo;
tamaño inicial de train;
tamaño de validation;
tamaño de test;
stride temporal entre folds;
cantidad mínima de observaciones para train;
política sobre folds parciales;
obligatoriedad del bloque validation;
fuente oficial de thresholds de régimen:
`train\_only`
Con esto, el esquema de evaluación deja de vivir como una decisión implícita en scripts y pasa a ser parte de la configuración oficial del proyecto.
---

20.2 Crear `src/quant\_platform/schemas/split\_record.py`
Se crea el contrato formal del fold temporal.
El schema `SplitRecord` fija los campos mínimos que debe tener cualquier partición walk-forward materializada, incluyendo:
`split\_version`
`split\_id`
`instrument\_id`
`train\_start`
`train\_end`
`validation\_start`
`validation\_end`
`test\_start`
`test\_end`
`train\_rows`
`validation\_rows`
`test\_rows`
`regime\_thresholds\_source`
Esto evita que la semántica de un fold quede difusa o dependa de estructuras ad hoc.
---

20.3 Modificar `src/quant\_platform/schemas/\_\_init\_\_.py`
Se exporta `SplitRecord` desde el agregador de schemas del dominio.
Con esto, el contrato nuevo pasa a ser importable desde:

```text
quant\_platform.schemas
```

## y queda integrado al resto del lenguaje interno del sistema, del mismo modo que `FeatureRow` y `TargetRow`.

20.4 Crear `docs/runbooks/walk\_forward\_splits\_v1.md`
Documento semántico del bloque de splits.
Este runbook deja congelado:
qué significa `walk\_forward\_expanding`;
cómo se ordenan `train`, `validation` y `test`;
por qué el train es expansivo;
por qué el test debe permanecer estrictamente out-of-sample;
cómo se conecta este esquema con `future\_rv\_5d`;
cómo se conecta con `future\_regime\_5d`;
por qué los thresholds de régimen deben calcularse solo con train;
qué propiedades mínimas debe cumplir un fold válido;
qué tipo de metadata debe devolver el builder reusable.
Este archivo actúa como la referencia conceptual del paso 20.
---

20.5 Crear `src/quant\_platform/evaluation/split\_builders.py`
Módulo central del paso 20.
Este archivo implementa la lógica reusable de construcción de folds temporales.
Entre sus responsabilidades están:
validar el input mínimo del builder de splits;
normalizar la columna temporal;
ordenar el dataset por activo y fecha;
construir fronteras temporales para cada fold;
materializar esas fronteras como registros del contrato `SplitRecord`;
construir folds por activo;
filtrar folds que no cumplan mínimos;
validar el output final.
La lógica adoptada es:
`train` con ventana expansiva;
`validation` explícita;
`test` explícito;
stride temporal fijo entre folds;
`regime\_thresholds\_source = train\_only`
Este módulo es el corazón analítico reusable del paso 20.
---

20.6 Modificar `src/quant\_platform/evaluation/\_\_init\_\_.py`
Se exportan las funciones del builder de splits desde el paquete de evaluación.
Con esto, las funciones principales del bloque quedan importables desde:

```text
quant\_platform.evaluation
```

## y el paquete gana una API interna más limpia para scripts y tests.

20.7 Crear `scripts/build\_walk\_forward\_splits.py`
Script operativo que materializa los folds walk-forward.
Para cada símbolo del universo:
descubre el parquet más reciente de `data/targets/`;
lo carga;
construye los splits con `build\_walk\_forward\_splits(...)`;
valida el output;
guarda el parquet resultante en:

```text
artifacts/evaluations/splits/<symbol>/
```

Este script convierte la política temporal definida en YAML y codificada en el builder en una capa persistida de splits reales.
La ejecución efectiva produjo:
7 folds para `SPY`
7 folds para `TLT`
7 folds para `GLD`
7 folds para `HYG`
todos con:
`split\_001` a `split\_007`
`regime\_thresholds\_source = train\_only`
y persistencia correcta bajo `artifacts/evaluations/splits/`.
---

20.8 Crear `scripts/check\_splits\_quality.py`
Checker específico para la nueva capa de splits.
Este script valida, por archivo:
existencia;
no vacío;
columnas esperadas;
validez de intervalos temporales;
no duplicados por `instrument\_id + split\_id`;
secuencia correcta de `split\_id`;
`train\_rows`, `validation\_rows`, `test\_rows` válidos;
crecimiento no decreciente de `train\_rows`;
consistencia de `regime\_thresholds\_source`.
Con esto, la capa de splits deja de ser un output “asumido correcto” y pasa a tener verificación explícita.
---

20.9 Crear `tests/unit/test\_split\_builders.py`
Suite de tests unitarios del builder de splits.
Esta suite verifica, al menos:
que `build\_walk\_forward\_splits(...)` devuelva un DataFrame no vacío;
que los `split\_id` sean secuenciales;
que `train`, `validation` y `test` estén correctamente ordenados en el tiempo;
que `train\_rows` sea no decreciente entre folds;
que `validate\_split\_output(...)` acepte un split válido;
que el builder soporte múltiples instrumentos.
Estos tests protegen el núcleo temporal del sistema frente a regresiones futuras.
---

20.10 Validaciones efectivamente ejecutadas
Durante la ejecución real del paso 20 se comprobaron, con salida satisfactoria:
import correcto de `SplitRecord`;
import correcto de `build\_walk\_forward\_splits`;
materialización efectiva de los folds por activo;
checker de calidad de splits: `PASS`;
tests unitarios del builder de splits: `6 passed`.
Esto confirma que el paso 20 quedó operativo tanto a nivel de:
contrato;
documentación;
lógica reusable;
script de materialización;
script de verificación;
cobertura de pruebas.
---

20.11 Qué se logra al cerrar este bloque
Al finalizar el paso 20, el proyecto ya dispone de una base de evaluación temporal seria y reproducible.
En términos prácticos, ahora existe:

```text
targets
-> walk-forward splits
-> folds persistidos por activo
-> validación temporal explícita
-> tests unitarios del esquema temporal
```

Esto deja listo el terreno para los siguientes bloques del SDD, en particular:
benchmark GARCH;
mapeo del benchmark a regímenes;
entrenamiento del modelo ML;
comparación formal benchmark vs ML.
Sin este paso, cualquier comparación posterior habría sido frágil o potencialmente contaminada por fuga de información.
---

Paso 19B del SSD ----------------------------------------------------------------------
Construcción de la materialización correcta del target discreto de régimen por split, usando los folds
walk-forward ya construidos en el paso 20 y respetando la regla central del proyecto:
los thresholds del régimen deben calcularse únicamente con `train`
`validation` y `test` no deben participar en la estimación de thresholds
la discretización de `future\_rv\_5d` debe hacerse por fold, no de forma global
Este bloque completa la transición entre:
la semántica reusable del régimen construida en 19A
los splits temporales construidos en 20
y la futura comparación benchmark vs ML sobre el mismo target discreto
En otras palabras, 19B convierte la definición abstracta de `future\_regime\_5d` en una capa
materializada, trazable y estadísticamente correcta.
---

19B.1 Congelar la semántica operativa del bloque
Antes de escribir el código, se fijó explícitamente que la capa correcta no sería un
`future\_regime\_5d` global sobre toda la historia, sino una construcción por split.
La semántica congelada fue:
tomar `future\_rv\_5d`
tomar un `split\_id`
usar solo el bloque `train` de ese split para calcular cuantiles
construir thresholds del fold
aplicar esos thresholds a `train`, `validation` y `test`
producir un target discreto por split con trazabilidad total
Con esto se evita la fuga de información que ocurriría si los thresholds se calcularan con toda la historia.
---

19B.2 Crear `docs/runbooks/regime\_targets\_by\_split\_v1.md`
Se creó el runbook semántico del bloque.
Este documento deja por escrito:
qué significa construir régimen “por split”
por qué no debe discretizarse globalmente
cómo se relaciona esta capa con `future\_rv\_5d`
cómo se relaciona con los splits walk-forward
por qué `regime\_thresholds\_source = train\_only`
qué columnas mínimas debe producir esta capa
cómo deben interpretarse `train`, `validation` y `test`
qué restricciones mínimas de validez deben cumplirse
Este archivo actúa como la fuente de verdad conceptual del paso 19B.
---

19B.3 Crear `src/quant\_platform/targets/regime\_split\_builders.py`
Se implementó el módulo reusable que materializa el target discreto por split.
Entre sus responsabilidades quedaron:
validar el `target\_df` de entrada
validar el `split\_df` de entrada
normalizar fechas
construir `dataset\_role` para cada fila del fold:
`train`
`validation`
`test`
calcular thresholds solo desde `train`
aplicar etiquetas de régimen al fold completo
conservar `NaN` cuando `future\_rv\_5d` sea `NaN`
adjuntar metadata del fold, incluyendo:
`split\_id`
`split\_version`
`threshold\_low`
`threshold\_high`
`regime\_thresholds\_source`
Este módulo es el corazón reusable del paso 19B.
---

19B.4 Modificar `src/quant\_platform/targets/\_\_init\_\_.py`
Se exportaron las nuevas funciones del builder de régimen por split desde el paquete `targets`.
Con esto, el módulo pasó a ser importable desde:

```text
quant\_platform.targets
```

## y quedó integrado formalmente a la API interna del dominio.

19B.5 Crear `scripts/build\_regime\_targets\_by\_split.py`
Se construyó el script operativo que materializa la nueva capa de targets discretos por split.
Para cada símbolo del universo:
descubre el parquet más reciente de `data/targets/`
descubre el parquet más reciente de `artifacts/evaluations/splits/`
carga ambos datasets
ejecuta `build\_regime\_targets\_by\_split(...)`
valida la salida
guarda el parquet resultante en:

```text
artifacts/evaluations/regime\_targets/<symbol>/
```

La ejecución real produjo:
`SPY`: 11,442 filas
`TLT`: 11,441 filas
`GLD`: 11,441 filas
`HYG`: 11,441 filas
y en todos los casos:
`split\_001` a `split\_007`
thresholds válidos por fold
`regime\_thresholds\_source = train\_only`
---

19B.6 Crear `scripts/check\_regime\_targets\_by\_split\_quality.py`
Se creó un checker específico para esta nueva capa.
Este script valida, por archivo:
existencia
no vacío
columnas esperadas
validez general del output
ausencia de duplicados por `split\_id + instrument\_id + date`
secuencia correcta de `split\_id`
presencia válida de `dataset\_role`
presencia de bloques `train`, `validation` y `test` cuando corresponde
no nulos en `threshold\_low` y `threshold\_high`
orden correcto `threshold\_low <= threshold\_high`
consistencia de `regime\_thresholds\_source`
Con esto, la nueva capa dejó de ser un output “asumido correcto” y pasó a tener control de calidad explícito.
---

19B.7 Crear `tests/unit/test\_regime\_split\_builders.py`
Se implementó la suite de tests unitarios del nuevo bloque.
Esta suite verifica, al menos:
que un split individual produzca las columnas esperadas
que existan roles `train`, `validation` y `test`
que los thresholds sean constantes dentro de un fold
que el builder multi-split produzca múltiples `split\_id`
que el validador acepte un output correcto
que los `NaN` de `future\_rv\_5d` se preserven como `NaN` en `future\_regime\_5d`
Esto protege la lógica del régimen por split frente a regresiones futuras.
---

19B.8 Validaciones efectivamente ejecutadas
Durante la ejecución real del paso 19B se comprobaron, con salida satisfactoria:
import correcto de `build\_regime\_targets\_by\_split`
materialización efectiva de `future\_regime\_5d` por split
persistencia de artifacts en `artifacts/evaluations/regime\_targets/`
checker de calidad de régimen por split: `PASS`
tests unitarios del builder de régimen por split: `6 passed`
Esto confirma que el bloque quedó operativo tanto a nivel de:
semántica
documentación
lógica reusable
script de construcción
script de verificación
cobertura de pruebas
---

19B.9 Qué se logra al cerrar este bloque
Al finalizar el paso 19B, el proyecto ya no tiene solo una definición reusable del régimen, sino una
capa materializada y estadísticamente correcta de targets discretos por fold.
En términos prácticos, ahora existe:

```text
future\_rv\_5d
-> splits walk-forward
-> thresholds calculados solo en train
-> future\_regime\_5d por split
-> artifacts persistidos
-> checker
-> tests unitarios
```

Esto deja listo el terreno para el siguiente gran bloque del SDD:
benchmark GARCH(1,1) Student-t
mapeo del benchmark a regímenes usando exactamente los mismos thresholds del fold
comparación formal benchmark vs ML sobre el mismo target discreto
Sin este paso, el benchmark y el modelo ML habrían carecido de una definición materializada y rigurosa
del target discreto de régimen.
---

Bitácora técnica · Paso 21
Benchmark GARCH(1,1) Student-t a horizonte 5d
Proyecto: Quant Market Intelligence
Nota editorial
Esta bitácora resume los subpasos ejecutados durante la implementación del paso 21 del SDD: construcción del benchmark GARCH(1,1) Student-t para `future\_rv\_5d`.
Incluye:
verificaciones útiles,
pruebas reales,
materialización operativa,
chequeos de calidad,
y tests unitarios.
Omite:
intentos rotos por copy-paste en terminal,
prompts intermedios,
y texto de asistencia que no produjo estado útil persistente.
---

Estado actual del paso 21
El benchmark GARCH(1,1) Student-t quedó implementado y validado a nivel de:
configuración (`benchmark` en `configs/base.yaml`);
builder reusable (`src/quant\_platform/models/garch\_benchmark.py`);
export del paquete (`src/quant\_platform/models/\_\_init\_\_.py`);
script operativo de materialización (`scripts/build\_garch\_benchmark\_forecasts.py`);
checker de calidad (`scripts/check\_garch\_benchmark\_quality.py`);
tests unitarios (`tests/unit/test\_garch\_benchmark.py`).
Universo validado
SPY
TLT
GLD
HYG
Resultado operativo validado
Se materializaron parquets en:
`artifacts/evaluations/benchmark\_forecasts/spy/`
`artifacts/evaluations/benchmark\_forecasts/tlt/`
`artifacts/evaluations/benchmark\_forecasts/gld/`
`artifacts/evaluations/benchmark\_forecasts/hyg/`
Estado de cierre
Build operativo: PASS
Quality checks: PASS
Unit tests: 5 passed
Commit final del bloque: aún no registrado en este tramo
---

Archivos involucrados en el paso 21
Configuración
`configs/base.yaml`
Código de dominio
`src/quant\_platform/models/garch\_benchmark.py`
`src/quant\_platform/models/\_\_init\_\_.py`
Scripts
`scripts/build\_garch\_benchmark\_forecasts.py`
`scripts/check\_garch\_benchmark\_quality.py`
Tests
`tests/unit/test\_garch\_benchmark.py`
---

Subpasos ejecutados
21.1 · Configurar el bloque `benchmark` en YAML
Propósito: congelar la configuración del benchmark clásico antes de construir el código.
Comando ejecutado (verificación):

```bash
python - <<'PY'
from quant\_platform.services.settings import load\_settings
s = load\_settings()
print(s\["benchmark"])
PY
```

Resultado / efecto útil:
Se confirmó que `settings\["benchmark"]` cargaba correctamente.
Quedaron validados, entre otros:
`benchmark\_name = garch\_11\_student\_t`
`distribution = studentst`
`forecast\_horizon\_days = 5`
`output\_target\_name = future\_rv\_5d`
`score\_roles = \['validation', 'test']`
---

21.2 · Crear el builder reusable del benchmark
Propósito: dejar el núcleo analítico del benchmark en el dominio del proyecto.
Acción efectiva:
Se creó `src/quant\_platform/models/garch\_benchmark.py`.
Verificación ejecutada:

```bash
python - <<'PY'
from quant\_platform.models.garch\_benchmark import (
    build\_benchmark\_input\_df,
    fit\_garch\_on\_train\_returns,
    build\_garch\_benchmark\_forecasts\_by\_split,
)
print(build\_benchmark\_input\_df)
print(fit\_garch\_on\_train\_returns)
print(build\_garch\_benchmark\_forecasts\_by\_split)
PY
```

Resultado / efecto útil:
El módulo importó correctamente.
Quedaron expuestas y utilizables las funciones principales del benchmark.
---

21.3 · Exportar el benchmark desde `models/\_\_init\_\_.py`
Propósito: integrar el nuevo builder al paquete `quant\_platform.models`.
Acción efectiva:
Se modificó `src/quant\_platform/models/\_\_init\_\_.py`.
Verificación ejecutada:

```bash
python - <<'PY'
from quant\_platform.models import (
    build\_benchmark\_input\_df,
    fit\_garch\_on\_train\_returns,
    build\_garch\_benchmark\_forecasts\_by\_split,
)
print(build\_benchmark\_input\_df)
print(fit\_garch\_on\_train\_returns)
print(build\_garch\_benchmark\_forecasts\_by\_split)
PY
```

Resultado / efecto útil:
La exportación agregada funcionó correctamente.
El builder quedó consumible desde el paquete, no solo desde el archivo módulo.
---

21.4 · Primera prueba real en SPY + split\_001
Propósito: validar el benchmark sobre datos reales antes de operacionalizarlo.
Comando ejecutado:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

from quant\_platform.models import build\_garch\_benchmark\_forecasts\_by\_split
from quant\_platform.services.settings import load\_settings

settings = load\_settings()
benchmark\_settings = settings\["benchmark"]

normalized\_dir = Path("data/normalized/spy")
splits\_dir = Path("artifacts/evaluations/splits/spy")

normalized\_files = sorted(normalized\_dir.glob("\*.parquet"))
split\_files = sorted(splits\_dir.glob("\*.parquet"))

if not normalized\_files:
    raise FileNotFoundError(f"No parquet files found in {normalized\_dir}")
if not split\_files:
    raise FileNotFoundError(f"No parquet files found in {splits\_dir}")

normalized\_path = normalized\_files\[0]
split\_path = split\_files\[0]

print(f"\[INFO] normalized\_path = {normalized\_path}")
print(f"\[INFO] split\_path      = {split\_path}")

normalized\_df = pd.read\_parquet(normalized\_path)
split\_df = pd.read\_parquet(split\_path)

split\_df = split\_df.loc\[split\_df\["split\_id"] == "split\_001"].copy()

forecast\_df = build\_garch\_benchmark\_forecasts\_by\_split(
    normalized\_df=normalized\_df,
    split\_df=split\_df,
    settings=benchmark\_settings,
    symbol="SPY",
)
PY
```

Resultado / efecto útil:
La corrida reveló un supuesto incorrecto: el `split\_df` real no estaba en formato daily-level.
Error útil detectado:
faltaban columnas `date` y `dataset\_role` en el parquet de splits.
Esto permitió identificar el ajuste necesario del builder.
---

21.5 · Corregir el builder para soportar splits interval-level
Propósito: adaptar el benchmark al formato real del proyecto.
Acción efectiva:
Se refactorizó `build\_garch\_benchmark\_forecasts\_by\_split(...)`.
Se añadió lógica para:
aceptar `split\_df` interval-level;
expandir internamente a daily-level;
soportar columnas reales del parquet de splits como:
`train\_start`
`train\_end`
`validation\_start`
`validation\_end`
`test\_start`
`test\_end`
Verificación ejecutada tras la corrección:

```bash
python - <<'PY'
from quant\_platform.models.garch\_benchmark import build\_garch\_benchmark\_forecasts\_by\_split
print(build\_garch\_benchmark\_forecasts\_by\_split)
PY
```

Resultado / efecto útil:
La función quedó nuevamente importable.
El builder pasó de una suposición incorrecta a una implementación compatible con los splits reales del proyecto.
---

21.6 · Segunda prueba real en SPY + split\_001 (ya corregida)
Propósito: confirmar que el builder corregido generaba forecasts reales y coherentes.
Comando ejecutado:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

from quant\_platform.models import build\_garch\_benchmark\_forecasts\_by\_split
from quant\_platform.services.settings import load\_settings

settings = load\_settings()
benchmark\_settings = settings\["benchmark"]

normalized\_path = sorted(Path("data/normalized/spy").glob("\*.parquet"))\[0]
split\_path = sorted(Path("artifacts/evaluations/splits/spy").glob("\*.parquet"))\[0]

print(f"\[INFO] normalized\_path = {normalized\_path}")
print(f"\[INFO] split\_path      = {split\_path}")

normalized\_df = pd.read\_parquet(normalized\_path)
split\_df = pd.read\_parquet(split\_path)

print("\\n\[INFO] split\_df columns:")
print(split\_df.columns.tolist())

split\_df = split\_df.loc\[split\_df\["split\_id"] == "split\_001"].copy()

forecast\_df = build\_garch\_benchmark\_forecasts\_by\_split(
    normalized\_df=normalized\_df,
    split\_df=split\_df,
    settings=benchmark\_settings,
    symbol="SPY",
)

print("\\n\[INFO] forecast\_df shape:")
print(forecast\_df.shape)

print("\\n\[INFO] head:")
print(forecast\_df.head(10).to\_string(index=False))

print("\\n\[INFO] tail:")
print(forecast\_df.tail(10).to\_string(index=False))

print("\\n\[INFO] roles:")
print(forecast\_df\["dataset\_role"].value\_counts(dropna=False))

print("\\n\[INFO] split ids:")
print(forecast\_df\["split\_id"].value\_counts(dropna=False))

print("\\n\[INFO] params summary:")
print(forecast\_df\[\["yhat\_future\_rv\_5d", "omega", "alpha\_1", "beta\_1", "nu"]].describe())
PY
```

Resultado / efecto útil:
Se confirmó el esquema real del parquet de splits:
`split\_version`
`split\_id`
`instrument\_id`
`train\_start`
`train\_end`
`validation\_start`
`validation\_end`
`test\_start`
`test\_end`
`train\_rows`
`validation\_rows`
`test\_rows`
`regime\_thresholds\_source`
Se generó `forecast\_df` correcto para `split\_001`:
filas: `251`
roles:
`validation = 125`
`test = 126`
Parámetros obtenidos en SPY/split\_001:
`omega ≈ 0.046816`
`alpha\_1 ≈ 0.22382`
`beta\_1 ≈ 0.756172`
`nu ≈ 5.904675`
El benchmark quedó validado en una vertical slice real.
---

21.7 · Crear y validar el script operativo del benchmark
Propósito: materializar forecasts benchmark para todo el universo y todos los folds.
Acción efectiva:
Se creó `scripts/build\_garch\_benchmark\_forecasts.py`.
Verificación de import:

```bash
python - <<'PY'
from scripts.build\_garch\_benchmark\_forecasts import main
print(main)
PY
```

Ejecución operativa completa:

```bash
python scripts/build\_garch\_benchmark\_forecasts.py
```

Resultado / efecto útil:
Se materializaron forecasts benchmark en los 4 activos del universo.
Resultados reportados:
SPY
output:
`artifacts/evaluations/benchmark\_forecasts/spy/spy\_2018-01-02\_2026-04-02\_garch\_11\_student\_t\_v1.parquet`
filas: `1755`
splits: `7`
roles:
`validation = 876`
`test = 879`
TLT
output:
`artifacts/evaluations/benchmark\_forecasts/tlt/tlt\_2018-01-02\_2026-04-02\_garch\_11\_student\_t\_v1.parquet`
filas: `1754`
splits: `7`
roles:
`validation = 876`
`test = 878`
GLD
output:
`artifacts/evaluations/benchmark\_forecasts/gld/gld\_2018-01-02\_2026-04-02\_garch\_11\_student\_t\_v1.parquet`
filas: `1754`
splits: `7`
roles:
`validation = 876`
`test = 878`
HYG
output:
`artifacts/evaluations/benchmark\_forecasts/hyg/hyg\_2018-01-02\_2026-04-02\_garch\_11\_student\_t\_v1.parquet`
filas: `1754`
splits: `7`
roles:
`validation = 876`
`test = 878`
El bloque cerró con:
`GARCH BENCHMARK BUILD: PASS`
---

21.8 · Crear y ejecutar el checker de calidad
Propósito: validar que la nueva capa benchmark\_forecasts fuera consistente con los splits y con el contrato esperado.
Acción efectiva:
Se creó `scripts/check\_garch\_benchmark\_quality.py`.
Comando ejecutado:

```bash
python scripts/check\_garch\_benchmark\_quality.py
```

Resultado / efecto útil:
Validación correcta de los 4 parquets materializados:
SPY OK
TLT OK
GLD OK
HYG OK
El checker cerró con:
`GARCH BENCHMARK QUALITY CHECKS: PASS`
---

21.9 · Crear y ejecutar tests unitarios del benchmark
Propósito: cubrir el bloque con pruebas automáticas mínimas pero reales.
Acción efectiva:
Se creó `tests/unit/test\_garch\_benchmark.py`.
Comando ejecutado:

```bash
pytest tests/unit/test\_garch\_benchmark.py -q
```

Resultado / efecto útil:
Resultado final:
`5 passed in 2.88s`
Cobertura alcanzada:
construcción de retornos escalados;
trayectoria forecast de varianzas;
expansión de splits interval-level;
ajuste básico GARCH;
generación de forecasts scored-only por split.
---

Resumen ejecutivo del paso 21
Qué quedó construido
benchmark GARCH(1,1) Student-t reusable;
adaptación correcta a splits interval-level del proyecto;
materialización completa para SPY / TLT / GLD / HYG;
checker de calidad;
tests unitarios.
Qué quedó validado
build operativo: PASS
quality checks: PASS
tests unitarios: PASS
Qué falta para cierre administrativo
runbook `docs/runbooks/garch\_benchmark\_v1.md`
commit final del bloque
actualización de la bitácora general del proyecto
Siguiente paso natural
Paso 22 del SDD: mapear el forecast continuo del benchmark a régimen discreto usando umbrales train-only.
---

Bitácora técnica · Paso 22
Benchmark → régimen discreto con thresholds train-only
Proyecto: Quant Market Intelligence
Nota editorial
Esta entrada resume el trabajo ejecutado en el paso 22 del SDD: convertir el forecast continuo del benchmark GARCH (`yhat\_future\_rv\_5d`) en una predicción discreta de régimen (`calm`, `normal`, `stress`) usando los thresholds ya materializados bajo política `train\_only`.
Incluye:
subpasos efectivos;
el porqué de cada bloque;
comandos realmente ejecutados;
resultados útiles obtenidos.
Omite:
intentos fallidos por copy-paste;
texto de guía no persistente;
y pasos aún no cerrados con commit en este tramo.
---

Objetivo del paso 22
Tomar la salida continua del benchmark del paso 21 y traducirla a régimen discreto sin redefinir la semántica del target.
Principio técnico aplicado
Los thresholds no se recalculan desde benchmark forecasts.  
Se heredan desde la capa ya materializada de `regime\_targets\_by\_split`, donde quedaron congelados con fuente:
`regime\_thresholds\_source = train\_only`
Esto asegura consistencia entre:
el target discreto real del proyecto;
la predicción discreta del benchmark;
y la evaluación posterior.
---

Estado actual del paso 22
Construido
configuración `benchmark\_regime` en `configs/base.yaml`
builder reusable:
`src/quant\_platform/models/benchmark\_regime.py`
export del paquete:
`src/quant\_platform/models/\_\_init\_\_.py`
script operativo:
`scripts/build\_benchmark\_regime\_predictions.py`
checker de calidad:
`scripts/check\_benchmark\_regime\_quality.py`
tests unitarios:
`tests/unit/test\_benchmark\_regime.py`
Materializado
`artifacts/evaluations/benchmark\_regimes/spy/`
`artifacts/evaluations/benchmark\_regimes/tlt/`
`artifacts/evaluations/benchmark\_regimes/gld/`
`artifacts/evaluations/benchmark\_regimes/hyg/`
Resultado de validación
build operativo: PASS
quality checks: PASS
tests unitarios: 5 passed
Cierre administrativo
runbook del paso 22 generado externamente en `.md`
commit final del paso 22: no registrado aún en este tramo
---

Subpasos ejecutados
22.1 · Congelar configuración `benchmark\_regime`
Por qué se hizo:  
Antes de construir el bloque, había que fijar explícitamente en configuración:
qué columna continua del benchmark se usaría;
qué columnas de thresholds y targets se consumirían;
y qué labels discretos serían la salida.
Acción efectiva:
se añadió la sección `benchmark\_regime` en `configs/base.yaml`.
Comando de verificación ejecutado:

```bash
python - <<'PY'
from quant\_platform.services.settings import load\_settings
s = load\_settings()
print(s\["benchmark\_regime"])
PY
```

Resultado / efecto útil:
la nueva sección cargó correctamente;
quedaron congeladas, entre otras:
`forecast\_value\_column = yhat\_future\_rv\_5d`
`output\_regime\_column = yhat\_future\_regime\_5d`
`threshold\_low\_column = threshold\_low`
`threshold\_high\_column = threshold\_high`
`expected\_threshold\_source = train\_only`
---

22.2 · Inspeccionar el contrato real de los inputs
Por qué se hizo:  
Antes de escribir el builder, era necesario confirmar el esquema exacto de:
benchmark forecasts
regime targets by split
para evitar suposiciones incorrectas sobre nombres de columnas.
Comando ejecutado:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

benchmark\_path = sorted(Path("artifacts/evaluations/benchmark\_forecasts/spy").glob("\*.parquet"))\[0]
regime\_path = sorted(Path("artifacts/evaluations/regime\_targets/spy").glob("\*.parquet"))\[0]

benchmark\_df = pd.read\_parquet(benchmark\_path)
regime\_df = pd.read\_parquet(regime\_path)

print("\[BENCHMARK PATH]")
print(benchmark\_path)

print("
\[BENCHMARK COLUMNS]")
print(benchmark\_df.columns.tolist())

print("
\[REGIME TARGETS PATH]")
print(regime\_path)

print("
\[REGIME TARGETS COLUMNS]")
print(regime\_df.columns.tolist())

print("
\[REGIME TARGETS HEAD]")
print(regime\_df.head(10).to\_string(index=False))
PY
```

Resultado / efecto útil:
se confirmó que el benchmark continuo contenía:
`yhat\_future\_rv\_5d`
se confirmó que la capa `regime\_targets\_by\_split` contenía:
`future\_rv\_5d`
`future\_regime\_5d`
`threshold\_low`
`threshold\_high`
`regime\_thresholds\_source`
Esto dejó bloqueada la interfaz real del paso 22.
---

22.3 · Crear el builder reusable del benchmark discreto
Por qué se hizo:  
El mapeo benchmark → régimen no debía quedar como lógica embebida en un script.  
Debía existir como función reusable del dominio.
Acción efectiva:
se creó `src/quant\_platform/models/benchmark\_regime.py`
Responsabilidad del módulo:
validar settings;
verificar constancia de thresholds por `split\_id`;
unir benchmark forecasts con regime targets reales;
asignar `yhat\_future\_regime\_5d`;
devolver una capa lista para evaluación posterior.
Comando de verificación ejecutado:

```bash
python - <<'PY'
from quant\_platform.models.benchmark\_regime import build\_benchmark\_regime\_predictions
print(build\_benchmark\_regime\_predictions)
PY
```

Resultado / efecto útil:
el módulo importó correctamente;
el builder quedó accesible y funcional.
---

22.4 · Exportar el builder desde `models/\_\_init\_\_.py`
Por qué se hizo:  
Para dejar el nuevo builder integrado al paquete y no depender de imports al archivo concreto.
Acción efectiva:
se modificó `src/quant\_platform/models/\_\_init\_\_.py`
Comando de verificación ejecutado:

```bash
python - <<'PY'
from quant\_platform.models import (
    build\_garch\_benchmark\_forecasts\_by\_split,
    build\_benchmark\_regime\_predictions,
)
print(build\_garch\_benchmark\_forecasts\_by\_split)
print(build\_benchmark\_regime\_predictions)
PY
```

Resultado / efecto útil:
la exportación agregada funcionó correctamente;
el nuevo builder quedó integrado al paquete del proyecto.
---

22.5 · Primera prueba real en SPY
Por qué se hizo:  
Antes de operar todo el universo, había que validar que la lógica realmente:
uniera benchmark forecasts con targets reales;
arrastrara thresholds correctos;
y produjera `yhat\_future\_regime\_5d`.
Comando ejecutado:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

from quant\_platform.models import build\_benchmark\_regime\_predictions
from quant\_platform.services.settings import load\_settings

settings = load\_settings()
benchmark\_regime\_settings = settings\["benchmark\_regime"]

benchmark\_path = sorted(Path("artifacts/evaluations/benchmark\_forecasts/spy").glob("\*.parquet"))\[0]
regime\_path = sorted(Path("artifacts/evaluations/regime\_targets/spy").glob("\*.parquet"))\[0]

print(f"\[INFO] benchmark\_path = {benchmark\_path}")
print(f"\[INFO] regime\_path    = {regime\_path}")

benchmark\_df = pd.read\_parquet(benchmark\_path)
regime\_df = pd.read\_parquet(regime\_path)

benchmark\_regime\_df = build\_benchmark\_regime\_predictions(
    benchmark\_forecast\_df=benchmark\_df,
    regime\_targets\_by\_split\_df=regime\_df,
    settings=benchmark\_regime\_settings,
)

print("
\[INFO] benchmark\_regime\_df shape:")
print(benchmark\_regime\_df.shape)

print("
\[INFO] columns:")
print(benchmark\_regime\_df.columns.tolist())

print("
\[INFO] head:")
print(benchmark\_regime\_df.head(10).to\_string(index=False))

print("
\[INFO] predicted regime counts:")
print(benchmark\_regime\_df\["yhat\_future\_regime\_5d"].value\_counts(dropna=False))

print("
\[INFO] actual regime counts:")
print(benchmark\_regime\_df\["future\_regime\_5d"].value\_counts(dropna=False))

print("
\[INFO] split ids:")
print(benchmark\_regime\_df\["split\_id"].value\_counts(dropna=False).sort\_index())

print("
\[INFO] threshold source values:")
print(benchmark\_regime\_df\["regime\_thresholds\_source"].value\_counts(dropna=False))
PY
```

Resultado / efecto útil:
se generó `benchmark\_regime\_df` con:
`1755` filas
`23` columnas
aparecieron las columnas nuevas esperadas:
`future\_rv\_5d`
`future\_regime\_5d`
`threshold\_low`
`threshold\_high`
`regime\_thresholds\_source`
`yhat\_future\_regime\_5d`
se confirmó:
cobertura de los `7 split\_id`
`regime\_thresholds\_source = train\_only` en todas las filas
Observación analítica temprana útil:
el benchmark en SPY predijo muy pocos `calm` (`8`), con fuerte peso en `normal` y `stress`.
esto no bloquea el pipeline; simplemente deja una señal para la evaluación posterior.
---

22.6 · Crear el script operativo para todo el universo
Por qué se hizo:  
La lógica del builder ya estaba validada en SPY; el siguiente paso natural era materializar una capa benchmark\_regimes para todo el universo y todos los splits.
Acción efectiva:
se creó `scripts/build\_benchmark\_regime\_predictions.py`
Comando ejecutado:

```bash
python scripts/build\_benchmark\_regime\_predictions.py
```

Resultado / efecto útil:
se materializó la nueva capa `benchmark\_regimes` para los 4 activos.
SPY
output:
`artifacts/evaluations/benchmark\_regimes/spy/spy\_2022-01-03\_2025-12-31\_benchmark\_regimes\_v1.parquet`
filas: `1755`
splits: `7`
predicted regimes:
`normal = 1053`
`stress = 694`
`calm = 8`
TLT
output:
`artifacts/evaluations/benchmark\_regimes/tlt/tlt\_2022-01-03\_2025-12-31\_benchmark\_regimes\_v1.parquet`
filas: `1754`
splits: `7`
predicted regimes:
`stress = 963`
`normal = 779`
`calm = 12`
GLD
output:
`artifacts/evaluations/benchmark\_regimes/gld/gld\_2022-01-03\_2025-12-31\_benchmark\_regimes\_v1.parquet`
filas: `1754`
splits: `7`
predicted regimes:
`stress = 1224`
`normal = 530`
HYG
output:
`artifacts/evaluations/benchmark\_regimes/hyg/hyg\_2022-01-03\_2025-12-31\_benchmark\_regimes\_v1.parquet`
filas: `1754`
splits: `7`
predicted regimes:
`normal = 984`
`stress = 758`
`calm = 12`
El bloque cerró con:
`BENCHMARK REGIME BUILD: PASS`
---

22.7 · Crear y ejecutar el checker de calidad
Por qué se hizo:  
La nueva capa debía validarse formalmente antes de considerarla apta para evaluación y comparación posterior.
Acción efectiva:
se creó `scripts/check\_benchmark\_regime\_quality.py`
Comando ejecutado:

```bash
python scripts/check\_benchmark\_regime\_quality.py
```

Resultado / efecto útil:
validación correcta de:
SPY
TLT
GLD
HYG
El checker cerró con:
`BENCHMARK REGIME QUALITY CHECKS: PASS`
---

22.8 · Crear y ejecutar tests unitarios
Por qué se hizo:  
Había que cubrir con pruebas automáticas mínimas los puntos más sensibles del bloque:
asignación de labels;
extracción de thresholds;
error ante inconsistencia de thresholds;
merge correcto con targets reales;
validación de forecasts finitos.
Acción efectiva:
se creó `tests/unit/test\_benchmark\_regime.py`
Comando ejecutado:

```bash
pytest tests/unit/test\_benchmark\_regime.py -q
```

Resultado / efecto útil:
resultado final:
`5 passed in 2.55s`
---

22.9 · Generar el runbook del bloque
Por qué se hizo:  
El bloque ya había quedado técnicamente operativo; faltaba dejarlo documentado de forma usable.
Acción efectiva:
se generó externamente el archivo:
`benchmark\_regime\_v1.md`
Estado real en este tramo:
el runbook quedó generado en `.md`
no quedó evidenciado en este tramo el comando de copia al repo ni el commit final del paso 22
---

Resumen ejecutivo del paso 22
Qué quedó construido
mapeo benchmark continuo → régimen discreto;
herencia correcta de thresholds `train\_only`;
unión del benchmark con targets reales por:
`split\_id`
`date`
`dataset\_role`
nueva capa materializada `benchmark\_regimes`;
checker y tests unitarios.
Qué quedó validado
build operativo: PASS
quality checks: PASS
tests unitarios: PASS
Qué falta para cierre administrativo
copiar / confirmar runbook dentro de `docs/runbooks/`
commit final del bloque
actualización de la bitácora general del proyecto
Siguiente paso natural
Pasar al siguiente bloque del SSD orientado a explotación evaluativa / comparativa de estas salidas.
---

# Bitácora técnica · Paso 23

## Pipeline ML base con XGBoost Regressor

**Proyecto:** Quant Market Intelligence

### Nota editorial

Esta entrada resume el trabajo ejecutado en el paso 23 del SDD: construcción del primer pipeline ML del proyecto para predecir `future\_rv\_5d` con `XGBoost Regressor`.

Incluye:

* subpasos efectivos;
* el porqué de cada bloque;
* comandos realmente ejecutados;
* resultados útiles obtenidos.

Omite:

* copy-paste rotos en terminal;
* texto de guía no persistente;
* y el commit final, que no quedó registrado todavía en este tramo.

\---

## Objetivo del paso 23

Construir el primer modelo ML serio del proyecto para forecast continuo:

* modelo: `XGBoost Regressor`
* target: `future\_rv\_5d`
* features: `features\_context`
* scoring OOS: `validation` y `test`

### Principio técnico aplicado

En esta fase no se abrió todavía el classifier.  
Se priorizó el regressor porque:

* el target principal del producto es continuo;
* la comparación con el benchmark GARCH se hace primero sobre `future\_rv\_5d`;
* y eso habilita una evaluación limpia en el paso siguiente.

\---

## Estado actual del paso 23

### Construido

* configuración `ml\_regressor` en `configs/base.yaml`
* builder reusable:

  * `src/quant\_platform/models/xgboost\_regressor.py`
* export del paquete:

  * `src/quant\_platform/models/\_\_init\_\_.py`
* script operativo:

  * `scripts/build\_xgboost\_regressor\_forecasts.py`
* checker de calidad:

  * `scripts/check\_xgboost\_regressor\_quality.py`
* tests unitarios:

  * `tests/unit/test\_xgboost\_regressor.py`

### Materializado

* `artifacts/evaluations/ml\_forecasts/spy/`
* `artifacts/evaluations/ml\_forecasts/tlt/`
* `artifacts/evaluations/ml\_forecasts/gld/`
* `artifacts/evaluations/ml\_forecasts/hyg/`

### Persistido

* modelos serializados por split en:

  * `artifacts/models/xgboost\_regressor/{symbol}/`

### Resultado de validación

* build operativo: **PASS**
* quality checks: **PASS**
* tests unitarios: **5 passed**

### Cierre administrativo

* runbook del paso 23 generado externamente en `.md`
* commit final del paso 23: **no registrado aún en este tramo**

\---

# Subpasos ejecutados

## 23.1 · Congelar configuración `ml\_regressor`

**Por qué se hizo:**  
Antes de construir el trainer ML había que dejar fija la definición operacional de:

* modelo;
* target;
* fuente de features;
* score roles;
* hiperparámetros base;
* política de missing values y persistencia.

**Acción efectiva:**

* se añadió la sección `ml\_regressor` en `configs/base.yaml`.

**Comando de verificación ejecutado:**

```bash
python - <<'PY'
from quant\_platform.services.settings import load\_settings
s = load\_settings()
print(s\["ml\_regressor"])
PY
```

**Resultado / efecto útil:**

* la configuración cargó correctamente;
* quedaron congelados, entre otros:

  * `feature\_source = features\_context`
  * `target\_column = future\_rv\_5d`
  * `objective = reg:squarederror`
  * `eval\_metric = rmse`
  * `early\_stopping\_rounds = 50`

\---

## 23.2 · Inspeccionar el contrato real de features, targets y splits

**Por qué se hizo:**  
Antes de escribir el trainer, era necesario verificar:

* qué columnas reales existían en `features\_context`;
* cómo venía el target continuo;
* y cómo estaba estructurado el parquet de splits.

**Comando ejecutado:**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

features\_path = sorted(Path("data/features\_context/spy").glob("\*.parquet"))\[0]
targets\_path = sorted(Path("data/targets/spy").glob("\*.parquet"))\[0]
splits\_path = sorted(Path("artifacts/evaluations/splits/spy").glob("\*.parquet"))\[0]

features\_df = pd.read\_parquet(features\_path)
targets\_df = pd.read\_parquet(targets\_path)
splits\_df = pd.read\_parquet(splits\_path)

print("\[FEATURES PATH]")
print(features\_path)

print("\\n\[FEATURES COLUMNS]")
print(features\_df.columns.tolist())

print("\\n\[FEATURES HEAD]")
print(features\_df.head(5).to\_string(index=False))

print("\\n\[TARGETS PATH]")
print(targets\_path)

print("\\n\[TARGETS COLUMNS]")
print(targets\_df.columns.tolist())

print("\\n\[TARGETS HEAD]")
print(targets\_df.head(5).to\_string(index=False))

print("\\n\[SPLITS PATH]")
print(splits\_path)

print("\\n\[SPLITS COLUMNS]")
print(splits\_df.columns.tolist())

print("\\n\[SPLITS HEAD]")
print(splits\_df.head(5).to\_string(index=False))
PY
```

**Resultado / efecto útil:**

* se confirmó que `features\_context` tenía:

  * features base
  * y señales `ctx\_\*`
* se confirmó que el target continuo era:

  * `future\_rv\_5d`
* se confirmó que los splits seguían en formato interval-level:

  * `train\_start`
  * `train\_end`
  * `validation\_start`
  * `validation\_end`
  * `test\_start`
  * `test\_end`

Esto dejó bloqueada la interfaz real del paso 23.

\---

## 23.3 · Crear el builder reusable del XGBoost Regressor

**Por qué se hizo:**  
El entrenamiento ML no debía quedar incrustado en un script operativo.
Debía existir como bloque reusable del dominio.

**Acción efectiva:**

* se creó `src/quant\_platform/models/xgboost\_regressor.py`

**Responsabilidades del módulo:**

* construir el panel features + target;
* expandir splits interval-level a daily-level;
* seleccionar columnas de features;
* entrenar `XGBRegressor` por split;
* usar `validation` para early stopping;
* generar forecasts OOS sobre `validation` y `test`;
* devolver forecasts + modelos entrenados.

**Comando de verificación ejecutado:**

```bash
python - <<'PY'
from quant\_platform.models.xgboost\_regressor import (
    build\_ml\_regressor\_input\_panel,
    build\_xgboost\_regressor\_forecasts\_by\_split,
)
print(build\_ml\_regressor\_input\_panel)
print(build\_xgboost\_regressor\_forecasts\_by\_split)
PY
```

**Resultado / efecto útil:**

* el módulo importó correctamente;
* el núcleo del trainer quedó disponible.

\---

## 23.4 · Exportar el trainer desde `models/\_\_init\_\_.py`

**Por qué se hizo:**  
Para dejar el trainer integrado al paquete y no depender de imports al archivo concreto.

**Acción efectiva:**

* se modificó `src/quant\_platform/models/\_\_init\_\_.py`

**Comando de verificación ejecutado:**

```bash
python - <<'PY'
from quant\_platform.models import (
    build\_ml\_regressor\_input\_panel,
    build\_xgboost\_regressor\_forecasts\_by\_split,
)
print(build\_ml\_regressor\_input\_panel)
print(build\_xgboost\_regressor\_forecasts\_by\_split)
PY
```

**Resultado / efecto útil:**

* la exportación agregada funcionó correctamente;
* el trainer quedó integrado al paquete del proyecto.

\---

## 23.5 · Primera prueba real en SPY + split\_001

**Por qué se hizo:**  
Antes de ejecutar el build completo del universo, había que verificar que el trainer:

* uniera bien features + target;
* expandiera bien los splits;
* entrenara sin romperse;
* generara forecasts OOS reales;
* y produjera metadata útil del modelo.

**Comando ejecutado:**

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

from quant\_platform.models import build\_xgboost\_regressor\_forecasts\_by\_split
from quant\_platform.services.settings import load\_settings

settings = load\_settings()
ml\_settings = settings\["ml\_regressor"]

features\_path = sorted(Path("data/features\_context/spy").glob("\*.parquet"))\[0]
targets\_path = sorted(Path("data/targets/spy").glob("\*.parquet"))\[0]
splits\_path = sorted(Path("artifacts/evaluations/splits/spy").glob("\*.parquet"))\[0]

print(f"\[INFO] features\_path = {features\_path}")
print(f"\[INFO] targets\_path  = {targets\_path}")
print(f"\[INFO] splits\_path   = {splits\_path}")

features\_df = pd.read\_parquet(features\_path)
targets\_df = pd.read\_parquet(targets\_path)
splits\_df = pd.read\_parquet(splits\_path)

forecast\_df, split\_models = build\_xgboost\_regressor\_forecasts\_by\_split(
    features\_df=features\_df,
    targets\_df=targets\_df,
    split\_df=splits\_df.loc\[splits\_df\["split\_id"] == "split\_001"].copy(),
    settings=ml\_settings,
    symbol="SPY",
)

print("\\n\[INFO] forecast\_df shape:")
print(forecast\_df.shape)

print("\\n\[INFO] columns:")
print(forecast\_df.columns.tolist())

print("\\n\[INFO] head:")
print(forecast\_df.head(10).to\_string(index=False))

print("\\n\[INFO] tail:")
print(forecast\_df.tail(10).to\_string(index=False))

print("\\n\[INFO] roles:")
print(forecast\_df\["dataset\_role"].value\_counts(dropna=False))

print("\\n\[INFO] split ids:")
print(forecast\_df\["split\_id"].value\_counts(dropna=False))

print("\\n\[INFO] prediction summary:")
print(forecast\_df\[\["yhat\_future\_rv\_5d", "future\_rv\_5d"]].describe())

print("\\n\[INFO] trained models:")
print(len(split\_models))
print(split\_models\[0].keys())
print(
    {
        "split\_id": split\_models\[0]\["split\_id"],
        "n\_train": split\_models\[0]\["n\_train"],
        "n\_validation": split\_models\[0]\["n\_validation"],
        "best\_iteration": split\_models\[0]\["best\_iteration"],
        "best\_score": split\_models\[0]\["best\_score"],
        "feature\_count": len(split\_models\[0]\["feature\_columns"]),
    }
)
PY
```

**Resultado / efecto útil:**

* se generó `forecast\_df` correcto para `split\_001`:

  * filas: `251`
  * roles:

    * `validation = 125`
    * `test = 126`
* se confirmó metadata útil del entrenamiento:

  * `n\_train = 1008`
  * `n\_validation = 125`
  * `feature\_count = 33`
  * `best\_iteration = 25`
  * `best\_score ≈ 0.09568`

Esto validó el trainer sobre una vertical slice real.

\---

## 23.6 · Crear y validar el script operativo completo

**Por qué se hizo:**  
El builder ya estaba validado sobre SPY; el siguiente paso era materializar forecasts ML y modelos serializados para todo el universo y todos los splits.

**Acción efectiva:**

* se creó `scripts/build\_xgboost\_regressor\_forecasts.py`

**Comando de verificación ejecutado:**

```bash
python - <<'PY'
from scripts.build\_xgboost\_regressor\_forecasts import main
print(main)
PY
```

**Ejecución operativa completa:**

```bash
python scripts/build\_xgboost\_regressor\_forecasts.py
```

**Resultado / efecto útil:**

* se materializó la nueva capa `ml\_forecasts` para los 4 activos;
* y además se persistieron modelos serializados por split.

### SPY

* output:

  * `artifacts/evaluations/ml\_forecasts/spy/spy\_2022-01-03\_2025-12-31\_xgboost\_regressor\_v1.parquet`
* filas: `1755`
* splits: `7`
* roles:

  * `validation = 876`
  * `test = 879`

### TLT

* output:

  * `artifacts/evaluations/ml\_forecasts/tlt/tlt\_2022-01-03\_2025-12-31\_xgboost\_regressor\_v1.parquet`
* filas: `1754`
* splits: `7`
* roles:

  * `validation = 876`
  * `test = 878`

### GLD

* output:

  * `artifacts/evaluations/ml\_forecasts/gld/gld\_2022-01-03\_2025-12-31\_xgboost\_regressor\_v1.parquet`
* filas: `1754`
* splits: `7`
* roles:

  * `validation = 876`
  * `test = 878`

### HYG

* output:

  * `artifacts/evaluations/ml\_forecasts/hyg/hyg\_2022-01-03\_2025-12-31\_xgboost\_regressor\_v1.parquet`
* filas: `1754`
* splits: `7`
* roles:

  * `validation = 876`
  * `test = 878`
* El bloque cerró con:

  * `XGBOOST REGRESSOR BUILD: PASS`

\---

## 23.7 · Crear y ejecutar el checker de calidad

**Por qué se hizo:**  
La nueva capa ML debía validarse antes de quedar lista para comparación formal con el benchmark.

**Acción efectiva:**

* se creó `scripts/check\_xgboost\_regressor\_quality.py`

**Comando ejecutado:**

```bash
python scripts/check\_xgboost\_regressor\_quality.py
```

**Resultado / efecto útil:**

* validación correcta de:

  * SPY
  * TLT
  * GLD
  * HYG
* El checker cerró con:

  * `XGBOOST REGRESSOR QUALITY CHECKS: PASS`

\---

## 23.8 · Crear y ejecutar tests unitarios

**Por qué se hizo:**  
Había que cubrir con pruebas automáticas mínimas los puntos sensibles del bloque:

* construcción del panel;
* forecasts OOS por split;
* metadata del modelo;
* manejo de missing values;
* y error cuando validation no contiene targets utilizables.

**Acción efectiva:**

* se creó `tests/unit/test\_xgboost\_regressor.py`

**Primer resultado observado:**

```bash
pytest tests/unit/test\_xgboost\_regressor.py -q
```

* resultado:

  * `1 failed, 4 passed`

**Causa útil detectada:**

* el test final no vaciaba realmente el conjunto de validation;
* al usar una ventana de un día, el split seguía teniendo una fila válida.

**Corrección efectiva aplicada:**

* se rehízo el test para anular los targets dentro de validation, en lugar de colapsar la ventana.

**Comando final ejecutado:**

```bash
pytest tests/unit/test\_xgboost\_regressor.py -q
```

**Resultado / efecto útil:**

* resultado final:

  * `5 passed in 3.64s`

\---

## 23.9 · Generar el runbook del bloque

**Por qué se hizo:**  
El bloque ya había quedado técnicamente operativo; faltaba dejarlo documentado de forma reusable.

**Acción efectiva:**

* se generó externamente el archivo:

  * `xgboost\_regressor\_v1.md`

**Estado real en este tramo:**

* el runbook quedó generado en `.md`
* no quedó evidenciado en este tramo el comando de copia al repo ni el commit final del paso 23

\---

# Resumen ejecutivo del paso 23

## Qué quedó construido

* trainer reusable con XGBoost Regressor;
* panel features + target;
* expansión de splits interval-level;
* forecasts OOS por split;
* persistencia de modelos;
* nueva capa `ml\_forecasts`;
* checker y tests unitarios.

## Qué quedó validado

* build operativo: PASS
* quality checks: PASS
* tests unitarios: PASS

## Qué falta para cierre administrativo

* copiar / confirmar runbook dentro de `docs/runbooks/`
* commit final del bloque
* actualización de la bitácora general del proyecto

## Siguiente paso natural

Pasar a la comparación formal benchmark vs ML en el siguiente bloque del SDD.

\-------------------------------------------------------------------------------------------

# tácora personal de desarrollo — Paso 24: model comparison

## Alcance de esta bitácora

Esta bitácora resume los comandos ejecutados durante esta sesión de trabajo para construir, validar y cerrar el Paso 24 del proyecto: evaluación formal benchmark vs ML.

Incluye:

* comandos de branch y verificación;
* comandos Python de inspección real de contratos;
* pruebas del builder reusable;
* materialización operativa de la capa;
* checker de calidad;
* tests unitarios.

No incluye:

* prompts del chat;
* comandos de edición de archivos no ejecutados explícitamente en terminal;
* commits todavía no realizados en esta sesión.

\---

## Estado de cierre de la sesión

Quedó logrado:

* `configs/base.yaml` con sección `evaluation` cargando correctamente;
* builder reusable de comparación en `src/quant\\\_platform/evaluation/model\\\_comparison.py`;
* script operativo `scripts/evaluate\\\_models.py`;
* checker `scripts/check\\\_model\\\_comparison\\\_quality.py`;
* tests unitarios `tests/unit/test\\\_model\\\_comparison.py`;
* artefactos materializados para `GLD`, `HYG`, `SPY` y `TLT`;
* checker en PASS;
* tests unitarios en PASS.

\---

## Secuencia cronológica de comandos ejecutados

### 1\) Crear rama del bloque

```bash
git status
git switch -c feat/model-evaluation-v1
```

**Resultado útil**

* Se confirmó working tree limpio.
* Se abrió la rama `feat/model-evaluation-v1` a partir de `feat/benchmark-garch-v1`.

\---

### 2\) Verificar que la configuración `evaluation` carga correctamente

```bash
python - <<'PY'
from quant\\\_platform.services.settings import load\\\_settings

s = load\\\_settings()
e = s\\\["evaluation"]

print("version:", e\\\["version"])
print("primary\\\_metric:", e\\\["continuous"]\\\["primary\\\_metric"])
print("secondary\\\_continuous:", e\\\["continuous"]\\\["secondary\\\_metrics"])
print("discrete\\\_metrics:", e\\\["discrete"]\\\["metrics"])
print("compute\\\_brier:", e\\\["discrete"]\\\["compute\\\_brier"])
print("ml\\\_forecasts\\\_dir:", e\\\["inputs"]\\\["ml\\\_forecasts\\\_dir"])
print("metrics\\\_dir:", e\\\["outputs"]\\\["metrics\\\_dir"])
PY
```

**Salida relevante**

* `version: v1`
* `primary\\\_metric: qlike`
* `secondary\\\_continuous: \\\['rmse', 'mae']`
* `discrete\\\_metrics: \\\['macro\\\_f1', 'balanced\\\_accuracy']`
* `compute\\\_brier: False`

**Resultado útil**

* Se confirmó que el bloque de configuración de evaluación quedó bien cargado.

\---

### 3\) Inspeccionar contratos reales de entrada

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

SYMBOL = "spy"

roots = {
    "targets": Path("data/targets"),
    "splits": Path("artifacts/evaluations/splits"),
    "regime\\\_targets": Path("artifacts/evaluations/regime\\\_targets"),
    "benchmark\\\_forecasts": Path("artifacts/evaluations/benchmark\\\_forecasts"),
    "benchmark\\\_regimes": Path("artifacts/evaluations/benchmark\\\_regimes"),
    "ml\\\_forecasts": Path("artifacts/evaluations/ml\\\_forecasts"),
}

print("\\\\n===== FILE DISCOVERY =====")
resolved = {}
for name, root in roots.items():
    files = sorted(root.glob(f"{SYMBOL}/\\\*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontró parquet para {name} en {root / SYMBOL}")
    resolved\\\[name] = files\\\[0]
    print(f"{name}: {files\\\[0]}")

interesting\\\_cols = \\\[
    "instrument\\\_id",
    "date",
    "split\\\_id",
    "dataset\\\_role",
    "future\\\_rv\\\_5d",
    "future\\\_regime\\\_5d",
    "yhat\\\_future\\\_rv\\\_5d",
    "yhat\\\_future\\\_regime\\\_5d",
    "threshold\\\_low",
    "threshold\\\_high",
    "regime\\\_thresholds\\\_source",
    "run\\\_id",
    "model\\\_name",
]

print("\\\\n===== CONTRACT INSPECTION =====")
frames = {}
for name, path in resolved.items():
    df = pd.read\\\_parquet(path)
    frames\\\[name] = df

    present = \\\[c for c in interesting\\\_cols if c in df.columns]

    print(f"\\\\n--- {name} ---")
    print("path:", path)
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("selected\\\_present:", present)
    print("dtypes\\\_subset:", {c: str(df\\\[c].dtype) for c in present})

    preview\\\_cols = present\\\[:8] if present else list(df.columns\\\[:8])
    print("preview:")
    print(df\\\[preview\\\_cols].head(3).to\\\_string(index=False))

print("\\\\n===== KEY ALIGNMENT CHECK :: benchmark\\\_forecasts vs ml\\\_forecasts =====")
bench = frames\\\["benchmark\\\_forecasts"].copy()
ml = frames\\\["ml\\\_forecasts"].copy()

candidate\\\_keys = \\\["instrument\\\_id", "date", "split\\\_id", "dataset\\\_role"]
common\\\_keys = \\\[c for c in candidate\\\_keys if c in bench.columns and c in ml.columns]

print("common\\\_keys:", common\\\_keys)

if not common\\\_keys:
    raise RuntimeError("No hay claves comunes suficientes entre benchmark\\\_forecasts y ml\\\_forecasts.")

print("benchmark duplicated keys:", int(bench.duplicated(subset=common\\\_keys).sum()))
print("ml duplicated keys:", int(ml.duplicated(subset=common\\\_keys).sum()))

bench\\\_view = bench\\\[common\\\_keys + \\\[c for c in \\\["yhat\\\_future\\\_rv\\\_5d", "future\\\_rv\\\_5d"] if c in bench.columns]].copy()
ml\\\_view = ml\\\[common\\\_keys + \\\[c for c in \\\["yhat\\\_future\\\_rv\\\_5d", "future\\\_rv\\\_5d"] if c in ml.columns]].copy()

merged = bench\\\_view.merge(
    ml\\\_view,
    on=common\\\_keys,
    how="outer",
    indicator=True,
    suffixes=("\\\_bench", "\\\_ml"),
)

print("merge\\\_status\\\_counts:", merged\\\["\\\_merge"].value\\\_counts(dropna=False).to\\\_dict())

print("\\\\n===== PROBABILITY CHECK =====")
for name in \\\["benchmark\\\_regimes", "ml\\\_forecasts", "benchmark\\\_forecasts", "regime\\\_targets"]:
    df = frames\\\[name]
    prob\\\_like = \\\[c for c in df.columns if "prob" in c.lower() or "proba" in c.lower()]
    print(f"{name}: probability\\\_like\\\_columns={prob\\\_like}")

print("\\\\n===== REGIME LABEL CHECK =====")
for name in \\\["benchmark\\\_regimes", "regime\\\_targets"]:
    df = frames\\\[name]
    label\\\_cols = \\\[c for c in \\\["future\\\_regime\\\_5d", "yhat\\\_future\\\_regime\\\_5d"] if c in df.columns]
    print(f"{name}: label\\\_cols={label\\\_cols}")
    for col in label\\\_cols:
        values = sorted(pd.Series(df\\\[col].dropna().unique()).astype(str).tolist())
        print(f"  {col}: {values}")

print("\\\\n===== DONE =====")
PY
```

**Resultado útil**

* Se confirmó que el input benchmark correcto era `benchmark\\\_regimes`.
* Se confirmó que el lado ML debía completarse con `regime\\\_targets`.
* Se confirmó que la clave OOS efectiva era `date + split\\\_id + dataset\\\_role`.
* Se confirmó ausencia de probabilidades, justificando `compute\\\_brier: false`.

\---

### 4\) Verificar import del builder reusable de evaluación

```bash
python - <<'PY'
from quant\\\_platform.evaluation import (
    ModelComparisonArtifacts,
    build\\\_model\\\_comparison\\\_artifacts,
)

print(ModelComparisonArtifacts)
print(build\\\_model\\\_comparison\\\_artifacts)
PY
```

**Resultado útil**

* Se confirmó que el builder y la dataclass quedaron correctamente exportados e importables.

\---

### 5\) Validar el builder sobre SPY real

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

from quant\\\_platform.evaluation import build\\\_model\\\_comparison\\\_artifacts

symbol = "spy"

benchmark\\\_path = sorted(
    Path("artifacts/evaluations/benchmark\\\_regimes").glob(f"{symbol}/\\\*.parquet")
)\\\[0]
ml\\\_path = sorted(
    Path("artifacts/evaluations/ml\\\_forecasts").glob(f"{symbol}/\\\*.parquet")
)\\\[0]
regime\\\_targets\\\_path = sorted(
    Path("artifacts/evaluations/regime\\\_targets").glob(f"{symbol}/\\\*.parquet")
)\\\[0]

benchmark\\\_df = pd.read\\\_parquet(benchmark\\\_path)
ml\\\_df = pd.read\\\_parquet(ml\\\_path)
regime\\\_targets\\\_df = pd.read\\\_parquet(regime\\\_targets\\\_path)

artifacts = build\\\_model\\\_comparison\\\_artifacts(
    benchmark\\\_regimes\\\_df=benchmark\\\_df,
    ml\\\_forecasts\\\_df=ml\\\_df,
    regime\\\_targets\\\_df=regime\\\_targets\\\_df,
    evaluation\\\_version="v1",
    labels=("calm", "normal", "stress"),
)

panel\\\_df = artifacts.evaluation\\\_panel\\\_df
metrics\\\_df = artifacts.metrics\\\_df
confusion\\\_df = artifacts.confusion\\\_df

print("===== SHAPES =====")
print("evaluation\\\_panel\\\_rows:", len(panel\\\_df))
print("metrics\\\_rows:", len(metrics\\\_df))
print("confusion\\\_rows:", len(confusion\\\_df))

print("\\\\n===== PANEL SUMMARY =====")
print("models:", sorted(panel\\\_df\\\["model\\\_name"].unique().tolist()))
print("roles:", sorted(panel\\\_df\\\["dataset\\\_role"].unique().tolist()))
print("splits:", sorted(panel\\\_df\\\["split\\\_id"].unique().tolist()))
print("instrument\\\_ids:", sorted(panel\\\_df\\\["instrument\\\_id"].unique().tolist()))
print("symbols:", sorted(panel\\\_df\\\["symbol"].unique().tolist()))

print("\\\\n===== METRIC SUMMARY =====")
print("metric\\\_names:", sorted(metrics\\\_df\\\["metric\\\_name"].unique().tolist()))
print("target\\\_families:", sorted(metrics\\\_df\\\["target\\\_family"].unique().tolist()))
print("metrics\\\_n\\\_obs\\\_unique:", sorted(metrics\\\_df\\\["n\\\_obs"].dropna().unique().tolist())\\\[:10])

print("\\\\n===== ML REGIME MAPPING CHECK =====")
ml\\\_preview = panel\\\_df.loc\\\[
    panel\\\_df\\\["model\\\_name"] == "xgboost\\\_regressor",
    \\\[
        "date",
        "split\\\_id",
        "dataset\\\_role",
        "future\\\_rv\\\_5d",
        "yhat\\\_future\\\_rv\\\_5d",
        "future\\\_regime\\\_5d",
        "yhat\\\_future\\\_regime\\\_5d",
        "threshold\\\_low",
        "threshold\\\_high",
    ],
].head(10)
print(ml\\\_preview.to\\\_string(index=False))

print("\\\\n===== METRICS PREVIEW =====")
print(metrics\\\_df.head(12).to\\\_string(index=False))

print("\\\\n===== CONFUSION PREVIEW =====")
print(confusion\\\_df.head(12).to\\\_string(index=False))

print("\\\\n===== EXPECTED GROUP COUNTS =====")
group\\\_counts = (
    panel\\\_df.groupby(\\\["model\\\_name", "split\\\_id", "dataset\\\_role"])
    .size()
    .reset\\\_index(name="rows")
)
print(group\\\_counts.head(20).to\\\_string(index=False))

print("\\\\n===== DONE =====")
PY
```

**Resultado útil**

* SPY produjo correctamente:

  * `evaluation\\\_panel\\\_rows: 3510`
  * `metrics\\\_rows: 140`
  * `confusion\\\_rows: 252`
* Se verificó que el mapeo de régimen para ML estaba funcionando.

\---

### 6\) Verificar import del script operativo

```bash
python - <<'PY'
from scripts.evaluate\\\_models import main
print(main)
PY
```

**Resultado útil**

* Se confirmó que el script `scripts/evaluate\\\_models.py` quedó importable.

\---

### 7\) Ejecutar la materialización completa del Paso 24

```bash
python scripts/evaluate\\\_models.py

echo
echo "===== METRICS ROOT ====="
find artifacts/evaluations/model\\\_comparison -maxdepth 2 -type f | sort

echo
echo "===== CONFUSION ROOT ====="
find artifacts/evaluations/model\\\_comparison/confusion\\\_matrices -maxdepth 2 -type f | sort
```

**Resultado útil**

* Se materializó la evaluación para:

  * `gld`
  * `hyg`
  * `spy`
  * `tlt`
* Se confirmaron los 12 archivos esperados:

  * 4 paneles
  * 4 métricas
  * 4 confusiones
* El script terminó con `MODEL EVALUATION BUILD: PASS`.

\---

### 8\) Verificar import del checker

```bash
python - <<'PY'
from scripts.check\\\_model\\\_comparison\\\_quality import main
print(main)
PY
```

**Resultado útil**

* Se confirmó que el checker quedó importable.

\---

### 9\) Ejecutar checker de calidad de la nueva capa

```bash
python scripts/check\\\_model\\\_comparison\\\_quality.py
```

**Resultado útil**

* La capa quedó validada con:

  * `MODEL COMPARISON QUALITY CHECKS: PASS`
* Se confirmaron en PASS los 4 `panel`, 4 `metrics` y 4 `confusion`.

\---

### 10\) Ejecutar tests unitarios del builder

```bash
pytest tests/unit/test\\\_model\\\_comparison.py -q
```

**Resultado útil**

* Cierre unitario del bloque con:

  * `4 passed in 2.82s`

\---

## Resumen técnico del paso logrado

Durante esta sesión quedó completado el Paso 24 del SDD en su versión operativa `v1`:

* comparación benchmark vs ML por fold, rol y modelo;
* métricas continuas y discretas;
* matrices de confusión persistidas;
* checker dedicado;
* tests unitarios dedicados.

## Siguiente paso natural

1. Generar commit del bloque.
2. Redactar la entrada formal de bitácora ejecutiva del Paso 24.
3. Pasar al criterio de decisión del paso siguiente para decidir si ML aporta valor suficiente frente al benchmark.









