# Context Features v1 — Explicación detallada

## 1. Propósito

Este documento explica en detalle las **context features** definidas para el **paso 17** del proyecto **Quant Market Intelligence**.

La idea central de estas features no es describir solamente al activo objetivo “mirándose a sí mismo”, sino incorporar a cada fila de features una representación compacta del **estado del mercado alrededor del activo**.

En esta versión `v1`, las context features se diseñan usando **roles de mercado**, no símbolos hardcodeados. Eso permite que el diseño sea reutilizable cuando el universo se expanda.

---

## 2. Roles de mercado usados en `v1`

En esta primera versión, se definen cuatro roles:

- `equity_proxy = SPY`
- `duration_proxy = TLT`
- `credit_proxy = HYG`
- `real_asset_proxy = GLD`

### Significado económico de cada rol

## `equity_proxy`
Representa el mercado accionario amplio y el apetito por riesgo agregado.

## `duration_proxy`
Representa exposición a bonos de larga duración y sensibilidad a tasas / refugio relativo.

## `credit_proxy`
Representa el estado del riesgo de crédito y del componente `risk-on` fuera del equity puro.

## `real_asset_proxy`
Representa un activo real o cuasi refugio, útil como aproximación parcial a inflación / stress / cobertura.

---

## 3. Filosofía de diseño

Las context features definidas aquí siguen tres principios:

1. **ser simples e interpretables**;
2. **reusar la capa `features_v1` ya construida** en el paso 16;
3. **ser generalizables a universos mayores** mediante roles y no mediante nombres de activos incrustados en la lógica.

---

## 4. Lista completa de context features de `v1`

```text
ctx_equity_proxy_log_ret_1d
ctx_duration_proxy_log_ret_1d
ctx_credit_proxy_log_ret_1d
ctx_real_asset_proxy_log_ret_1d

ctx_equity_proxy_log_ret_5d
ctx_duration_proxy_log_ret_5d
ctx_credit_proxy_log_ret_5d
ctx_real_asset_proxy_log_ret_5d

ctx_equity_proxy_vol_20d
ctx_duration_proxy_vol_20d
ctx_credit_proxy_vol_20d
ctx_real_asset_proxy_vol_20d

ctx_equity_duration_ret_5d_spread
ctx_credit_duration_ret_5d_spread

ctx_rel_vol_20d_vs_equity_proxy
ctx_corr_20d_vs_equity_proxy
```

---

# 5. Explicación detallada feature por feature

# 5.1 Retornos contextuales de 1 día

## `ctx_equity_proxy_log_ret_1d`

### Qué es
Es el retorno logarítmico a 1 día del activo que cumple el rol de `equity_proxy`.

### Fórmula

\[
ctx\_equity\_proxy\_log\_ret\_1d_t = \ln\left(\frac{P^{equity}_t}{P^{equity}_{t-1}}\right)
\]

### Qué captura
Resume cómo se movió el mercado accionario amplio en la fecha `t`.

### Por qué sirve
Aporta una señal simple de estado de mercado: sesión positiva, negativa o neutral del benchmark accionario agregado.

### Interpretación práctica
Si el activo objetivo es, por ejemplo, `TLT`, esta feature le añade a su fila información sobre qué hizo `SPY` ese mismo día.

---

## `ctx_duration_proxy_log_ret_1d`

### Qué es
Retorno logarítmico a 1 día del activo que cumple el rol de `duration_proxy`.

### Fórmula

\[
ctx\_duration\_proxy\_log\_ret\_1d_t = \ln\left(\frac{P^{duration}_t}{P^{duration}_{t-1}}\right)
\]

### Qué captura
Estado diario del componente de duración / bonos largos / refugio relativo.

### Por qué sirve
Permite saber si, en esa fecha, hubo fuerza en bonos de duración o debilidad en ese bloque.

---

## `ctx_credit_proxy_log_ret_1d`

### Qué es
Retorno logarítmico a 1 día del activo que cumple el rol de `credit_proxy`.

### Fórmula

\[
ctx\_credit\_proxy\_log\_ret\_1d_t = \ln\left(\frac{P^{credit}_t}{P^{credit}_{t-1}}\right)
\]

### Qué captura
Estado diario del crédito riesgoso o del sentimiento hacia crédito corporativo más sensible al ciclo.

### Por qué sirve
El crédito suele reaccionar con fuerza al endurecimiento o relajación de las condiciones financieras.

---

## `ctx_real_asset_proxy_log_ret_1d`

### Qué es
Retorno logarítmico a 1 día del activo que cumple el rol de `real_asset_proxy`.

### Fórmula

\[
ctx\_real\_asset\_proxy\_log\_ret\_1d_t = \ln\left(\frac{P^{realasset}_t}{P^{realasset}_{t-1}}\right)
\]

### Qué captura
Estado diario de un activo real / refugio / inflación-adjacent, según cómo se defina el proxy.

### Por qué sirve
Aporta una dimensión distinta a equity, duration y crédito, útil en cambios de régimen.

---

# 5.2 Retornos contextuales de 5 días

## `ctx_equity_proxy_log_ret_5d`

### Qué es
Retorno logarítmico acumulado a 5 días del `equity_proxy`.

### Fórmula

\[
ctx\_equity\_proxy\_log\_ret\_5d_t = \ln\left(\frac{P^{equity}_t}{P^{equity}_{t-5}}\right)
\]

### Qué captura
Movimiento reciente del mercado accionario a un horizonte algo más estable que el diario.

### Por qué sirve
Reduce algo del ruido diario y refleja mejor el sesgo reciente del mercado.

---

## `ctx_duration_proxy_log_ret_5d`

### Qué es
Retorno logarítmico a 5 días del `duration_proxy`.

### Qué captura
Movimiento reciente del bloque de duración / bonos largos.

### Por qué sirve
Sirve como referencia de comportamiento defensivo o de tasas durante la última semana bursátil aproximada.

---

## `ctx_credit_proxy_log_ret_5d`

### Qué es
Retorno logarítmico a 5 días del `credit_proxy`.

### Qué captura
Sesgo reciente del crédito, más estable que el movimiento de un solo día.

### Por qué sirve
Ayuda a detectar si el mercado viene premiando o penalizando riesgo crediticio en la ventana reciente.

---

## `ctx_real_asset_proxy_log_ret_5d`

### Qué es
Retorno logarítmico a 5 días del `real_asset_proxy`.

### Qué captura
Dirección reciente del activo real / refugio en un horizonte corto ampliado.

### Por qué sirve
Permite comparar el comportamiento reciente de activos reales con equity, bonds y crédito.

---

# 5.3 Volatilidad contextual de 20 días

## `ctx_equity_proxy_vol_20d`

### Qué es
Volatilidad rolling anualizada de 20 días del `equity_proxy`.

### Fórmula conceptual

Se calcula igual que `vol_20d` del paso 16, pero aplicada al `equity_proxy`:

\[
ctx\_equity\_proxy\_vol\_20d_t = \operatorname{std}(r^{equity}_{t-19}, ..., r^{equity}_t) \cdot \sqrt{252}
\]

### Qué captura
Cuán turbulento viene el mercado accionario agregado.

### Por qué sirve
Es una de las señales más útiles de régimen: un equity proxy con volatilidad creciente suele indicar deterioro del entorno de riesgo.

---

## `ctx_duration_proxy_vol_20d`

### Qué es
Volatilidad rolling anualizada de 20 días del `duration_proxy`.

### Qué captura
Nivel de turbulencia en el bloque de duración / bonos largos.

### Por qué sirve
En ciertas fases del ciclo, la volatilidad en bonos largos también cambia de régimen y aporta contexto valioso.

---

## `ctx_credit_proxy_vol_20d`

### Qué es
Volatilidad rolling anualizada de 20 días del `credit_proxy`.

### Qué captura
Turbulencia del bloque de crédito.

### Por qué sirve
El crédito suele amplificar episodios de estrés financiero. Su volatilidad contextual añade información de fragilidad sistémica.

---

## `ctx_real_asset_proxy_vol_20d`

### Qué es
Volatilidad rolling anualizada de 20 días del `real_asset_proxy`.

### Qué captura
Nivel de incertidumbre o agitación del activo real / refugio.

### Por qué sirve
Aporta una dimensión contextual distinta del equity y del crédito.

---

# 5.4 Spreads contextuales

## `ctx_equity_duration_ret_5d_spread`

### Qué es
Spread entre el retorno a 5 días del `equity_proxy` y el retorno a 5 días del `duration_proxy`.

### Fórmula

\[
ctx\_equity\_duration\_ret\_5d\_spread_t =
ctx\_equity\_proxy\_log\_ret\_5d_t - ctx\_duration\_proxy\_log\_ret\_5d_t
\]

### Qué captura
La tensión relativa entre riesgo accionario y refugio/duración.

### Interpretación
- valor alto: equity superando claramente a duration;
- valor bajo o negativo: duration resistiendo o superando a equity.

### Por qué sirve
Resume un eje clásico `risk-on / risk-off` de manera muy compacta.

---

## `ctx_credit_duration_ret_5d_spread`

### Qué es
Spread entre el retorno a 5 días del `credit_proxy` y el del `duration_proxy`.

### Fórmula

\[
ctx\_credit\_duration\_ret\_5d\_spread_t =
ctx\_credit\_proxy\_log\_ret\_5d_t - ctx\_duration\_proxy\_log\_ret\_5d_t
\]

### Qué captura
La tensión relativa entre crédito riesgoso y refugio/duración.

### Interpretación
- valor alto: crédito superando a bonos;
- valor bajo o negativo: bonos resistiendo mejor que crédito.

### Por qué sirve
Es una señal muy útil de condiciones financieras y apetito por riesgo crediticio.

---

# 5.5 Volatilidad relativa frente al proxy accionario

## `ctx_rel_vol_20d_vs_equity_proxy`

### Qué es
La volatilidad rolling de 20 días del activo objetivo dividida por la volatilidad rolling de 20 días del `equity_proxy`.

### Fórmula

\[
ctx\_rel\_vol\_20d\_vs\_equity\_proxy_t =
\frac{vol\_20d^{target}_t}{ctx\_equity\_proxy\_vol\_20d_t}
\]

### Qué captura
Qué tan volátil es el activo objetivo en relación con el benchmark accionario agregado.

### Interpretación
- mayor que 1: el activo objetivo está más turbulento que el equity proxy;
- cercano a 1: volatilidad similar;
- menor que 1: menos volátil que el equity proxy.

### Por qué sirve
Porque no solo importa la volatilidad absoluta del activo, sino su posición relativa respecto al mercado agregado.

---

# 5.6 Correlación rolling frente al proxy accionario

## `ctx_corr_20d_vs_equity_proxy`

### Qué es
Correlación rolling de 20 días entre los retornos diarios del activo objetivo y los retornos diarios del `equity_proxy`.

### Fórmula conceptual

\[
ctx\_corr\_20d\_vs\_equity\_proxy_t =
\operatorname{corr}\left(r^{target}_{t-19:t}, r^{equity}_{t-19:t}\right)
\]

### Qué captura
El grado de comovimiento reciente del activo objetivo con el mercado accionario amplio.

### Interpretación
- alta y positiva: el activo se está moviendo junto al mercado;
- baja o negativa: el activo se desacopla o actúa como diversificador relativo.

### Por qué sirve
La correlación es una variable de régimen. En entornos de estrés, las correlaciones suelen cambiar y eso altera el valor informativo de muchas señales.

### Política para el caso especial del proxy consigo mismo
Si el activo objetivo coincide con `equity_proxy`, esta feature debe definirse como:

```text
NaN
```

### Por qué
Porque una autocorrelación contemporánea consigo mismo sería trivial y no añadiría información útil.

---

# 6. Lectura económica del bloque completo

Este conjunto de context features intenta resumir cuatro ejes de estado de mercado:

1. **dirección reciente del mercado**
   - vía retornos contextuales;
2. **turbulencia reciente del mercado**
   - vía volatilidades contextuales;
3. **balance risk-on / risk-off**
   - vía spreads equity-duration y credit-duration;
4. **posición relativa del activo objetivo dentro del sistema**
   - vía volatilidad relativa y correlación rolling.

No intenta capturar todavía toda la macroeconomía. Busca algo más acotado:

> que cada fila del activo objetivo sepa, de forma compacta, cómo estaba el entorno financiero en esa fecha.

---

# 7. Qué no entra todavía en `v1`

Estas families de contexto quedan fuera en esta primera iteración:

- term spread “real” de tasas largas vs cortas;
- breakevens de inflación;
- dólar / FX;
- commodities amplios / energía;
- liquidity spreads explícitos;
- breadth;
- indicadores macroeconómicos publicados con calendario.

La razón no es que no sean importantes, sino que **no pertenecen al primer `context v1` más limpio y adaptable** con el universo actual.

---

# 8. Resumen final

Las context features de `v1` quedan congeladas como:

```text
ctx_equity_proxy_log_ret_1d
ctx_duration_proxy_log_ret_1d
ctx_credit_proxy_log_ret_1d
ctx_real_asset_proxy_log_ret_1d

ctx_equity_proxy_log_ret_5d
ctx_duration_proxy_log_ret_5d
ctx_credit_proxy_log_ret_5d
ctx_real_asset_proxy_log_ret_5d

ctx_equity_proxy_vol_20d
ctx_duration_proxy_vol_20d
ctx_credit_proxy_vol_20d
ctx_real_asset_proxy_vol_20d

ctx_equity_duration_ret_5d_spread
ctx_credit_duration_ret_5d_spread

ctx_rel_vol_20d_vs_equity_proxy
ctx_corr_20d_vs_equity_proxy
```

Y la lógica subyacente es:

- **4 proxies de mercado**
- **2 horizontes de retorno**
- **1 horizonte de volatilidad**
- **2 spreads estructurales**
- **1 feature de posición relativa en volatilidad**
- **1 feature de comovimiento con el benchmark accionario**

Ese bloque es lo bastante simple para implementarse ya, y lo bastante general para sobrevivir cuando el universo se expanda.
