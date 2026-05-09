from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.services.settings import load_settings


def _get_services_layer_settings() -> dict:
    """
    Recupera la sección `services_layer` desde la configuración global
    del proyecto y valida que tenga el tipo esperado.

    Propósito:
    Esta función centraliza el acceso a la configuración específica de la
    capa de servicios. En lugar de repetir en muchas funciones el patrón
    de cargar settings y extraer `services_layer`, se encapsula aquí.

    Retorna:
    - Un diccionario con la configuración de `services_layer`.

    Lanza:
    - ValueError si la clave `services_layer` no existe o no contiene
      un diccionario válido.

    Importancia:
    Esta función actúa como punto de entrada confiable para todas las demás
    funciones del archivo. Si la configuración base está rota, el fallo
    aparece aquí de forma explícita y temprana.
    """
    settings = load_settings()

    # Intenta extraer la sección `services_layer` de la configuración global.
    services_layer = settings.get("services_layer")

    # Valida que dicha sección exista y sea un diccionario. Si no lo es,
    # el resto del archivo no puede operar correctamente.
    if not isinstance(services_layer, dict):
        raise ValueError(
            "services_layer settings are missing or invalid in configs/base.yaml"
        )

    return services_layer


def _get_input_root(key: str) -> Path:
    """
    Obtiene una ruta raíz de entrada desde `services_layer.inputs`.

    Parámetros:
    - key:
      nombre lógico de la ruta a buscar dentro de:
          services_layer["inputs"]

      Ejemplos posibles:
      - "features_dir"
      - "targets_dir"
      - "model_comparison_dir"
      - "structural_breaks_dir"

    Retorna:
    - Un objeto `Path` con la ruta configurada.

    Lanza:
    - ValueError si la clave solicitada no existe dentro de los inputs.

    Propósito:
    Evitar hardcodear rutas en funciones específicas. Cada loader pide
    su root por nombre lógico y esta función resuelve el valor real
    desde la configuración.
    """
    services_layer = _get_services_layer_settings()

    # Recupera el bloque `inputs`; si no existe, usa dict vacío para que
    # la validación posterior arroje un error claro.
    inputs = services_layer.get("inputs", {})

    # Verifica que la clave solicitada exista.
    if key not in inputs:
        raise ValueError(f"Missing services_layer.inputs[{key!r}] in configuration")

    # Convierte el valor configurado a Path para trabajar con rutas
    # de forma robusta y expresiva.
    return Path(inputs[key])


def _get_default_value(key: str) -> str:
    """
    Obtiene un valor por defecto desde `services_layer.defaults`.

    Parámetros:
    - key:
      nombre del valor por defecto a recuperar.

      Ejemplos posibles:
      - "preferred_decision_summary_file"
      - "preferred_decision_reasons_file"

    Retorna:
    - El valor asociado a la clave solicitada.

    Lanza:
    - ValueError si la clave no existe dentro de `defaults`.

    Propósito:
    Algunas funciones no descubren archivos dinámicamente, sino que usan
    un archivo preferido definido en configuración. Esta función encapsula
    ese acceso.
    """
    services_layer = _get_services_layer_settings()

    # Recupera el bloque de valores por defecto desde la configuración.
    defaults = services_layer.get("defaults", {})

    # Verifica que el valor solicitado exista.
    if key not in defaults:
        raise ValueError(f"Missing services_layer.defaults[{key!r}] in configuration")

    return defaults[key]


def _get_single_symbol_parquet(
    root_dir: Path, symbol: str, glob_pattern: str = "*.parquet"
) -> Path:
    """
    Encuentra exactamente un archivo parquet para un símbolo dentro de una
    raíz dada, opcionalmente filtrando con un patrón glob.

    Parámetros:
    - root_dir:
      carpeta raíz bajo la cual se espera una subcarpeta por símbolo.
    - symbol:
      símbolo cuyo parquet se desea cargar.
    - glob_pattern:
      patrón de búsqueda dentro de la carpeta del símbolo.
      Por defecto busca cualquier parquet:
          "*.parquet"

      Pero puede refinarse, por ejemplo:
          "*_model_comparison_metrics_*.parquet"
          "*_structural_break_signal_*.parquet"

    Retorna:
    - La ruta `Path` al único parquet encontrado.

    Lanza:
    - FileNotFoundError si no existe la carpeta del símbolo.
    - FileNotFoundError si no hay archivos que coincidan.
    - RuntimeError si hay más de un archivo coincidente.

    Convención importante:
    El script asume que cada tipo de artefacto por símbolo tiene un único
    archivo vigente. Si hay cero o varios, eso se considera un estado
    inválido o ambiguo del pipeline.
    """
    # Normaliza el símbolo a minúsculas para alinearlo con la convención
    # de nombres de carpetas usada en el proyecto.
    symbol_dir = root_dir / symbol.lower()

    # Busca todos los parquets que coincidan con el patrón dentro de
    # la carpeta del símbolo.
    matches = sorted(symbol_dir.glob(glob_pattern))

    # Primero verifica que exista la carpeta del símbolo.
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Missing symbol directory: {symbol_dir}")

    # Si no hay matches, el artefacto esperado no está presente.
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No parquet files found in {symbol_dir} with pattern={glob_pattern!r}"
        )

    # Si hay más de uno, el estado es ambiguo: no está claro cuál tomar.
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one parquet in {symbol_dir} with pattern={glob_pattern!r}, found {len(matches)}: {matches}"
        )

    return matches[0]


def list_available_symbols() -> list[str]:
    """
    Lista los símbolos disponibles a partir del directorio de features.

    Retorna:
    - Una lista de símbolos en mayúsculas, ordenados alfabéticamente.

    Lanza:
    - FileNotFoundError si no existe la carpeta raíz de features.
    - RuntimeError si la carpeta existe pero no contiene subdirectorios
      de símbolos.

    Propósito:
    Esta función sirve como mecanismo de descubrimiento del universo de
    símbolos disponibles para la capa de servicios.

    Decisión de diseño:
    Los nombres se devuelven en mayúsculas para una representación más
    amigable hacia capas superiores, aunque las carpetas se lean como
    estén definidas en disco.
    """
    # Obtiene la raíz configurada donde viven los features por símbolo.
    features_root = _get_input_root("features_dir")

    # Verifica que la raíz exista.
    if not features_root.exists():
        raise FileNotFoundError(f"Features root does not exist: {features_root}")

    # Recorre las carpetas hijas y toma cada nombre como un símbolo.
    # Se convierten a mayúsculas para estandarizar la salida pública.
    symbols = sorted(
        path.name.upper() for path in features_root.iterdir() if path.is_dir()
    )

    # Si no hay carpetas de símbolos, se reporta el estado como inválido.
    if not symbols:
        raise RuntimeError(f"No symbol directories found under: {features_root}")

    return symbols


def load_symbol_features(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de features correspondiente a un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con las features del símbolo.

    Flujo:
    1. Obtiene la raíz `features_dir` desde configuración.
    2. Encuentra el único parquet dentro de la carpeta del símbolo.
    3. Lo carga con `pd.read_parquet`.

    Propósito:
    Ofrecer una función de alto nivel, simple y directa, para acceder
    a las features de un símbolo sin exponer la lógica de rutas al resto
    del proyecto.
    """
    root = _get_input_root("features_dir")
    path = _get_single_symbol_parquet(root, symbol)
    return pd.read_parquet(path)


def load_symbol_targets(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de targets correspondiente a un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con los targets del símbolo.

    Propósito:
    Resolver de forma uniforme el acceso a la capa de targets, siguiendo
    exactamente la misma convención usada para features.
    """
    root = _get_input_root("targets_dir")
    path = _get_single_symbol_parquet(root, symbol)
    return pd.read_parquet(path)


def load_symbol_normalized_bars(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de barras OHLCV normalizadas para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con barras diarias normalizadas del símbolo.

    Propósito:
    Exponer de forma explícita la capa de mercado base que la UI necesita para
    mostrar contexto visual del activo sin acceder directamente a archivos.
    """
    root = _get_input_root("normalized_bars_dir")
    path = _get_single_symbol_parquet(root, symbol, "*_daily_bars.parquet")
    return pd.read_parquet(path)


def load_symbol_benchmark_forecasts(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de forecasts continuos del benchmark para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con los forecasts del benchmark.
    """
    root = _get_input_root("benchmark_forecasts_dir")
    path = _get_single_symbol_parquet(root, symbol)
    return pd.read_parquet(path)


def load_symbol_benchmark_regimes(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de regímenes del benchmark para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con la información de regímenes asociada al benchmark.

    Propósito:
    Exponer una función específica y semánticamente clara para acceder a
    los artefactos de benchmark regimes desde la capa de servicios.
    """
    root = _get_input_root("benchmark_regimes_dir")
    path = _get_single_symbol_parquet(root, symbol)
    return pd.read_parquet(path)


def load_symbol_ml_forecasts(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de pronósticos ML correspondiente a un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con los forecasts del modelo ML.

    Propósito:
    Separar conceptualmente el acceso a forecasts ML de otros artefactos,
    aunque internamente siga el mismo patrón de carga.
    """
    root = _get_input_root("ml_forecasts_dir")
    path = _get_single_symbol_parquet(root, symbol)
    return pd.read_parquet(path)


def load_symbol_model_comparison_metrics(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de métricas de comparación de modelos para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con métricas de comparación entre modelos.

    Particularidad:
    Aquí no basta con cualquier parquet: se usa un patrón específico para
    localizar únicamente el archivo de métricas dentro de la carpeta del símbolo.

    Propósito:
    Diferenciar explícitamente este artefacto del panel de comparación u otros
    posibles outputs que vivan en el mismo root.
    """
    root = _get_input_root("model_comparison_dir")
    path = _get_single_symbol_parquet(
        root, symbol, "*_model_comparison_metrics_*.parquet"
    )
    return pd.read_parquet(path)


def load_symbol_model_comparison_panel(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet del panel de comparación de modelos para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con el panel de comparación de modelos.

    Diferencia respecto a `load_symbol_model_comparison_metrics`:
    Esta función busca el artefacto tipo panel, no el de métricas agregadas.
    Ambos comparten root, pero se distinguen por el patrón de nombre.
    """
    root = _get_input_root("model_comparison_dir")
    path = _get_single_symbol_parquet(
        root, symbol, "*_model_comparison_panel_*.parquet"
    )
    return pd.read_parquet(path)


def load_symbol_model_comparison_confusion_matrix(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de confusion matrices de model comparison para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con conteos por celda de confusion matrix.
    """
    root = _get_input_root("confusion_matrices_dir")
    path = _get_single_symbol_parquet(
        root, symbol, "*_model_comparison_confusion_*.parquet"
    )
    return pd.read_parquet(path)


def load_symbol_structural_break_signal(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de señal de structural breaks para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con la señal usada o generada para el análisis de
      quiebres estructurales.

    Propósito:
    Proveer acceso explícito a uno de los dos artefactos principales del
    módulo de structural breaks: la señal.
    """
    root = _get_input_root("structural_breaks_dir")
    path = _get_single_symbol_parquet(
        root, symbol, "*_structural_break_signal_*.parquet"
    )
    return pd.read_parquet(path)


def load_symbol_structural_break_events(symbol: str) -> pd.DataFrame:
    """
    Carga el parquet de eventos de structural breaks para un símbolo.

    Parámetros:
    - symbol:
      símbolo a cargar.

    Retorna:
    - Un DataFrame con los eventos/quiebres detectados.

    Propósito:
    Complementar `load_symbol_structural_break_signal` con el segundo
    artefacto principal del módulo de structural breaks: la tabla de eventos.
    """
    root = _get_input_root("structural_breaks_dir")
    path = _get_single_symbol_parquet(
        root, symbol, "*_structural_break_events_*.parquet"
    )
    return pd.read_parquet(path)


def load_decision_summary() -> pd.DataFrame:
    """
    Carga el parquet de resumen de decisiones desde el directorio de decisiones.

    Retorna:
    - Un DataFrame con el summary de decisiones.

    Flujo:
    1. Obtiene el root `decision_dir`.
    2. Obtiene desde configuración el nombre del archivo preferido.
    3. Construye la ruta completa.
    4. Verifica que exista.
    5. Lo carga en un DataFrame.

    Propósito:
    A diferencia de los loaders por símbolo, aquí el archivo se selecciona
    por nombre configurado, no por descubrimiento dinámico.
    """
    root = _get_input_root("decision_dir")
    filename = _get_default_value("preferred_decision_summary_file")
    path = root / filename

    # Se verifica explícitamente existencia porque aquí no se usa el helper
    # de unicidad por símbolo.
    if not path.exists():
        raise FileNotFoundError(f"Decision summary parquet not found: {path}")

    return pd.read_parquet(path)


def load_decision_reasons() -> pd.DataFrame:
    """
    Carga el parquet de razones de decisión desde el directorio de decisiones.

    Retorna:
    - Un DataFrame con las razones o explicaciones de decisión.

    Flujo:
    1. Obtiene el root `decision_dir`.
    2. Obtiene desde configuración el nombre del archivo preferido.
    3. Construye la ruta completa.
    4. Verifica que exista.
    5. Lo carga en un DataFrame.

    Propósito:
    Exponer una función dedicada para acceder al artefacto de razones,
    separándolo semánticamente del summary aunque ambos provengan del
    mismo directorio.
    """
    root = _get_input_root("decision_dir")
    filename = _get_default_value("preferred_decision_reasons_file")
    path = root / filename

    # Se valida existencia antes de intentar leer el archivo.
    if not path.exists():
        raise FileNotFoundError(f"Decision reasons parquet not found: {path}")

    return pd.read_parquet(path)
