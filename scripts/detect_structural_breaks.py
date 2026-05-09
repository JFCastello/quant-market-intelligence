from __future__ import annotations
from pathlib import Path
import pandas as pd
from quant_platform.evaluation import build_structural_break_artifacts
from quant_platform.services.settings import load_settings



def ensure_directory(path: Path) -> None:
    """
    Garantiza que exista el directorio indicado por `path`.

    Propósito:
    - Evitar errores al intentar guardar archivos en carpetas que todavía
      no existen.
    - Crear toda la jerarquía necesaria de directorios si hace falta.

    Parámetros:
    - path: ruta del directorio que se quiere asegurar.

    Comportamiento:
    - `parents=True` permite crear carpetas intermedias.
    - `exist_ok=True` evita que falle si la carpeta ya existe.

    Ejemplo:
    Si `path` es:
        artifacts/evaluations/structural_breaks/spy
    y la carpeta no existe, esta función la crea.
    """
    path.mkdir(parents=True, exist_ok=True)


def discover_symbols(features_root: Path) -> list[str]:
    """
    Descubre automáticamente qué símbolos existen en el directorio de features.

    Idea general:
    El pipeline asume una estructura tipo:

        features_root/
            spy/
                spy_...parquet
            tlt/
                tlt_...parquet
            gld/
                gld_...parquet

    Esta función recorre `features_root` y toma el nombre de cada subdirectorio
    como un símbolo disponible.

    Parámetros:
    - features_root: carpeta raíz donde viven las features separadas por símbolo.

    Retorna:
    - Una lista ordenada alfabéticamente con los símbolos encontrados.
      Ejemplo:
        ["gld", "hyg", "spy", "tlt"]

    Validaciones:
    1. Si la carpeta raíz no existe, lanza FileNotFoundError.
       Esto evita seguir el pipeline con una ruta mal configurada.
    2. Si existe la carpeta pero no contiene subdirectorios de símbolos,
       lanza RuntimeError.
       Esto evita un "éxito vacío" donde el script corre pero no procesa nada.

    Observación:
    Aquí se usa `path.is_dir()` para asegurarse de tomar solo carpetas y no
    archivos sueltos.
    """
    if not features_root.exists():
        raise FileNotFoundError(f"features root does not exist: {features_root}")

    symbols = sorted(
        path.name
        for path in features_root.iterdir()
        if path.is_dir()
    )

    if not symbols:
        raise RuntimeError(f"No symbol directories found under: {features_root}")

    return symbols


def get_single_symbol_parquet(root_dir: Path, symbol: str) -> Path:
    """
    Busca y devuelve el único archivo parquet esperado para un símbolo
    dentro de un directorio raíz dado.

    Estructura esperada:
        root_dir/
            spy/
                <un solo archivo .parquet>
            tlt/
                <un solo archivo .parquet>

    Parámetros:
    - root_dir: raíz donde están los subdirectorios por símbolo
      (por ejemplo features_root o targets_root).
    - symbol: símbolo que se quiere resolver, por ejemplo "spy".

    Retorna:
    - La ruta (`Path`) del archivo parquet encontrado.

    Validaciones importantes:
    1. Si no existe la carpeta del símbolo, lanza FileNotFoundError.
    2. Si la carpeta existe pero no tiene archivos parquet, lanza FileNotFoundError.
    3. Si tiene más de un parquet, lanza RuntimeError.

    ¿Por qué exigir exactamente uno?
    Porque este script está diseñado bajo la convención de que para cada símbolo
    y para cada etapa (features / targets) debe existir un único parquet vigente.
    Si hubiera varios, el script no sabría cuál usar, y elegir arbitrariamente
    sería peligroso desde el punto de vista de reproducibilidad.

    Ejemplo:
        get_single_symbol_parquet(features_root, "spy")
    podría devolver algo como:
        data/features/spy/spy_2018-01-01_2026-04-05_features_v1.parquet
    """
    symbol_dir = root_dir / symbol
    matches = sorted(symbol_dir.glob("*.parquet"))

    if not symbol_dir.exists():
        raise FileNotFoundError(f"Missing symbol directory: {symbol_dir}")
    if len(matches) == 0:
        raise FileNotFoundError(f"No parquet files found in: {symbol_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one parquet in {symbol_dir}, found {len(matches)}: {matches}"
        )

    return matches[0]


def build_window_token(signal_df: pd.DataFrame) -> str:
    """
    Construye un identificador temporal (window token) a partir del rango
    de fechas presente en `signal_df`.

    Propósito:
    - Incluir en el nombre del archivo el período exacto cubierto por la señal.
    - Hacer los outputs más trazables y fáciles de auditar.

    Parámetros:
    - signal_df: DataFrame que debe contener una columna llamada "date".

    Retorna:
    - Un string con el formato:
        "YYYY-MM-DD_YYYY-MM-DD"
      donde la primera fecha es la mínima y la segunda es la máxima.

    Ejemplo:
    Si signal_df["date"] cubre desde 2018-01-02 hasta 2026-04-05,
    retornará:
        "2018-01-02_2026-04-05"

    Validación:
    - Si `signal_df` está vacío, lanza ValueError.
      No tiene sentido construir una ventana temporal sin filas.

    Nota técnica:
    - `pd.to_datetime(...)` asegura que la columna se trate como fecha.
    - `.date().isoformat()` deja cada fecha en formato limpio ISO.
    """
    if signal_df.empty:
        raise ValueError("signal_df is empty; cannot build window token.")

    min_date = pd.to_datetime(signal_df["date"]).min().date().isoformat()
    max_date = pd.to_datetime(signal_df["date"]).max().date().isoformat()
    return f"{min_date}_{max_date}"


def save_symbol_artifacts(
    symbol: str,
    break_version: str,
    signal_df: pd.DataFrame,
    events_df: pd.DataFrame,
    events_root: Path,
) -> tuple[Path, Path]:
    """
    Guarda en disco los artefactos producidos para un símbolo:
    1. la señal usada/producida para detectar quiebres,
    2. la tabla de eventos de quiebre detectados.

    Parámetros:
    - symbol: símbolo procesado, por ejemplo "spy".
    - break_version: versión lógica del pipeline/modelo de structural breaks.
      Se usa en los nombres de archivo para versionado explícito.
    - signal_df: DataFrame con la señal o señales procesadas.
    - events_df: DataFrame con los eventos de quiebre detectados.
    - events_root: carpeta raíz donde se almacenarán los outputs.

    Flujo interno:
    1. Construye la carpeta del símbolo:
         events_root / symbol
    2. Se asegura de que exista esa carpeta.
    3. Construye el token temporal a partir de `signal_df`.
    4. Define dos nombres de archivo:
         - structural_break_signal
         - structural_break_events
    5. Guarda ambos DataFrames en formato parquet.
    6. Devuelve las rutas generadas.

    Formato de salida:
    - signal:
        {symbol}_{window}_structural_break_signal_{break_version}.parquet
    - events:
        {symbol}_{window}_structural_break_events_{break_version}.parquet

    ¿Por qué esto es útil?
    - Mantiene los outputs organizados por símbolo.
    - Hace visible el período temporal cubierto.
    - Hace visible la versión del pipeline.
    - Facilita reconstrucción, auditoría y debugging.

    Retorna:
    - Una tupla:
        (signal_path, events_path)
      con las rutas exactas de los archivos guardados.
    """
    symbol_dir = events_root / symbol
    ensure_directory(symbol_dir)

    window_token = build_window_token(signal_df)

    signal_path = (
        symbol_dir
        / f"{symbol}_{window_token}_structural_break_signal_{break_version}.parquet"
    )
    events_path = (
        symbol_dir
        / f"{symbol}_{window_token}_structural_break_events_{break_version}.parquet"
    )

    signal_df.to_parquet(signal_path, index=False)
    # index=False evita guardar el índice del DataFrame como una columna adicional.
    # Esto suele ser lo correcto cuando el índice no contiene información de negocio.

    events_df.to_parquet(events_path, index=False)

    return signal_path, events_path


def main() -> None:
    """
    Función principal del script.

    Responsabilidad global:
    Orquestar de punta a punta el proceso de detección de quiebres estructurales
    para todos los símbolos disponibles.

    Flujo detallado:
    ------------------------------------------------------------------------
    1. Cargar configuración
       - Lee settings del proyecto.
       - Extrae la sección `structural_breaks`.

    2. Resolver rutas
       - features_root: dónde están las features por símbolo.
       - targets_root: dónde están los targets por símbolo.
       - events_root: dónde se guardarán los outputs del proceso.

    3. Descubrir símbolos
       - Busca automáticamente qué símbolos existen en `features_root`.

    4. Imprimir encabezado de ejecución
       - Muestra qué símbolos se procesarán.
       - Muestra versión, algoritmo y cost model.
       - Esto ayuda a trazabilidad en logs.

    5. Iterar símbolo por símbolo
       Para cada símbolo:
       a. localizar el parquet de features
       b. localizar el parquet de targets
       c. leer ambos DataFrames
       d. llamar `build_structural_break_artifacts(...)`
       e. guardar signal_df y events_df
       f. imprimir resumen del resultado

    6. Imprimir estado final PASS
       - Señal simple de finalización exitosa.

    Diseño:
    Esta función no implementa la lógica matemática de segmentación.
    Su papel es de orquestación del pipeline:
    - cargar entradas,
    - pasar configuración,
    - invocar el motor analítico,
    - persistir salidas,
    - reportar resultados.

    Esto es deseable porque separa:
    - lógica de negocio / pipeline (este script)
    de
    - lógica analítica / estadística (build_structural_break_artifacts)
    """
    settings = load_settings()
    # Carga toda la configuración del proyecto.

    cfg = settings["structural_breaks"]
    # Extrae solo la sección relevante para este pipeline.
    # Aquí deben vivir rutas, versión, parámetros del algoritmo, etc.

    features_root = Path(cfg["inputs"]["features_dir"])
    targets_root = Path(cfg["inputs"]["targets_dir"])
    events_root = Path(cfg["outputs"]["events_dir"])
    # Convierte a Path las rutas configuradas.
    # Esto facilita concatenación segura de subrutas.

    symbols = discover_symbols(features_root)
    # Descubre automáticamente qué símbolos deben procesarse.

    print("STRUCTURAL BREAK DETECTION START")
    print(f"symbols={symbols}")
    print(f"break_version={cfg['version']}")
    print(f"algorithm={cfg['method']['algorithm']}")
    print(f"cost_model={cfg['method']['cost_model']}")
    print("-" * 80)
    # Bloque de logging inicial: deja claro con qué configuración arrancó
    # la corrida.

    for symbol in symbols:
        # ------------------------------------------------------------------
        # 1. Resolver archivos de entrada del símbolo actual
        # ------------------------------------------------------------------
        feature_path = get_single_symbol_parquet(features_root, symbol)
        target_path = get_single_symbol_parquet(targets_root, symbol)

        # ------------------------------------------------------------------
        # 2. Leer entradas
        # ------------------------------------------------------------------
        features_df = pd.read_parquet(feature_path)
        targets_df = pd.read_parquet(target_path)

        # ------------------------------------------------------------------
        # 3. Construir artefactos de structural breaks
        # ------------------------------------------------------------------
        artifacts = build_structural_break_artifacts(
            features_df=features_df,
            targets_df=targets_df,
            break_version=cfg["version"],

            # Señales base que pueden usarse para detectar quiebres.
            return_signal_column=cfg["signals"]["return_signal_column"],
            volatility_signal_column=cfg["signals"]["volatility_signal_column"],

            # Si el pipeline usa una señal conjunta en lugar de una sola.
            use_joint_signal=cfg["signals"]["use_joint_signal"],
            joint_signal_columns=tuple(cfg["signals"]["joint_signal_columns"]),

            # Configuración del método de segmentación.
            algorithm=cfg["method"]["algorithm"],
            cost_model=cfg["method"]["cost_model"],

            # Preprocesamiento.
            dropna=cfg["preprocessing"]["dropna"],
            min_required_rows=cfg["preprocessing"]["min_required_rows"],
            standardize_joint_signal=cfg["preprocessing"]["standardize_joint_signal"],

            # Hiperparámetros de segmentación / change-point detection.
            penalty=cfg["segmentation"]["penalty"],
            min_size=cfg["segmentation"]["min_size"],
            jump=cfg["segmentation"]["jump"],
        )
        # Esta llamada centraliza la parte analítica.
        #
        # En términos conceptuales, aquí ocurre algo como:
        # - alinear features y targets,
        # - seleccionar la señal relevante,
        # - limpiar / filtrar datos,
        # - correr un algoritmo de detección de quiebres,
        # - devolver:
        #     artifacts.signal_df
        #     artifacts.events_df

        # ------------------------------------------------------------------
        # 4. Persistir resultados
        # ------------------------------------------------------------------
        signal_path, events_path = save_symbol_artifacts(
            symbol=symbol,
            break_version=cfg["version"],
            signal_df=artifacts.signal_df,
            events_df=artifacts.events_df,
            events_root=events_root,
        )

        # ------------------------------------------------------------------
        # 5. Reportar resumen del símbolo procesado
        # ------------------------------------------------------------------
        print(f"symbol={symbol.upper()}")
        print(f"feature_path={feature_path}")
        print(f"target_path={target_path}")
        print(f"signal_rows={len(artifacts.signal_df)}")
        print(f"event_rows={len(artifacts.events_df)}")
        print(f"signal_path={signal_path}")
        print(f"events_path={events_path}")

        if not artifacts.events_df.empty:
            print("break_dates:")
            print(
                artifacts.events_df["break_date"]
                .dt.date.astype(str)
                .tolist()
            )
            # Si se detectaron quiebres, imprime sus fechas en formato legible.
        else:
            print("break_dates=[]")
            # Si no hubo eventos detectados, se deja explícito.

        print("-" * 80)

    print("STRUCTURAL BREAK DETECTION: PASS")
    # Mensaje final simple de éxito.
    # Útil para identificar rápidamente en logs si el proceso terminó.


if __name__ == "__main__":
    """
    Punto de entrada estándar de Python.

    Significa:
    - Si este archivo se ejecuta directamente:
          python nombre_del_script.py
      entonces se llama `main()`.

    - Si este archivo se importa desde otro módulo:
          from scripts.detect_structural_breaks import main
      entonces `main()` NO se ejecuta automáticamente.

    Esto permite reutilizar funciones del archivo sin disparar la corrida completa.
    """
    main()