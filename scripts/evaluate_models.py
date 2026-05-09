from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.evaluation import build_model_comparison_artifacts
from quant_platform.services.settings import load_settings


def ensure_directory(path: Path) -> None:
    """
    Garantiza que un directorio exista en disco.

    Parameters
    ----------
    path : Path
        Ruta del directorio que se quiere asegurar.

    Returns
    -------
    None

    Notes
    -----
    - `parents=True` permite crear también directorios padres faltantes.
    - `exist_ok=True` evita que falle si el directorio ya existe.

    Esta función encapsula una operación simple pero muy común en pipelines:
    preparar la estructura de salida antes de intentar guardar archivos.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_single_symbol_parquet(root_dir: Path, symbol: str) -> Path:
    """
    Obtiene el único archivo `.parquet` esperado para un símbolo dado
    dentro de un directorio raíz.

    Parameters
    ----------
    root_dir : Path
        Directorio raíz bajo el cual se espera una subcarpeta por símbolo.
    symbol : str
        Símbolo cuyo archivo parquet se quiere localizar.

    Returns
    -------
    Path
        Ruta al único archivo parquet encontrado.

    Raises
    ------
    FileNotFoundError
        Si la carpeta del símbolo no existe o si no contiene ningún parquet.
    RuntimeError
        Si se encuentra más de un parquet, rompiendo la suposición de que
        debe existir exactamente uno por símbolo.

    Notes
    -----
    La función impone una convención de estructura de datos:

        root_dir/
            spy/
                <unico_archivo>.parquet
            tlt/
                <unico_archivo>.parquet

    Esto es útil para evitar ambigüedades y mantener un pipeline determinista.
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


def build_oos_window_token(panel_df: pd.DataFrame) -> str:
    """
    Construye un token de ventana temporal OOS (out-of-sample) a partir
    de las fechas mínimas y máximas presentes en un panel.

    Parameters
    ----------
    panel_df : pd.DataFrame
        DataFrame de evaluación que debe contener la columna `date`.

    Returns
    -------
    str
        Token en formato:
        `<fecha_min>_<fecha_max>`

        por ejemplo:
        `2020-01-01_2024-12-31`

    Raises
    ------
    ValueError
        Si el DataFrame está vacío.

    Notes
    -----
    Este token se usa para nombrar archivos de salida de forma trazable,
    permitiendo identificar rápidamente el rango temporal evaluado sin
    necesidad de abrir el archivo.
    """
    if panel_df.empty:
        raise ValueError("evaluation_panel_df is empty; cannot build output window token.")

    min_date = pd.to_datetime(panel_df["date"]).min().date().isoformat()
    max_date = pd.to_datetime(panel_df["date"]).max().date().isoformat()
    return f"{min_date}_{max_date}"


def save_symbol_artifacts(
    symbol: str,
    evaluation_version: str,
    panel_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    metrics_root: Path,
    confusion_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Guarda en disco los artefactos de evaluación de un símbolo:
    panel, métricas y matriz de confusión.

    Parameters
    ----------
    symbol : str
        Símbolo del activo evaluado.
    evaluation_version : str
        Versión lógica de la evaluación, usada en los nombres de archivo.
    panel_df : pd.DataFrame
        Panel detallado de comparación entre modelos.
    metrics_df : pd.DataFrame
        Tabla agregada de métricas.
    confusion_df : pd.DataFrame
        Tabla en formato largo con la matriz de confusión.
    metrics_root : Path
        Directorio raíz donde se guardarán panel y métricas.
    confusion_root : Path
        Directorio raíz donde se guardará la matriz de confusión.

    Returns
    -------
    tuple[Path, Path, Path]
        Rutas de salida en el orden:
        - panel_path
        - metrics_path
        - confusion_path

    Workflow
    --------
    1. Crea los directorios de salida si no existen.
    2. Construye un token temporal OOS a partir de `panel_df`.
    3. Arma nombres de archivo reproducibles y expresivos.
    4. Guarda cada artefacto en formato parquet.
    5. Devuelve las rutas resultantes.

    Notes
    -----
    Se guardan panel y métricas bajo `metrics_root/<symbol>/`,
    mientras que la confusión se guarda bajo `confusion_root/<symbol>/`.
    Esto separa lógicamente salidas tabulares generales de matrices
    de confusión.
    """
    symbol_metrics_dir = metrics_root / symbol
    symbol_confusion_dir = confusion_root / symbol

    ensure_directory(symbol_metrics_dir)
    ensure_directory(symbol_confusion_dir)

    # El token temporal permite que el nombre del archivo capture
    # explícitamente el rango OOS cubierto por el panel.
    window_token = build_oos_window_token(panel_df)

    panel_path = (
        symbol_metrics_dir
        / f"{symbol}_{window_token}_model_comparison_panel_{evaluation_version}.parquet"
    )
    metrics_path = (
        symbol_metrics_dir
        / f"{symbol}_{window_token}_model_comparison_metrics_{evaluation_version}.parquet"
    )
    confusion_path = (
        symbol_confusion_dir
        / f"{symbol}_{window_token}_model_comparison_confusion_{evaluation_version}.parquet"
    )

    # Persistimos cada artefacto sin índice para dejar archivos más limpios
    # y evitar columnas índice accidentales al reabrirlos.
    panel_df.to_parquet(panel_path, index=False)
    metrics_df.to_parquet(metrics_path, index=False)
    confusion_df.to_parquet(confusion_path, index=False)

    return panel_path, metrics_path, confusion_path


def discover_symbols(benchmark_regimes_root: Path) -> list[str]:
    """
    Descubre automáticamente los símbolos disponibles a partir de las
    subcarpetas presentes en el directorio raíz del benchmark.

    Parameters
    ----------
    benchmark_regimes_root : Path
        Directorio raíz donde se espera una subcarpeta por símbolo.

    Returns
    -------
    list[str]
        Lista ordenada de nombres de símbolo detectados.

    Raises
    ------
    FileNotFoundError
        Si el directorio raíz no existe.
    RuntimeError
        Si no se encuentra ninguna subcarpeta de símbolo.

    Notes
    -----
    Esta función asume que cada carpeta hija representa un símbolo.
    No inspecciona todavía si cada símbolo tiene todos los archivos requeridos;
    sólo descubre candidatos a procesar.
    """
    if not benchmark_regimes_root.exists():
        raise FileNotFoundError(
            f"benchmark_regimes root does not exist: {benchmark_regimes_root}"
        )

    symbols = sorted(
        path.name
        for path in benchmark_regimes_root.iterdir()
        if path.is_dir()
    )

    if not symbols:
        raise RuntimeError(
            f"No symbol directories found under: {benchmark_regimes_root}"
        )

    return symbols


def main() -> None:
    """
    Ejecuta el pipeline completo de construcción de artefactos de evaluación
    para todos los símbolos descubiertos.

    Workflow
    --------
    1. Carga la configuración global desde `settings`.
    2. Extrae rutas de entrada y salida desde la sección `evaluation`.
    3. Descubre los símbolos disponibles.
    4. Para cada símbolo:
       - localiza sus archivos parquet de benchmark, ML y targets;
       - los carga en DataFrames;
       - construye los artefactos de comparación;
       - guarda los resultados en disco;
       - imprime un resumen de ejecución.
    5. Imprime un mensaje final de éxito.

    Returns
    -------
    None

    Notes
    -----
    Esta función es el punto de entrada operativo del script.
    Su responsabilidad es de orquestación, no de cálculo estadístico detallado.
    Delega:
    - construcción de artefactos a `build_model_comparison_artifacts`,
    - persistencia a `save_symbol_artifacts`,
    - descubrimiento de símbolos y archivos a funciones auxiliares.
    """
    settings = load_settings()
    evaluation_cfg = settings["evaluation"]

    # Rutas de entrada: cada una apunta al árbol donde vive el insumo
    # correspondiente del pipeline de evaluación.
    benchmark_regimes_root = Path(evaluation_cfg["inputs"]["benchmark_regimes_dir"])
    ml_forecasts_root = Path(evaluation_cfg["inputs"]["ml_forecasts_dir"])
    regime_targets_root = Path(evaluation_cfg["inputs"]["regime_targets_dir"])

    # Rutas de salida para métricas y matrices de confusión.
    metrics_root = Path(evaluation_cfg["outputs"]["metrics_dir"])
    confusion_root = Path(evaluation_cfg["outputs"]["confusion_matrices_dir"])

    evaluation_version = evaluation_cfg["version"]
    labels = tuple(evaluation_cfg["discrete"]["labels"])

    # Descubre qué símbolos se van a procesar.
    symbols = discover_symbols(benchmark_regimes_root)

    print("MODEL EVALUATION BUILD START")
    print(f"symbols={symbols}")
    print(f"evaluation_version={evaluation_version}")
    print(f"labels={labels}")
    print("-" * 80)

    for symbol in symbols:
        # Para cada símbolo, se espera exactamente un parquet por fuente.
        benchmark_path = get_single_symbol_parquet(benchmark_regimes_root, symbol)
        ml_path = get_single_symbol_parquet(ml_forecasts_root, symbol)
        regime_targets_path = get_single_symbol_parquet(regime_targets_root, symbol)

        # Carga de insumos tabulares desde parquet.
        benchmark_df = pd.read_parquet(benchmark_path)
        ml_df = pd.read_parquet(ml_path)
        regime_targets_df = pd.read_parquet(regime_targets_path)

        # Construcción del panel unificado, métricas y confusión.
        artifacts = build_model_comparison_artifacts(
            benchmark_regimes_df=benchmark_df,
            ml_forecasts_df=ml_df,
            regime_targets_df=regime_targets_df,
            evaluation_version=evaluation_version,
            labels=labels,
        )

        # Persistencia de artefactos por símbolo.
        panel_path, metrics_path, confusion_path = save_symbol_artifacts(
            symbol=symbol,
            evaluation_version=evaluation_version,
            panel_df=artifacts.evaluation_panel_df,
            metrics_df=artifacts.metrics_df,
            confusion_df=artifacts.confusion_df,
            metrics_root=metrics_root,
            confusion_root=confusion_root,
        )

        # Reporte textual de lo producido para trazabilidad en consola.
        print(f"symbol={symbol.upper()}")
        print(f"benchmark_path={benchmark_path}")
        print(f"ml_path={ml_path}")
        print(f"regime_targets_path={regime_targets_path}")
        print(f"evaluation_panel_rows={len(artifacts.evaluation_panel_df)}")
        print(f"metrics_rows={len(artifacts.metrics_df)}")
        print(f"confusion_rows={len(artifacts.confusion_df)}")
        print(f"panel_path={panel_path}")
        print(f"metrics_path={metrics_path}")
        print(f"confusion_path={confusion_path}")
        print("-" * 80)

    print("MODEL EVALUATION BUILD: PASS")


if __name__ == "__main__":
    """
    Punto de entrada cuando el archivo se ejecuta directamente como script.

    Esto evita que `main()` se ejecute automáticamente si este módulo es
    importado desde otro lugar del proyecto.
    """
    main()