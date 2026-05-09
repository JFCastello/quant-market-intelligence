from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.evaluation import build_model_decision_artifacts
from quant_platform.services.settings import load_settings


def ensure_directory(path: Path) -> None:
    """
    Garantiza que el directorio de destino exista antes de escribir archivos.

    Parameters
    ----------
    path : Path
        Ruta del directorio que se debe crear o asegurar.

    Returns
    -------
    None

    Notes
    -----
    - `parents=True` permite crear también los directorios padres faltantes.
    - `exist_ok=True` evita que falle si el directorio ya existe.

    Esta función encapsula una operación pequeña pero recurrente en pipelines
    de persistencia: preparar la estructura de salida antes de guardar artefactos.
    """
    path.mkdir(parents=True, exist_ok=True)


def discover_metric_paths(metrics_root: Path) -> list[Path]:
    """
    Descubre todos los archivos parquet de métricas de comparación de modelos
    que existen bajo el directorio raíz de métricas.

    Parameters
    ----------
    metrics_root : Path
        Directorio raíz donde se espera encontrar archivos de métricas
        organizados por símbolo.

    Returns
    -------
    list[Path]
        Lista ordenada de rutas a archivos parquet de métricas.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra ningún archivo que coincida con el patrón esperado.

    Notes
    -----
    El patrón usado es:

        */*_model_comparison_metrics_*.parquet

    lo que implica una convención de organización tipo:

        metrics_root/
            spy/
                spy_..._model_comparison_metrics_v1.parquet
            tlt/
                tlt_..._model_comparison_metrics_v1.parquet

    Esta función no valida todavía el contenido interno de los archivos;
    sólo descubre qué artefactos existen para ser cargados y concatenados.
    """
    metric_paths = sorted(metrics_root.glob("*/*_model_comparison_metrics_*.parquet"))
    if not metric_paths:
        raise FileNotFoundError(
            f"No model comparison metrics parquets found under: {metrics_root}"
        )
    return metric_paths


def build_window_token_from_metrics(metrics_df: pd.DataFrame) -> str:
    """
    Construye un identificador textual de ventana para los artefactos de decisión
    a partir del DataFrame de métricas agregado.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Tabla consolidada de métricas provenientes de múltiples símbolos.

    Returns
    -------
    str
        Token descriptivo que se usará dentro del nombre de los archivos
        de salida. En la implementación actual, siempre devuelve
        `"all_symbols"`.

    Raises
    ------
    ValueError
        Si `metrics_df` está vacío o si no contiene símbolos válidos.

    Notes
    -----
    Aunque esta función inspecciona el DataFrame, actualmente no construye un
    token dinámico por rango temporal ni por lista de símbolos; simplemente
    valida que el DataFrame tenga contenido y devuelve el literal:

        "all_symbols"

    Conceptualmente, esto deja preparada una posible evolución futura donde el
    token podría incorporar más información de trazabilidad.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; cannot build window token.")

    symbols = sorted(metrics_df["symbol"].dropna().unique().tolist())
    if not symbols:
        raise ValueError("No symbols found in metrics_df.")

    return "all_symbols"


def save_decision_artifacts(
    decision_panel_df: pd.DataFrame,
    decision_summary_df: pd.DataFrame,
    decision_reasons_df: pd.DataFrame,
    decision_root: Path,
    evaluation_version: str,
) -> tuple[Path, Path, Path]:
    """
    Guarda en disco los tres artefactos del proceso de decisión del modelo:
    panel, resumen y razones.

    Parameters
    ----------
    decision_panel_df : pd.DataFrame
        Tabla intermedia donde benchmark y ML aparecen alineados por métrica.
    decision_summary_df : pd.DataFrame
        Tabla agregada con la decisión final por símbolo y sus indicadores clave.
    decision_reasons_df : pd.DataFrame
        Tabla en formato largo que documenta qué reglas pasaron y cuáles no.
    decision_root : Path
        Directorio raíz donde se guardarán los artefactos de decisión.
    evaluation_version : str
        Versión lógica de la evaluación/decisión, usada en los nombres de archivo.

    Returns
    -------
    tuple[Path, Path, Path]
        Rutas de salida en el orden:
        - panel_path
        - summary_path
        - reasons_path

    Workflow
    --------
    1. Asegura que exista el directorio de salida.
    2. Define un token de ventana común (`all_symbols`).
    3. Construye nombres de archivo estables y trazables.
    4. Guarda cada artefacto en formato parquet.
    5. Devuelve las rutas generadas.

    Notes
    -----
    Aquí los artefactos no se guardan por símbolo individual, sino como una
    vista agregada sobre todos los símbolos disponibles en la corrida.
    Por eso el token fijo `all_symbols` tiene sentido en este diseño.
    """
    ensure_directory(decision_root)

    # En esta etapa la decisión se toma de manera agregada sobre todos los
    # símbolos presentes en la tabla de métricas consolidada.
    window_token = "all_symbols"

    panel_path = decision_root / f"{window_token}_decision_panel_{evaluation_version}.parquet"
    summary_path = decision_root / f"{window_token}_decision_summary_{evaluation_version}.parquet"
    reasons_path = decision_root / f"{window_token}_decision_reasons_{evaluation_version}.parquet"

    # Se guarda sin índice para producir archivos más limpios y evitar
    # columnas índice accidentales al reabrir los parquets.
    decision_panel_df.to_parquet(panel_path, index=False)
    decision_summary_df.to_parquet(summary_path, index=False)
    decision_reasons_df.to_parquet(reasons_path, index=False)

    return panel_path, summary_path, reasons_path


def main() -> None:
    """
    Ejecuta el pipeline completo de construcción de artefactos de decisión
    sobre promoción o no promoción del modelo ML.

    Workflow
    --------
    1. Carga la configuración global del proyecto.
    2. Extrae la sección `decision` del settings.
    3. Localiza todos los archivos de métricas generados previamente
       por la etapa de evaluación.
    4. Carga y concatena esas métricas en un único DataFrame.
    5. Construye los artefactos de decisión usando las reglas definidas
       en configuración.
    6. Guarda en disco:
       - el panel de decisión,
       - el resumen de decisión,
       - la tabla de razones.
    7. Imprime un resumen de ejecución y una vista tabular del resumen final.
    8. Finaliza con mensaje de éxito.

    Returns
    -------
    None

    Notes
    -----
    Esta función es el punto de entrada operativo del script. Su rol es de
    orquestación: no calcula las métricas base, sino que toma como entrada
    los artefactos ya evaluados y aplica una capa de gobernanza/decisión
    sobre ellos.
    """
    settings = load_settings()
    decision_cfg = settings["decision"]

    # Directorio donde viven las métricas agregadas provenientes de la etapa
    # de comparación benchmark vs ML.
    metrics_root = Path(decision_cfg["inputs"]["metrics_dir"])

    # Directorio donde se escribirán los artefactos finales de decisión.
    decision_root = Path(decision_cfg["outputs"]["decision_dir"])

    # Se descubren todos los archivos de métricas y luego se consolidan
    # verticalmente en un único DataFrame.
    metric_paths = discover_metric_paths(metrics_root)
    metrics_df = pd.concat(
        [pd.read_parquet(path) for path in metric_paths],
        axis=0,
        ignore_index=True,
    )

    # Se construyen los artefactos de decisión usando parámetros explícitos
    # provenientes de la configuración:
    # - nombres de modelos,
    # - roles a considerar,
    # - métrica principal,
    # - guardrails discretos,
    # - umbrales de promoción,
    # - estado de calibración.
    artifacts = build_model_decision_artifacts(
        metrics_df=metrics_df,
        benchmark_model_name=decision_cfg["comparison"]["benchmark_model_name"],
        ml_model_name=decision_cfg["comparison"]["ml_model_name"],
        score_roles=tuple(decision_cfg["aggregation"]["score_roles"]),
        primary_metric=decision_cfg["comparison"]["primary_metric"],
        discrete_guardrail_metric=decision_cfg["comparison"]["discrete_guardrail_metric"],
        secondary_discrete_metric=decision_cfg["comparison"]["secondary_discrete_metric"],
        min_relative_qlike_improvement=decision_cfg["promotion_rule"]["min_relative_qlike_improvement"],
        max_macro_f1_drop=decision_cfg["promotion_rule"]["max_macro_f1_drop"],
        max_balanced_accuracy_drop=decision_cfg["promotion_rule"]["max_balanced_accuracy_drop"],
        calibration_available=decision_cfg["calibration"]["is_available"],
    )

    # Persistencia de los tres artefactos producidos por la etapa de decisión.
    panel_path, summary_path, reasons_path = save_decision_artifacts(
        decision_panel_df=artifacts.decision_panel_df,
        decision_summary_df=artifacts.decision_summary_df,
        decision_reasons_df=artifacts.decision_reasons_df,
        decision_root=decision_root,
        evaluation_version=decision_cfg["version"],
    )

    # Resumen de alto nivel de la corrida, útil para logs, debugging
    # y verificación rápida del resultado del script.
    print("MODEL DECISION BUILD START")
    print(f"metric_files={len(metric_paths)}")
    print(f"decision_panel_rows={len(artifacts.decision_panel_df)}")
    print(f"decision_summary_rows={len(artifacts.decision_summary_df)}")
    print(f"decision_reasons_rows={len(artifacts.decision_reasons_df)}")
    print(f"panel_path={panel_path}")
    print(f"summary_path={summary_path}")
    print(f"reasons_path={reasons_path}")
    print("-" * 80)

    # Se imprime una vista compacta del resumen final de decisión,
    # enfocada en las métricas y reglas más importantes para determinar
    # si el modelo ML se promueve o no.
    print("DECISION SUMMARY")
    print(
        artifacts.decision_summary_df[
            [
                "symbol",
                "relative_qlike_improvement_mean",
                "macro_f1_delta",
                "balanced_accuracy_delta",
                "qlike_pass",
                "macro_f1_pass",
                "balanced_accuracy_pass",
                "calibration_status",
                "decision",
            ]
        ]
        .sort_values("symbol")
        .to_string(index=False)
    )

    print("-" * 80)
    print("MODEL DECISION BUILD: PASS")


if __name__ == "__main__":
    """
    Punto de entrada del script cuando el archivo se ejecuta directamente.

    Esto evita que `main()` se ejecute automáticamente si este módulo es
    importado desde otro archivo del proyecto.
    """
    main()