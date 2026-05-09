from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quant_platform.services.settings import load_settings


# ---------------------------------------------------------------------
# Esquemas mínimos esperados para cada artefacto.
#
# Estas constantes definen las columnas obligatorias que deben existir en:
# - el panel detallado de evaluación,
# - la tabla de métricas,
# - la tabla de matriz de confusión.
#
# El objetivo es detectar de forma temprana cualquier desviación del contrato
# de datos esperado por el pipeline de validación.
# ---------------------------------------------------------------------

PANEL_REQUIRED_COLS = {
    "instrument_id",
    "symbol",
    "date",
    "split_id",
    "dataset_role",
    "model_name",
    "future_rv_5d",
    "future_regime_5d",
    "yhat_future_rv_5d",
    "yhat_future_regime_5d",
    "threshold_low",
    "threshold_high",
    "regime_thresholds_source",
    "error",
    "abs_error",
    "sq_error",
    "qlike_term",
}

METRICS_REQUIRED_COLS = {
    "evaluation_version",
    "instrument_id",
    "symbol",
    "split_id",
    "dataset_role",
    "model_name",
    "target_family",
    "metric_name",
    "metric_value",
    "n_obs",
}

CONFUSION_REQUIRED_COLS = {
    "evaluation_version",
    "instrument_id",
    "symbol",
    "split_id",
    "dataset_role",
    "model_name",
    "y_true",
    "y_pred",
    "count",
}


def discover_single_file_per_symbol(
    root_dir: Path,
    pattern: str,
) -> dict[str, Path]:
    """
    Descubre archivos que cumplen un patrón y construye un mapeo
    `symbol -> path`, asumiendo exactamente un archivo por símbolo.

    Parameters
    ----------
    root_dir : Path
        Directorio raíz desde el cual se hará el `glob`.
    pattern : str
        Patrón de búsqueda relativo a `root_dir`.

    Returns
    -------
    dict[str, Path]
        Diccionario donde la llave es el símbolo y el valor es la ruta
        del archivo correspondiente.

    Raises
    ------
    RuntimeError
        Si para un mismo símbolo se detecta más de un archivo que coincide
        con el patrón.

    Notes
    -----
    La función asume que el símbolo se puede inferir como el nombre de la
    carpeta padre inmediata del archivo:

        <root>/<symbol>/<archivo>.parquet

    Esto permite recorrer artefactos organizados por símbolo y validar que
    haya una única pieza por categoría.
    """
    file_map: dict[str, Path] = {}

    for path in sorted(root_dir.glob(pattern)):
        symbol = path.parent.name
        if symbol in file_map:
            raise RuntimeError(
                f"Expected one file per symbol for pattern={pattern}, found duplicate for symbol={symbol}"
            )
        file_map[symbol] = path

    return file_map


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: set[str],
    df_name: str,
    path: Path,
) -> list[str]:
    """
    Verifica que un DataFrame contenga todas las columnas obligatorias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a inspeccionar.
    required_cols : set[str]
        Conjunto de columnas que deben existir.
    df_name : str
        Nombre lógico del artefacto, por ejemplo: `PANEL`, `METRICS`
        o `CONFUSION`.
    path : Path
        Ruta del archivo desde donde se cargó el DataFrame, usada para
        enriquecer los mensajes de error.

    Returns
    -------
    list[str]
        Lista de issues detectados. Si no hay problemas, la lista es vacía.

    Notes
    -----
    Esta función no lanza excepción: acumula errores como strings para que
    el pipeline de calidad pueda reportar múltiples problemas en una sola corrida.
    """
    issues: list[str] = []
    missing = sorted(required_cols - set(df.columns))
    if missing:
        issues.append(f"[{df_name}] Missing columns in {path}: {missing}")
    return issues


def check_panel_df(
    df: pd.DataFrame,
    path: Path,
    expected_models: set[str],
    expected_roles: set[str],
    expected_labels: set[str],
) -> list[str]:
    """
    Ejecuta validaciones de calidad sobre el panel detallado de comparación.

    Parameters
    ----------
    df : pd.DataFrame
        Panel de evaluación a validar.
    path : Path
        Ruta del archivo asociado.
    expected_models : set[str]
        Conjunto de nombres de modelo esperados.
    expected_roles : set[str]
        Conjunto de roles de dataset esperados, por ejemplo `validation`
        y/o `test`.
    expected_labels : set[str]
        Etiquetas válidas esperadas para los regímenes.

    Returns
    -------
    list[str]
        Lista de issues encontrados.

    Checks performed
    ----------------
    - Esquema mínimo de columnas.
    - No vacío.
    - No duplicados a nivel de fila OOS completa.
    - Conjunto de modelos correcto.
    - Conjunto de dataset roles correcto.
    - Etiquetas reales y predichas dentro del universo esperado.
    - Exactamente 2 filas por llave OOS (una por modelo).
    - Exactamente 2 modelos distintos por llave OOS.
    - Ausencia de nulos en targets y predicciones clave.
    - Valores numéricos finitos en columnas relevantes.
    - Fuente de umbrales igual a `train_only`.

    Notes
    -----
    El panel es el artefacto más rico a nivel fila, por lo que aquí se concentran
    varias validaciones estructurales y semánticas.
    """
    issues: list[str] = []
    issues.extend(validate_required_columns(df, PANEL_REQUIRED_COLS, "PANEL", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[PANEL] Empty dataframe: {path}")
        return issues

    # Llave OOS sin modelo: identifica una observación única en fecha/split/rol.
    group_key_without_model = ["instrument_id", "symbol", "date", "split_id", "dataset_role"]

    # Llave completa incluyendo el modelo: cada combinación debería ser única.
    full_row_key = group_key_without_model + ["model_name"]

    if df.duplicated(subset=full_row_key).any():
        issues.append(f"[PANEL] Duplicate full OOS rows found in {path}")

    model_set = set(df["model_name"].dropna().unique().tolist())
    if model_set != expected_models:
        issues.append(
            f"[PANEL] Unexpected model set in {path}. expected={sorted(expected_models)} got={sorted(model_set)}"
        )

    role_set = set(df["dataset_role"].dropna().unique().tolist())
    if role_set != expected_roles:
        issues.append(
            f"[PANEL] Unexpected dataset_role set in {path}. expected={sorted(expected_roles)} got={sorted(role_set)}"
        )

    true_label_set = set(df["future_regime_5d"].dropna().unique().tolist())
    pred_label_set = set(df["yhat_future_regime_5d"].dropna().unique().tolist())
    if not true_label_set.issubset(expected_labels):
        issues.append(
            f"[PANEL] Unexpected true regime labels in {path}: {sorted(true_label_set - expected_labels)}"
        )
    if not pred_label_set.issubset(expected_labels):
        issues.append(
            f"[PANEL] Unexpected predicted regime labels in {path}: {sorted(pred_label_set - expected_labels)}"
        )

    # Cada llave OOS debería aparecer exactamente dos veces:
    # una fila para el benchmark y una para el modelo ML.
    per_oos_key_counts = df.groupby(group_key_without_model).size()
    if not (per_oos_key_counts == 2).all():
        issues.append(f"[PANEL] Not every OOS key has exactly 2 model rows in {path}")

    # Además, esas dos filas deberían corresponder a dos modelos distintos.
    per_oos_key_models = df.groupby(group_key_without_model)["model_name"].nunique()
    if not (per_oos_key_models == 2).all():
        issues.append(f"[PANEL] Not every OOS key has exactly 2 distinct models in {path}")

    if df["future_rv_5d"].isna().any():
        issues.append(f"[PANEL] Null future_rv_5d values found in {path}")
    if df["yhat_future_rv_5d"].isna().any():
        issues.append(f"[PANEL] Null yhat_future_rv_5d values found in {path}")
    if df["future_regime_5d"].isna().any():
        issues.append(f"[PANEL] Null future_regime_5d values found in {path}")
    if df["yhat_future_regime_5d"].isna().any():
        issues.append(f"[PANEL] Null yhat_future_regime_5d values found in {path}")

    numeric_cols = [
        "future_rv_5d",
        "yhat_future_rv_5d",
        "threshold_low",
        "threshold_high",
        "error",
        "abs_error",
        "sq_error",
        "qlike_term",
    ]
    for col in numeric_cols:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            issues.append(f"[PANEL] Non-finite numeric values in column `{col}` of {path}")

    threshold_source_set = set(df["regime_thresholds_source"].dropna().unique().tolist())
    if threshold_source_set != {"train_only"}:
        issues.append(
            f"[PANEL] Unexpected regime_thresholds_source in {path}: {sorted(threshold_source_set)}"
        )

    return issues


def check_metrics_df(
    df: pd.DataFrame,
    path: Path,
    expected_models: set[str],
    expected_roles: set[str],
) -> list[str]:
    """
    Valida la tabla agregada de métricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de métricas.
    path : Path
        Ruta del archivo asociado.
    expected_models : set[str]
        Conjunto de nombres de modelo esperados.
    expected_roles : set[str]
        Conjunto de roles de dataset esperados.

    Returns
    -------
    list[str]
        Lista de issues detectados.

    Checks performed
    ----------------
    - Esquema mínimo.
    - No vacío.
    - Nombres de métricas esperados.
    - Familias de target esperadas.
    - Modelos esperados.
    - Roles esperados.
    - Ausencia de duplicados por grupo+métrica.
    - `metric_value` finito.
    - `n_obs` positivo.
    - Exactamente 5 métricas distintas por grupo.

    Notes
    -----
    Esta función valida coherencia tanto estructural como semántica de la
    salida agregada usada para reporting y comparación de desempeño.
    """
    issues: list[str] = []
    issues.extend(validate_required_columns(df, METRICS_REQUIRED_COLS, "METRICS", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[METRICS] Empty dataframe: {path}")
        return issues

    expected_metric_names = {"qlike", "rmse", "mae", "macro_f1", "balanced_accuracy"}
    expected_target_families = {"continuous", "discrete"}

    metric_name_set = set(df["metric_name"].dropna().unique().tolist())
    if metric_name_set != expected_metric_names:
        issues.append(
            f"[METRICS] Unexpected metric names in {path}. expected={sorted(expected_metric_names)} got={sorted(metric_name_set)}"
        )

    target_family_set = set(df["target_family"].dropna().unique().tolist())
    if target_family_set != expected_target_families:
        issues.append(
            f"[METRICS] Unexpected target families in {path}. expected={sorted(expected_target_families)} got={sorted(target_family_set)}"
        )

    model_set = set(df["model_name"].dropna().unique().tolist())
    if model_set != expected_models:
        issues.append(
            f"[METRICS] Unexpected model set in {path}. expected={sorted(expected_models)} got={sorted(model_set)}"
        )

    role_set = set(df["dataset_role"].dropna().unique().tolist())
    if role_set != expected_roles:
        issues.append(
            f"[METRICS] Unexpected dataset_role set in {path}. expected={sorted(expected_roles)} got={sorted(role_set)}"
        )

    dedup_cols = [
        "instrument_id",
        "symbol",
        "split_id",
        "dataset_role",
        "model_name",
        "metric_name",
    ]
    if df.duplicated(subset=dedup_cols).any():
        issues.append(f"[METRICS] Duplicate metric rows found in {path}")

    if not np.isfinite(df["metric_value"].to_numpy(dtype=float)).all():
        issues.append(f"[METRICS] Non-finite metric_value found in {path}")

    if (df["n_obs"].astype(int) <= 0).any():
        issues.append(f"[METRICS] Non-positive n_obs found in {path}")

    # Cada grupo (instrumento/split/rol/modelo) debe tener exactamente
    # cinco métricas distintas: qlike, rmse, mae, macro_f1, balanced_accuracy.
    per_group_metric_count = df.groupby(
        ["instrument_id", "symbol", "split_id", "dataset_role", "model_name"]
    )["metric_name"].nunique()
    if not (per_group_metric_count == 5).all():
        issues.append(f"[METRICS] Not every group has exactly 5 distinct metrics in {path}")

    return issues


def check_confusion_df(
    df: pd.DataFrame,
    path: Path,
    expected_models: set[str],
    expected_roles: set[str],
    expected_labels: set[str],
) -> list[str]:
    """
    Valida la tabla de matriz de confusión en formato largo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de matriz de confusión.
    path : Path
        Ruta del archivo asociado.
    expected_models : set[str]
        Modelos esperados.
    expected_roles : set[str]
        Roles de dataset esperados.
    expected_labels : set[str]
        Etiquetas de clase esperadas.

    Returns
    -------
    list[str]
        Lista de issues encontrados.

    Checks performed
    ----------------
    - Esquema mínimo.
    - No vacío.
    - Modelos esperados.
    - Roles esperados.
    - Etiquetas `y_true` y `y_pred` exactamente iguales a las esperadas.
    - Ausencia de duplicados por celda lógica.
    - Exactamente 9 celdas por grupo (3x3).
    - Conteos no negativos.

    Notes
    -----
    Dado que hay 3 etiquetas esperadas, cada grupo debería producir una matriz
    3x3, es decir, 9 filas en formato largo.
    """
    issues: list[str] = []
    issues.extend(validate_required_columns(df, CONFUSION_REQUIRED_COLS, "CONFUSION", path))
    if issues:
        return issues

    if df.empty:
        issues.append(f"[CONFUSION] Empty dataframe: {path}")
        return issues

    model_set = set(df["model_name"].dropna().unique().tolist())
    if model_set != expected_models:
        issues.append(
            f"[CONFUSION] Unexpected model set in {path}. expected={sorted(expected_models)} got={sorted(model_set)}"
        )

    role_set = set(df["dataset_role"].dropna().unique().tolist())
    if role_set != expected_roles:
        issues.append(
            f"[CONFUSION] Unexpected dataset_role set in {path}. expected={sorted(expected_roles)} got={sorted(role_set)}"
        )

    true_label_set = set(df["y_true"].dropna().unique().tolist())
    pred_label_set = set(df["y_pred"].dropna().unique().tolist())
    if true_label_set != expected_labels:
        issues.append(
            f"[CONFUSION] Unexpected y_true labels in {path}. expected={sorted(expected_labels)} got={sorted(true_label_set)}"
        )
    if pred_label_set != expected_labels:
        issues.append(
            f"[CONFUSION] Unexpected y_pred labels in {path}. expected={sorted(expected_labels)} got={sorted(pred_label_set)}"
        )

    dedup_cols = [
        "instrument_id",
        "symbol",
        "split_id",
        "dataset_role",
        "model_name",
        "y_true",
        "y_pred",
    ]
    if df.duplicated(subset=dedup_cols).any():
        issues.append(f"[CONFUSION] Duplicate confusion rows found in {path}")

    # Para 3 clases, cada grupo debe tener 3 x 3 = 9 celdas.
    per_group_cell_count = df.groupby(
        ["instrument_id", "symbol", "split_id", "dataset_role", "model_name"]
    ).size()
    if not (per_group_cell_count == 9).all():
        issues.append(f"[CONFUSION] Not every group has exactly 9 confusion cells in {path}")

    if (df["count"].astype(int) < 0).any():
        issues.append(f"[CONFUSION] Negative counts found in {path}")

    return issues


def check_cross_consistency(
    panel_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    symbol: str,
) -> list[str]:
    """
    Valida consistencia cruzada entre panel, métricas y confusión
    para un mismo símbolo.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel detallado.
    metrics_df : pd.DataFrame
        Tabla de métricas agregadas.
    confusion_df : pd.DataFrame
        Tabla de matriz de confusión.
    symbol : str
        Símbolo del activo, usado en mensajes de error.

    Returns
    -------
    list[str]
        Lista de issues detectados.

    Checks performed
    ----------------
    1. Los grupos presentes en panel, métricas y confusión deben coincidir.
    2. La suma total de la matriz de confusión por grupo debe coincidir
       con `n_obs` de las métricas discretas.

    Notes
    -----
    Esta función no valida cada artefacto de manera aislada, sino su
    compatibilidad mutua. Eso es crucial porque un archivo puede lucir bien
    internamente y aun así estar desalineado respecto a otro.
    """
    issues: list[str] = []

    group_cols = ["instrument_id", "symbol", "split_id", "dataset_role", "model_name"]

    panel_groups = panel_df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
    metrics_groups = metrics_df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
    confusion_groups = confusion_df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)

    if not panel_groups.equals(metrics_groups):
        issues.append(f"[CROSS] panel groups != metrics groups for symbol={symbol}")
    if not panel_groups.equals(confusion_groups):
        issues.append(f"[CROSS] panel groups != confusion groups for symbol={symbol}")

    # Nos quedamos únicamente con las métricas discretas, porque son las que
    # deberían corresponder exactamente al número total de observaciones
    # reflejadas en la matriz de confusión.
    discrete_metrics_df = metrics_df[metrics_df["target_family"] == "discrete"].copy()
    discrete_metrics_df = discrete_metrics_df[
        ["instrument_id", "symbol", "split_id", "dataset_role", "model_name", "n_obs"]
    ].drop_duplicates()

    confusion_sum_df = (
        confusion_df.groupby(group_cols, as_index=False)["count"]
        .sum()
        .rename(columns={"count": "confusion_total"})
    )

    merged = discrete_metrics_df.merge(
        confusion_sum_df,
        on=group_cols,
        how="outer",
        validate="one_to_one",
    )

    if merged["n_obs"].isna().any() or merged["confusion_total"].isna().any():
        issues.append(f"[CROSS] Missing cross-match between discrete metrics and confusion totals for symbol={symbol}")
    else:
        if not (merged["n_obs"].astype(int) == merged["confusion_total"].astype(int)).all():
            issues.append(f"[CROSS] discrete n_obs != confusion total for symbol={symbol}")

    return issues


def main() -> None:
    """
    Ejecuta el pipeline completo de chequeos de calidad para los artefactos
    de comparación de modelos.

    Workflow
    --------
    1. Carga la configuración.
    2. Obtiene rutas de artefactos desde `settings["evaluation"]["outputs"]`.
    3. Construye los conjuntos esperados de:
       - modelos,
       - roles,
       - etiquetas.
    4. Descubre archivos de panel, métricas y confusión por símbolo.
    5. Verifica que los símbolos coincidan entre las tres familias de artefactos.
    6. Para cada símbolo:
       - carga los tres dataframes,
       - valida cada uno por separado,
       - valida consistencia cruzada entre ellos.
    7. Si hay issues, imprime todos y termina con código de error 1.
    8. Si no hay issues, imprime PASS y un resumen de archivos validados.

    Returns
    -------
    None

    Notes
    -----
    Esta función está diseñada como un quality gate del pipeline:
    si algo falla, el proceso termina explícitamente con `sys.exit(1)`,
    lo cual es útil para CI/CD, scripts de validación o automatizaciones.
    """
    settings = load_settings()
    evaluation_cfg = settings["evaluation"]

    metrics_root = Path(evaluation_cfg["outputs"]["metrics_dir"])
    confusion_root = Path(evaluation_cfg["outputs"]["confusion_matrices_dir"])

    expected_models = {
        evaluation_cfg["comparison"]["benchmark_model_name"],
        evaluation_cfg["comparison"]["ml_model_name"],
    }
    expected_roles = set(evaluation_cfg["score_roles"])
    expected_labels = set(evaluation_cfg["discrete"]["labels"])

    panel_files = discover_single_file_per_symbol(
        metrics_root,
        "*/*_model_comparison_panel_*.parquet",
    )
    metrics_files = discover_single_file_per_symbol(
        metrics_root,
        "*/*_model_comparison_metrics_*.parquet",
    )
    confusion_files = discover_single_file_per_symbol(
        confusion_root,
        "*/*_model_comparison_confusion_*.parquet",
    )

    issues: list[str] = []

    all_symbols = sorted(set(panel_files) | set(metrics_files) | set(confusion_files))
    if not all_symbols:
        print("MODEL COMPARISON QUALITY CHECKS: FAIL")
        print("[DISCOVERY] No model comparison artifacts found.")
        sys.exit(1)

    # Si los conjuntos de símbolos no coinciden entre las tres familias
    # de artefactos, ya existe una inconsistencia estructural del pipeline.
    if set(panel_files) != set(metrics_files) or set(panel_files) != set(confusion_files):
        issues.append(
            "[DISCOVERY] Symbol sets do not match across panel/metrics/confusion artifact roots."
        )

    for symbol in all_symbols:
        panel_path = panel_files.get(symbol)
        metrics_path = metrics_files.get(symbol)
        confusion_path = confusion_files.get(symbol)

        if panel_path is None:
            issues.append(f"[DISCOVERY] Missing panel artifact for symbol={symbol}")
            continue
        if metrics_path is None:
            issues.append(f"[DISCOVERY] Missing metrics artifact for symbol={symbol}")
            continue
        if confusion_path is None:
            issues.append(f"[DISCOVERY] Missing confusion artifact for symbol={symbol}")
            continue

        panel_df = pd.read_parquet(panel_path)
        metrics_df = pd.read_parquet(metrics_path)
        confusion_df = pd.read_parquet(confusion_path)

        # Validaciones individuales por artefacto.
        issues.extend(
            check_panel_df(
                df=panel_df,
                path=panel_path,
                expected_models=expected_models,
                expected_roles=expected_roles,
                expected_labels=expected_labels,
            )
        )
        issues.extend(
            check_metrics_df(
                df=metrics_df,
                path=metrics_path,
                expected_models=expected_models,
                expected_roles=expected_roles,
            )
        )
        issues.extend(
            check_confusion_df(
                df=confusion_df,
                path=confusion_path,
                expected_models=expected_models,
                expected_roles=expected_roles,
                expected_labels=expected_labels,
            )
        )

        # Validaciones cruzadas entre artefactos.
        issues.extend(
            check_cross_consistency(
                panel_df=panel_df,
                metrics_df=metrics_df,
                confusion_df=confusion_df,
                symbol=symbol,
            )
        )

    if issues:
        print("MODEL COMPARISON QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    print("MODEL COMPARISON QUALITY CHECKS: PASS")
    for symbol in sorted(all_symbols):
        print(f"[PANEL] OK -> {panel_files[symbol]}")
        print(f"[METRICS] OK -> {metrics_files[symbol]}")
        print(f"[CONFUSION] OK -> {confusion_files[symbol]}")


if __name__ == "__main__":
    """
    Punto de entrada del script cuando se ejecuta directamente.

    Evita que `main()` corra al importar este archivo como módulo desde
    otro lugar del proyecto.
    """
    main()