from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelDecisionArtifacts:
    """
    Contenedor inmutable de los artefactos finales del proceso de decisión
    sobre si promover o no el modelo ML frente al benchmark.

    Attributes
    ----------
    decision_panel_df : pd.DataFrame
        Tabla intermedia a nivel de métrica donde benchmark y modelo ML ya
        aparecen alineados lado a lado para facilitar comparaciones directas.

    decision_summary_df : pd.DataFrame
        Resumen agregado por instrumento/símbolo con los valores medios de
        las métricas relevantes, el resultado de cada regla de decisión y
        la decisión final de promoción.

    decision_reasons_df : pd.DataFrame
        Tabla en formato largo que descompone la decisión final en reglas
        individuales, mostrando para cada una:
        - el umbral esperado,
        - el valor observado,
        - si la regla pasó o no.
    """
    decision_panel_df: pd.DataFrame
    decision_summary_df: pd.DataFrame
    decision_reasons_df: pd.DataFrame


# ---------------------------------------------------------------------
# Esquema mínimo esperado para la tabla de métricas de entrada.
#
# Este módulo no trabaja sobre predicciones fila a fila, sino sobre la tabla
# agregada de métricas producida por la etapa previa de evaluación.
# Por eso, aquí se valida que esa tabla tenga el contrato de columnas esperado.
# ---------------------------------------------------------------------
REQUIRED_METRICS_COLS = {
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


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: set[str],
    df_name: str,
) -> None:
    """
    Verifica que un DataFrame contenga todas las columnas obligatorias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    required_cols : set[str]
        Conjunto de nombres de columnas que deben existir.
    df_name : str
        Nombre lógico del DataFrame, usado para construir mensajes de error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Si falta al menos una columna requerida.

    Notes
    -----
    Esta validación permite fallar temprano y con claridad cuando el contrato
    entre etapas del pipeline se rompe.
    """
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _validate_metric_uniqueness(metrics_df: pd.DataFrame) -> None:
    """
    Verifica que no existan filas duplicadas de métricas para una misma clave
    lógica de decisión.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Tabla de métricas agregadas.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Si existen métricas duplicadas para la misma combinación de:
        instrumento, símbolo, split, rol, modelo y nombre de métrica.

    Notes
    -----
    Esta validación es importante porque el paso siguiente hace un pivot.
    Si hubiera duplicados, el alineamiento entre benchmark y ML podría volverse
    ambiguo o producir resultados engañosos.
    """
    dedup_cols = [
        "instrument_id",
        "symbol",
        "split_id",
        "dataset_role",
        "model_name",
        "metric_name",
    ]
    duplicate_count = int(metrics_df.duplicated(subset=dedup_cols).sum())
    if duplicate_count != 0:
        raise ValueError(
            f"metrics_df contains duplicated metric rows for decision keys: {duplicate_count}"
        )


def _build_decision_panel_df(
    metrics_df: pd.DataFrame,
    benchmark_model_name: str,
    ml_model_name: str,
    score_roles: Sequence[str],
) -> pd.DataFrame:
    """
    Construye el panel intermedio de decisión alineando benchmark y modelo ML
    lado a lado para cada métrica relevante.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Tabla de métricas agregadas producida por la etapa de evaluación.
    benchmark_model_name : str
        Nombre del modelo benchmark.
    ml_model_name : str
        Nombre del modelo ML a comparar.
    score_roles : Sequence[str]
        Roles del dataset que se consideran válidos para la decisión,
        por ejemplo `validation` y `test`.

    Returns
    -------
    pd.DataFrame
        Panel pivotado donde, para cada combinación de:
        - evaluation_version
        - instrument_id
        - symbol
        - split_id
        - dataset_role
        - metric_name

        se tienen columnas con:
        - valor del benchmark,
        - valor del ML,
        - dirección de optimización,
        - diferencias absolutas,
        - mejora relativa versus benchmark.

    Workflow
    --------
    1. Filtra únicamente los roles y modelos relevantes.
    2. Hace un pivot para alinear benchmark y ML en columnas separadas.
    3. Renombra esas columnas a nombres estandarizados.
    4. Determina si cada métrica se optimiza hacia abajo o hacia arriba.
    5. Calcula diferencias absolutas y mejora relativa.

    Notes
    -----
    Esta tabla es la base cuantitativa del proceso de decisión.
    No toma todavía la decisión final, pero deja preparados todos los números
    necesarios para hacerlo.
    """
    filtered_df = metrics_df.loc[
        metrics_df["dataset_role"].isin(score_roles)
        & metrics_df["model_name"].isin([benchmark_model_name, ml_model_name])
    ].copy()

    # Se pivotea para que benchmark y ML queden comparables fila a fila
    # dentro de la misma métrica y contexto de evaluación.
    pivot_df = (
        filtered_df.pivot_table(
            index=[
                "evaluation_version",
                "instrument_id",
                "symbol",
                "split_id",
                "dataset_role",
                "metric_name",
            ],
            columns="model_name",
            values="metric_value",
            aggfunc="first",
        )
        .reset_index()
    )

    pivot_df.columns.name = None

    expected_cols = {benchmark_model_name, ml_model_name}
    missing_model_cols = sorted(expected_cols - set(pivot_df.columns))
    if missing_model_cols:
        raise ValueError(
            f"decision panel pivot is missing model columns: {missing_model_cols}"
        )

    pivot_df = pivot_df.rename(
        columns={
            benchmark_model_name: "benchmark_metric_value",
            ml_model_name: "ml_metric_value",
        }
    )

    # Métricas donde menor es mejor.
    lower_is_better_metrics = {"qlike", "rmse", "mae"}

    # Métricas donde mayor es mejor.
    higher_is_better_metrics = {"macro_f1", "balanced_accuracy"}

    # Se guarda explícitamente la dirección de optimización para que el panel
    # sea autoexplicativo y usable downstream.
    pivot_df["optimization_direction"] = np.where(
        pivot_df["metric_name"].isin(lower_is_better_metrics),
        "lower_is_better",
        "higher_is_better",
    )

    # Diferencia simple: ML menos benchmark.
    pivot_df["ml_minus_benchmark"] = (
        pivot_df["ml_metric_value"] - pivot_df["benchmark_metric_value"]
    )

    # Diferencia en sentido inverso: benchmark menos ML.
    pivot_df["benchmark_minus_ml"] = (
        pivot_df["benchmark_metric_value"] - pivot_df["ml_metric_value"]
    )

    # Se usa el valor absoluto del benchmark como denominador para que la mejora
    # relativa sea interpretable incluso si el benchmark tuviera signo negativo.
    # Si el benchmark fuera 0, se reemplaza por NaN para evitar división por cero.
    abs_benchmark = pivot_df["benchmark_metric_value"].abs().replace(0.0, np.nan)

    # La mejora relativa depende de la dirección de optimización:
    # - si menor es mejor, mejora = (benchmark - ml) / |benchmark|
    # - si mayor es mejor, mejora = (ml - benchmark) / |benchmark|
    pivot_df["relative_improvement_vs_benchmark"] = np.where(
        pivot_df["metric_name"].isin(lower_is_better_metrics),
        pivot_df["benchmark_minus_ml"] / abs_benchmark,
        pivot_df["ml_minus_benchmark"] / abs_benchmark,
    )

    return pivot_df


def _get_mean_metric(
    group_df: pd.DataFrame,
    metric_name: str,
    value_col: str,
) -> float:
    """
    Extrae la media de una métrica específica dentro de un grupo.

    Parameters
    ----------
    group_df : pd.DataFrame
        Subconjunto de datos correspondiente a un grupo de decisión.
    metric_name : str
        Nombre de la métrica a buscar.
    value_col : str
        Columna numérica cuyo promedio se quiere calcular.

    Returns
    -------
    float
        Promedio de la métrica solicitada dentro del grupo.
        Devuelve `np.nan` si la métrica no está presente.

    Notes
    -----
    Esta función encapsula una operación repetida varias veces en la
    construcción del resumen de decisión.
    """
    metric_df = group_df.loc[group_df["metric_name"] == metric_name]
    if metric_df.empty:
        return np.nan
    return float(metric_df[value_col].mean())


def _build_decision_summary_df(
    decision_panel_df: pd.DataFrame,
    benchmark_model_name: str,
    ml_model_name: str,
    primary_metric: str,
    discrete_guardrail_metric: str,
    secondary_discrete_metric: str,
    min_relative_qlike_improvement: float,
    max_macro_f1_drop: float,
    max_balanced_accuracy_drop: float,
    calibration_available: bool,
) -> pd.DataFrame:
    """
    Construye el resumen agregado de decisión por instrumento/símbolo.

    Parameters
    ----------
    decision_panel_df : pd.DataFrame
        Panel intermedio con benchmark y ML ya alineados.
    benchmark_model_name : str
        Nombre del benchmark.
    ml_model_name : str
        Nombre del modelo ML.
    primary_metric : str
        Métrica principal que gobierna la mejora requerida para promoción.
        En este diseño suele ser `qlike`.
    discrete_guardrail_metric : str
        Métrica discreta principal usada como guardrail, típicamente `macro_f1`.
    secondary_discrete_metric : str
        Segunda métrica discreta usada como guardrail, típicamente
        `balanced_accuracy`.
    min_relative_qlike_improvement : float
        Mejora relativa mínima exigida en la métrica principal.
    max_macro_f1_drop : float
        Máxima caída tolerada en macro F1.
    max_balanced_accuracy_drop : float
        Máxima caída tolerada en balanced accuracy.
    calibration_available : bool
        Indica si ya existe una evaluación de calibración disponible.

    Returns
    -------
    pd.DataFrame
        Tabla resumen por símbolo con:
        - medias benchmark/ML de métricas clave,
        - deltas,
        - resultado de reglas,
        - estado de calibración,
        - decisión final,
        - contadores descriptivos.

    Decision logic
    --------------
    El modelo ML se promueve sólo si:
    1. alcanza la mejora relativa mínima en la métrica principal, y
    2. no empeora más allá de lo permitido en macro F1, y
    3. no empeora más allá de lo permitido en balanced accuracy.

    Notes
    -----
    Esta función realiza la agregación final de evidencia cuantitativa
    para cada símbolo. La decisión se toma a nivel de grupo
    (`evaluation_version`, `instrument_id`, `symbol`).
    """
    summary_rows: list[dict[str, object]] = []

    group_cols = ["evaluation_version", "instrument_id", "symbol"]

    for group_keys, group_df in decision_panel_df.groupby(group_cols, dropna=False):
        evaluation_version, instrument_id, symbol = group_keys

        # Métrica principal: normalmente QLIKE.
        benchmark_qlike_mean = _get_mean_metric(
            group_df, primary_metric, "benchmark_metric_value"
        )
        ml_qlike_mean = _get_mean_metric(
            group_df, primary_metric, "ml_metric_value"
        )
        qlike_relative_improvement_mean = _get_mean_metric(
            group_df, primary_metric, "relative_improvement_vs_benchmark"
        )

        # Guardrail discreto principal: normalmente macro F1.
        benchmark_macro_f1_mean = _get_mean_metric(
            group_df, discrete_guardrail_metric, "benchmark_metric_value"
        )
        ml_macro_f1_mean = _get_mean_metric(
            group_df, discrete_guardrail_metric, "ml_metric_value"
        )
        macro_f1_delta = ml_macro_f1_mean - benchmark_macro_f1_mean

        # Guardrail discreto secundario: normalmente balanced accuracy.
        benchmark_balanced_accuracy_mean = _get_mean_metric(
            group_df, secondary_discrete_metric, "benchmark_metric_value"
        )
        ml_balanced_accuracy_mean = _get_mean_metric(
            group_df, secondary_discrete_metric, "ml_metric_value"
        )
        balanced_accuracy_delta = (
            ml_balanced_accuracy_mean - benchmark_balanced_accuracy_mean
        )

        # Regla 1: el ML debe mejorar suficientemente en la métrica principal.
        qlike_pass = bool(
            pd.notna(qlike_relative_improvement_mean)
            and qlike_relative_improvement_mean >= min_relative_qlike_improvement
        )

        # Regla 2: el ML no puede empeorar más de lo tolerado en macro F1.
        macro_f1_pass = bool(
            pd.notna(macro_f1_delta)
            and macro_f1_delta >= -max_macro_f1_drop
        )

        # Regla 3: el ML no puede empeorar más de lo tolerado en balanced accuracy.
        balanced_accuracy_pass = bool(
            pd.notna(balanced_accuracy_delta)
            and balanced_accuracy_delta >= -max_balanced_accuracy_drop
        )

        # La calibración todavía no participa en la decisión cuantitativa,
        # pero se registra explícitamente para dejar trazabilidad del estado.
        calibration_status = (
            "not_evaluable_yet" if not calibration_available else "pending_evaluation"
        )

        # La promoción sólo ocurre si todas las reglas cuantitativas pasan.
        decision = (
            "promote_ml"
            if qlike_pass and macro_f1_pass and balanced_accuracy_pass
            else "do_not_promote_ml"
        )

        summary_rows.append(
            {
                "evaluation_version": evaluation_version,
                "instrument_id": instrument_id,
                "symbol": symbol,
                "benchmark_model_name": benchmark_model_name,
                "ml_model_name": ml_model_name,
                "benchmark_mean_qlike": benchmark_qlike_mean,
                "ml_mean_qlike": ml_qlike_mean,
                "relative_qlike_improvement_mean": qlike_relative_improvement_mean,
                "benchmark_mean_macro_f1": benchmark_macro_f1_mean,
                "ml_mean_macro_f1": ml_macro_f1_mean,
                "macro_f1_delta": macro_f1_delta,
                "benchmark_mean_balanced_accuracy": benchmark_balanced_accuracy_mean,
                "ml_mean_balanced_accuracy": ml_balanced_accuracy_mean,
                "balanced_accuracy_delta": balanced_accuracy_delta,
                "qlike_pass": qlike_pass,
                "macro_f1_pass": macro_f1_pass,
                "balanced_accuracy_pass": balanced_accuracy_pass,
                "calibration_status": calibration_status,
                "decision": decision,
                "comparison_rows": int(len(group_df)),
                "distinct_splits": int(group_df["split_id"].nunique()),
                "distinct_roles": int(group_df["dataset_role"].nunique()),
            }
        )

    return pd.DataFrame(summary_rows)


def _build_decision_reasons_df(
    decision_summary_df: pd.DataFrame,
    min_relative_qlike_improvement: float,
    max_macro_f1_drop: float,
    max_balanced_accuracy_drop: float,
) -> pd.DataFrame:
    """
    Descompone la decisión final en reglas individuales y construye una tabla
    explicativa en formato largo.

    Parameters
    ----------
    decision_summary_df : pd.DataFrame
        Resumen agregado de decisión por símbolo.
    min_relative_qlike_improvement : float
        Umbral esperado para la mejora relativa en la métrica principal.
    max_macro_f1_drop : float
        Caída máxima tolerada en macro F1.
    max_balanced_accuracy_drop : float
        Caída máxima tolerada en balanced accuracy.

    Returns
    -------
    pd.DataFrame
        Tabla en formato largo donde cada fila representa una regla evaluada
        para un símbolo determinado.

    Notes
    -----
    Este artefacto sirve como capa explicativa/auditable del proceso de decisión.
    En vez de dejar sólo un "promote" o "do_not_promote", documenta exactamente
    qué regla pasó y cuál no.
    """
    reason_rows: list[dict[str, object]] = []

    for _, row in decision_summary_df.iterrows():
        base = {
            "evaluation_version": row["evaluation_version"],
            "instrument_id": row["instrument_id"],
            "symbol": row["symbol"],
            "decision": row["decision"],
        }

        reason_rows.extend(
            [
                {
                    **base,
                    "rule_name": "qlike_relative_improvement",
                    "expected_threshold": min_relative_qlike_improvement,
                    "observed_value": row["relative_qlike_improvement_mean"],
                    "passed": bool(row["qlike_pass"]),
                },
                {
                    **base,
                    "rule_name": "macro_f1_no_worse",
                    "expected_threshold": -max_macro_f1_drop,
                    "observed_value": row["macro_f1_delta"],
                    "passed": bool(row["macro_f1_pass"]),
                },
                {
                    **base,
                    "rule_name": "balanced_accuracy_no_worse",
                    "expected_threshold": -max_balanced_accuracy_drop,
                    "observed_value": row["balanced_accuracy_delta"],
                    "passed": bool(row["balanced_accuracy_pass"]),
                },
                {
                    **base,
                    "rule_name": "calibration_status",
                    "expected_threshold": np.nan,
                    "observed_value": row["calibration_status"],
                    "passed": row["calibration_status"] == "pending_evaluation",
                },
            ]
        )

    reasons_df = pd.DataFrame(reason_rows)
    reasons_df["observed_value"] = reasons_df["observed_value"].astype(str)
    return reasons_df


def build_model_decision_artifacts(
    metrics_df: pd.DataFrame,
    benchmark_model_name: str = "garch_11_student_t",
    ml_model_name: str = "xgboost_regressor",
    score_roles: Sequence[str] = ("validation", "test"),
    primary_metric: str = "qlike",
    discrete_guardrail_metric: str = "macro_f1",
    secondary_discrete_metric: str = "balanced_accuracy",
    min_relative_qlike_improvement: float = 0.10,
    max_macro_f1_drop: float = 0.00,
    max_balanced_accuracy_drop: float = 0.00,
    calibration_available: bool = False,
) -> ModelDecisionArtifacts:
    """
    Orquesta la construcción completa de los artefactos de decisión sobre
    promoción del modelo ML.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Tabla de métricas agregadas generada por la etapa de evaluación.
    benchmark_model_name : str, default="garch_11_student_t"
        Nombre del modelo benchmark.
    ml_model_name : str, default="xgboost_regressor"
        Nombre del modelo ML candidato.
    score_roles : Sequence[str], default=("validation", "test")
        Roles del dataset que se usarán para tomar la decisión.
    primary_metric : str, default="qlike"
        Métrica principal de decisión.
    discrete_guardrail_metric : str, default="macro_f1"
        Guardrail discreto principal.
    secondary_discrete_metric : str, default="balanced_accuracy"
        Guardrail discreto secundario.
    min_relative_qlike_improvement : float, default=0.10
        Mejora relativa mínima requerida en la métrica principal.
    max_macro_f1_drop : float, default=0.00
        Máxima caída tolerada en macro F1.
    max_balanced_accuracy_drop : float, default=0.00
        Máxima caída tolerada en balanced accuracy.
    calibration_available : bool, default=False
        Indica si la evaluación de calibración está disponible.

    Returns
    -------
    ModelDecisionArtifacts
        Objeto con:
        - decision_panel_df
        - decision_summary_df
        - decision_reasons_df

    Workflow
    --------
    1. Valida el esquema mínimo de `metrics_df`.
    2. Verifica unicidad lógica de métricas.
    3. Construye el panel intermedio alineando benchmark y ML.
    4. Construye el resumen agregado de decisión por símbolo.
    5. Construye la tabla explicativa de reglas/razones.
    6. Empaqueta todo en un dataclass inmutable.

    Notes
    -----
    Esta es la función principal del módulo. No calcula métricas desde datos
    crudos, sino que toma la salida de evaluación ya agregada y la transforma
    en una decisión de promoción auditable y reproducible.
    """
    _validate_required_columns(metrics_df, REQUIRED_METRICS_COLS, "metrics_df")
    _validate_metric_uniqueness(metrics_df)

    decision_panel_df = _build_decision_panel_df(
        metrics_df=metrics_df,
        benchmark_model_name=benchmark_model_name,
        ml_model_name=ml_model_name,
        score_roles=score_roles,
    )

    decision_summary_df = _build_decision_summary_df(
        decision_panel_df=decision_panel_df,
        benchmark_model_name=benchmark_model_name,
        ml_model_name=ml_model_name,
        primary_metric=primary_metric,
        discrete_guardrail_metric=discrete_guardrail_metric,
        secondary_discrete_metric=secondary_discrete_metric,
        min_relative_qlike_improvement=min_relative_qlike_improvement,
        max_macro_f1_drop=max_macro_f1_drop,
        max_balanced_accuracy_drop=max_balanced_accuracy_drop,
        calibration_available=calibration_available,
    )

    decision_reasons_df = _build_decision_reasons_df(
        decision_summary_df=decision_summary_df,
        min_relative_qlike_improvement=min_relative_qlike_improvement,
        max_macro_f1_drop=max_macro_f1_drop,
        max_balanced_accuracy_drop=max_balanced_accuracy_drop,
    )

    return ModelDecisionArtifacts(
        decision_panel_df=decision_panel_df,
        decision_summary_df=decision_summary_df,
        decision_reasons_df=decision_reasons_df,
    )