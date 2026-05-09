from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.evaluation import build_model_decision_artifacts


def make_metrics_df_promote_case() -> pd.DataFrame:
    """
    Construye un DataFrame sintético de métricas que representa un caso
    en el que el modelo ML sí debería ser promovido frente al benchmark.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas agregadas con una única combinación de:
        - evaluation_version
        - instrument_id
        - symbol
        - split_id
        - dataset_role

        y dos modelos:
        - benchmark: `garch_11_student_t`
        - ML: `xgboost_regressor`

    Notes
    -----
    Este fixture fue diseñado para que el modelo ML cumpla las reglas
    principales de promoción:

    1. Mejora suficiente en QLIKE:
       benchmark = -2.00
       ml        = -2.30

       Dado que para QLIKE "menor es mejor", pasar de -2.00 a -2.30
       representa una mejora relativa del 15%.

    2. No empeora en macro F1:
       benchmark = 0.30
       ml        = 0.31

    3. No empeora en balanced accuracy:
       benchmark = 0.35
       ml        = 0.36

    También se incluyen RMSE y MAE para que la tabla tenga la forma
    completa esperada por el pipeline, aunque la decisión de promoción
    se apoya principalmente en QLIKE y los guardrails discretos.
    """
    rows = []
    base = {
        "evaluation_version": "v1",
        "instrument_id": "spy_us",
        "symbol": "SPY",
        "split_id": "split_001",
        "dataset_role": "validation",
    }

    metrics = [
        ("qlike", "continuous", -2.00, -2.30, 100),              # ML improves by 15%
        ("rmse", "continuous", 0.10, 0.09, 100),
        ("mae", "continuous", 0.08, 0.07, 100),
        ("macro_f1", "discrete", 0.30, 0.31, 100),               # no worse
        ("balanced_accuracy", "discrete", 0.35, 0.36, 100),      # no worse
    ]

    # Por cada métrica se generan dos filas:
    # una para el benchmark y otra para el modelo ML.
    for metric_name, target_family, benchmark_value, ml_value, n_obs in metrics:
        rows.append(
            {
                **base,
                "model_name": "garch_11_student_t",
                "target_family": target_family,
                "metric_name": metric_name,
                "metric_value": benchmark_value,
                "n_obs": n_obs,
            }
        )
        rows.append(
            {
                **base,
                "model_name": "xgboost_regressor",
                "target_family": target_family,
                "metric_name": metric_name,
                "metric_value": ml_value,
                "n_obs": n_obs,
            }
        )

    return pd.DataFrame(rows)


def make_metrics_df_do_not_promote_case() -> pd.DataFrame:
    """
    Construye un DataFrame sintético de métricas que representa un caso
    en el que el modelo ML NO debería ser promovido.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas agregadas para un único símbolo (`TLT`) y un único
        split/rol, con benchmark y modelo ML.

    Notes
    -----
    Este fixture fue diseñado para que falle la regla principal de promoción:

    1. QLIKE empeora:
       benchmark = -2.00
       ml        = -1.80

       Como en QLIKE menor es mejor, el modelo ML es peor que el benchmark.

    2. Las métricas discretas no empeoran:
       - macro_f1 permanece igual
       - balanced_accuracy permanece igual

    Aun así, la decisión final debería ser `do_not_promote_ml`, porque
    no basta con no empeorar en los guardrails: también se exige mejora
    suficiente en la métrica principal.
    """
    rows = []
    base = {
        "evaluation_version": "v1",
        "instrument_id": "tlt_us",
        "symbol": "TLT",
        "split_id": "split_001",
        "dataset_role": "validation",
    }

    metrics = [
        ("qlike", "continuous", -2.00, -1.80, 100),              # ML worsens
        ("rmse", "continuous", 0.10, 0.11, 100),
        ("mae", "continuous", 0.08, 0.09, 100),
        ("macro_f1", "discrete", 0.30, 0.30, 100),
        ("balanced_accuracy", "discrete", 0.35, 0.35, 100),
    ]

    # Igual que en el fixture anterior, se crean dos filas por métrica:
    # benchmark y ML.
    for metric_name, target_family, benchmark_value, ml_value, n_obs in metrics:
        rows.append(
            {
                **base,
                "model_name": "garch_11_student_t",
                "target_family": target_family,
                "metric_name": metric_name,
                "metric_value": benchmark_value,
                "n_obs": n_obs,
            }
        )
        rows.append(
            {
                **base,
                "model_name": "xgboost_regressor",
                "target_family": target_family,
                "metric_name": metric_name,
                "metric_value": ml_value,
                "n_obs": n_obs,
            }
        )

    return pd.DataFrame(rows)


def test_build_model_decision_artifacts_creates_expected_outputs() -> None:
    """
    Verifica el caso base exitoso de construcción de artefactos de decisión.

    What this test checks
    ---------------------
    1. Que la función principal no falle con una tabla de métricas válida.
    2. Que el panel de decisión tenga 5 filas:
       una por cada métrica.
    3. Que el summary tenga 1 fila:
       una por símbolo.
    4. Que la tabla de razones tenga 4 filas:
       una por regla evaluada.
    5. Que el conjunto de métricas y reglas sea el esperado.
    6. Que la decisión final sea `promote_ml` en este caso sintético.

    Notes
    -----
    Este test valida la estructura general de la salida del módulo,
    no solamente una regla puntual. Es el equivalente al “smoke test”
    del pipeline de decisión.
    """
    metrics_df = make_metrics_df_promote_case()

    artifacts = build_model_decision_artifacts(
        metrics_df=metrics_df,
        benchmark_model_name="garch_11_student_t",
        ml_model_name="xgboost_regressor",
        score_roles=("validation", "test"),
        primary_metric="qlike",
        discrete_guardrail_metric="macro_f1",
        secondary_discrete_metric="balanced_accuracy",
        min_relative_qlike_improvement=0.10,
        max_macro_f1_drop=0.00,
        max_balanced_accuracy_drop=0.00,
        calibration_available=False,
    )

    panel_df = artifacts.decision_panel_df
    summary_df = artifacts.decision_summary_df
    reasons_df = artifacts.decision_reasons_df

    # Hay una fila por métrica en el panel pivotado.
    assert len(panel_df) == 5

    # Hay una fila resumen por símbolo.
    assert len(summary_df) == 1

    # Hay una fila por regla en la tabla explicativa de razones.
    assert len(reasons_df) == 4

    assert set(panel_df["metric_name"].unique()) == {
        "qlike",
        "rmse",
        "mae",
        "macro_f1",
        "balanced_accuracy",
    }
    assert set(summary_df["decision"].unique()) == {"promote_ml"}
    assert set(reasons_df["rule_name"].unique()) == {
        "qlike_relative_improvement",
        "macro_f1_no_worse",
        "balanced_accuracy_no_worse",
        "calibration_status",
    }


def test_build_model_decision_artifacts_promotes_when_rule_is_satisfied() -> None:
    """
    Verifica que el modelo ML sea promovido cuando todas las reglas cuantitativas
    de decisión se cumplen.

    Test setup
    ----------
    Se usa el fixture `make_metrics_df_promote_case()`, construido para que:
    - QLIKE supere el umbral mínimo de mejora relativa.
    - macro F1 no empeore.
    - balanced accuracy no empeore.

    Expected behavior
    -----------------
    - `qlike_pass` debe ser True
    - `macro_f1_pass` debe ser True
    - `balanced_accuracy_pass` debe ser True
    - la decisión final debe ser `promote_ml`

    Notes
    -----
    Este test se enfoca en la lógica positiva del módulo: demostrar que el
    sistema sí promueve cuando el candidato cumple las condiciones definidas.
    """
    metrics_df = make_metrics_df_promote_case()

    artifacts = build_model_decision_artifacts(
        metrics_df=metrics_df,
        min_relative_qlike_improvement=0.10,
        max_macro_f1_drop=0.00,
        max_balanced_accuracy_drop=0.00,
        calibration_available=False,
    )

    summary_row = artifacts.decision_summary_df.iloc[0]

    # Se usa esta forma algo redundante porque a veces pandas devuelve
    # booleanos numpy en lugar de bool nativo de Python.
    assert summary_row["qlike_pass"] is True or bool(summary_row["qlike_pass"]) is True
    assert summary_row["macro_f1_pass"] is True or bool(summary_row["macro_f1_pass"]) is True
    assert summary_row["balanced_accuracy_pass"] is True or bool(summary_row["balanced_accuracy_pass"]) is True
    assert summary_row["decision"] == "promote_ml"


def test_build_model_decision_artifacts_does_not_promote_when_qlike_fails() -> None:
    """
    Verifica que el modelo ML no sea promovido cuando falla la regla principal
    de QLIKE, aunque los guardrails discretos no empeoren.

    Test setup
    ----------
    Se usa el fixture `make_metrics_df_do_not_promote_case()`, donde:
    - QLIKE empeora respecto al benchmark.
    - macro F1 permanece igual.
    - balanced accuracy permanece igual.

    Expected behavior
    -----------------
    - `qlike_pass` debe ser False
    - la decisión final debe ser `do_not_promote_ml`

    Why this matters
    ----------------
    Este test deja claro que la promoción no depende sólo de no empeorar
    en métricas discretas: la mejora en la métrica principal sigue siendo
    una condición necesaria.
    """
    metrics_df = make_metrics_df_do_not_promote_case()

    artifacts = build_model_decision_artifacts(
        metrics_df=metrics_df,
        min_relative_qlike_improvement=0.10,
        max_macro_f1_drop=0.00,
        max_balanced_accuracy_drop=0.00,
        calibration_available=False,
    )

    summary_row = artifacts.decision_summary_df.iloc[0]

    assert summary_row["qlike_pass"] is False or bool(summary_row["qlike_pass"]) is False
    assert summary_row["decision"] == "do_not_promote_ml"


def test_build_model_decision_artifacts_raises_on_duplicate_metric_rows() -> None:
    """
    Verifica que la función falle cuando la tabla de métricas contiene
    filas duplicadas para una misma clave lógica.

    Test setup
    ----------
    Se toma el fixture válido del caso de promoción y se le agrega
    una copia adicional de la primera fila.

    Expected behavior
    -----------------
    La función debe lanzar `ValueError` con un mensaje relacionado con
    filas duplicadas de métricas.

    Why this matters
    ----------------
    El módulo de decisión pivotea la tabla de métricas para alinear benchmark
    y ML. Si existen duplicados, esa alineación deja de ser inequívoca.
    """
    metrics_df = make_metrics_df_promote_case()
    metrics_df = pd.concat([metrics_df, metrics_df.iloc[[0]]], axis=0, ignore_index=True)

    with pytest.raises(ValueError, match="duplicated metric rows"):
        build_model_decision_artifacts(metrics_df=metrics_df)


def test_build_model_decision_artifacts_raises_when_required_columns_are_missing() -> None:
    """
    Verifica que la función falle cuando falta una columna obligatoria
    en la tabla de métricas de entrada.

    Test setup
    ----------
    Se elimina la columna `n_obs`, que forma parte del esquema requerido.

    Expected behavior
    -----------------
    La función debe lanzar `ValueError` con un mensaje que indique que
    faltan columnas requeridas.

    Why this matters
    ----------------
    Esta prueba asegura que el módulo falle temprano y con mensajes claros
    cuando el contrato de entrada no se cumple.
    """
    metrics_df = make_metrics_df_promote_case().drop(columns=["n_obs"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_model_decision_artifacts(metrics_df=metrics_df)