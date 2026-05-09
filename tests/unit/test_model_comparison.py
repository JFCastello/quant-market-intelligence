from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.evaluation import build_model_comparison_artifacts


def make_regime_targets_df() -> pd.DataFrame:
    """
    Construye un DataFrame de ejemplo con los targets reales de volatilidad
    y de régimen, junto con los umbrales usados para clasificar regímenes.

    Returns
    -------
    pd.DataFrame
        DataFrame sintético con 4 observaciones OOS para un único instrumento.

    Notes
    -----
    Este fixture manual representa la "verdad terreno" mínima necesaria para
    evaluar el pipeline de comparación de modelos. Incluye:

    - `instrument_id`: identificador interno del activo.
    - `date`: fechas de evaluación.
    - `split_id`: identificador del split temporal.
    - `dataset_role`: rol del subconjunto, aquí `validation`.
    - `future_rv_5d`: target continuo de volatilidad futura a 5 días.
    - `future_regime_5d`: target discreto de régimen.
    - `threshold_low` y `threshold_high`: umbrales para mapear la volatilidad
      a clases de régimen.
    - `regime_thresholds_source`: metadato que indica de dónde provienen
      los umbrales.

    Este DataFrame se usa como base común para varios tests.
    """
    return pd.DataFrame(
        {
            "instrument_id": ["spy_us"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "split_id": ["split_001"] * 4,
            "dataset_role": ["validation"] * 4,
            "future_rv_5d": [0.05, 0.10, 0.20, 0.30],
            "future_regime_5d": ["calm", "normal", "stress", "stress"],
            "threshold_low": [0.08] * 4,
            "threshold_high": [0.18] * 4,
            "regime_thresholds_source": ["train_only"] * 4,
        }
    )


def make_ml_forecasts_df() -> pd.DataFrame:
    """
    Construye un DataFrame de ejemplo con las predicciones del modelo ML.

    Returns
    -------
    pd.DataFrame
        DataFrame con 4 observaciones de forecast continuo para un único símbolo.

    Notes
    -----
    Este DataFrame simula la salida de un modelo como `xgboost_regressor`.
    Importante: aquí sólo existe el forecast continuo `yhat_future_rv_5d`.

    La clase de régimen predicha (`yhat_future_regime_5d`) NO viene en esta
    tabla, sino que debe ser inferida posteriormente en el pipeline usando
    los umbrales provenientes de `regime_targets_df`.

    Esto permite probar específicamente que la función principal:
    `build_model_comparison_artifacts(...)`
    esté derivando correctamente los regímenes discretos del modelo ML.
    """
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "split_id": ["split_001"] * 4,
            "dataset_role": ["validation"] * 4,
            "model_name": ["xgboost_regressor"] * 4,
            "future_rv_5d": [0.05, 0.10, 0.20, 0.30],
            "yhat_future_rv_5d": [0.07, 0.12, 0.19, 0.17],
        }
    )


def make_benchmark_regimes_df() -> pd.DataFrame:
    """
    Construye un DataFrame de ejemplo con la salida del modelo benchmark,
    incluyendo tanto predicción continua como predicción discreta de régimen.

    Returns
    -------
    pd.DataFrame
        DataFrame con 4 observaciones del benchmark.

    Notes
    -----
    A diferencia del modelo ML, el benchmark ya trae explícitamente:
    - `yhat_future_rv_5d`
    - `yhat_future_regime_5d`

    Por eso este fixture representa un caso donde el benchmark llega más
    "completo" al pipeline, mientras que el modelo ML necesita que su
    régimen sea derivado usando umbrales.

    Este contraste es importante porque el módulo de evaluación debe ser
    capaz de unificar ambas fuentes bajo un mismo esquema final.
    """
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "split_id": ["split_001"] * 4,
            "dataset_role": ["validation"] * 4,
            "model_name": ["garch_11_student_t"] * 4,
            "future_rv_5d": [0.05, 0.10, 0.20, 0.30],
            "future_regime_5d": ["calm", "normal", "stress", "stress"],
            "yhat_future_rv_5d": [0.06, 0.11, 0.16, 0.25],
            "yhat_future_regime_5d": ["calm", "normal", "normal", "stress"],
            "threshold_low": [0.08] * 4,
            "threshold_high": [0.18] * 4,
            "regime_thresholds_source": ["train_only"] * 4,
        }
    )


def test_build_model_comparison_artifacts_creates_expected_outputs() -> None:
    """
    Verifica el caso base exitoso: la función principal construye correctamente
    los tres artefactos esperados y les da una forma coherente.

    What this test checks
    ---------------------
    1. Que la función no falle con entradas válidas.
    2. Que el panel combinado tenga 8 filas:
       - 4 del benchmark
       - 4 del modelo ML
    3. Que la tabla de métricas tenga 10 filas:
       - 5 métricas por modelo
       - 2 modelos
    4. Que la tabla de confusión tenga 18 filas:
       - 9 celdas por matriz 3x3
       - 2 modelos
    5. Que los nombres de modelos, métricas y etiquetas sean los esperados.
    6. Que el panel final contenga columnas derivadas de error y régimen predicho.

    Notes
    -----
    Este es el test más general del módulo: no valida detalles finos de una
    sola transformación, sino que asegura que el pipeline completo produce
    salidas estructuralmente correctas.
    """
    benchmark_df = make_benchmark_regimes_df()
    ml_df = make_ml_forecasts_df()
    regime_targets_df = make_regime_targets_df()

    artifacts = build_model_comparison_artifacts(
        benchmark_regimes_df=benchmark_df,
        ml_forecasts_df=ml_df,
        regime_targets_df=regime_targets_df,
        evaluation_version="v1",
        labels=("calm", "normal", "stress"),
    )

    panel_df = artifacts.evaluation_panel_df
    metrics_df = artifacts.metrics_df
    confusion_df = artifacts.confusion_df

    # 4 filas benchmark + 4 filas ML = 8
    assert len(panel_df) == 8

    # 5 métricas por modelo x 2 modelos = 10
    assert len(metrics_df) == 10

    # 3x3 celdas de confusión por modelo x 2 modelos = 18
    assert len(confusion_df) == 18

    assert set(panel_df["model_name"].unique()) == {
        "garch_11_student_t",
        "xgboost_regressor",
    }
    assert set(metrics_df["metric_name"].unique()) == {
        "qlike",
        "rmse",
        "mae",
        "macro_f1",
        "balanced_accuracy",
    }
    assert set(confusion_df["y_true"].unique()) == {"calm", "normal", "stress"}
    assert set(confusion_df["y_pred"].unique()) == {"calm", "normal", "stress"}

    # Estas columnas prueban que el pipeline sí calculó artefactos derivados,
    # no sólo que concatenó tablas de entrada.
    required_panel_cols = {
        "error",
        "abs_error",
        "sq_error",
        "qlike_term",
        "yhat_future_regime_5d",
    }
    assert required_panel_cols.issubset(panel_df.columns)


def test_build_model_comparison_artifacts_maps_ml_regimes_from_thresholds() -> None:
    """
    Verifica que el pipeline derive correctamente las etiquetas de régimen
    del modelo ML usando los umbrales `threshold_low` y `threshold_high`.

    Logic being tested
    ------------------
    Dado:
    - threshold_low = 0.08
    - threshold_high = 0.18

    Entonces:
    - valor < 0.08                -> "calm"
    - 0.08 <= valor <= 0.18       -> "normal"
    - valor > 0.18                -> "stress"

    For the synthetic ML predictions
    --------------------------------
    - 0.07 -> calm
    - 0.12 -> normal
    - 0.19 -> stress
    - 0.17 -> normal

    Expected result
    ---------------
    ["calm", "normal", "stress", "normal"]

    Notes
    -----
    Este test es muy importante porque valida una de las transformaciones
    más delicadas del pipeline: convertir un forecast continuo del modelo ML
    en una clasificación discreta de regímenes.
    """
    benchmark_df = make_benchmark_regimes_df()
    ml_df = make_ml_forecasts_df()
    regime_targets_df = make_regime_targets_df()

    artifacts = build_model_comparison_artifacts(
        benchmark_regimes_df=benchmark_df,
        ml_forecasts_df=ml_df,
        regime_targets_df=regime_targets_df,
        evaluation_version="v1",
        labels=("calm", "normal", "stress"),
    )

    # Nos quedamos únicamente con las filas del modelo ML para validar
    # el mapeo de sus regímenes predichos.
    ml_panel_df = artifacts.evaluation_panel_df.loc[
        artifacts.evaluation_panel_df["model_name"] == "xgboost_regressor"
    ].sort_values("date")

    assert ml_panel_df["yhat_future_regime_5d"].tolist() == [
        "calm",
        "normal",
        "stress",
        "normal",
    ]


def test_build_model_comparison_artifacts_raises_when_oos_keys_do_not_align() -> None:
    """
    Verifica que la función falle cuando benchmark y ML no están alineados
    exactamente en sus llaves OOS.

    Test setup
    ----------
    Se toma el DataFrame ML válido y se elimina la última fila con `iloc[:-1]`.
    Eso hace que benchmark y ML ya no tengan el mismo conjunto de observaciones
    out-of-sample.

    Expected behavior
    -----------------
    La función debe lanzar `ValueError` con un mensaje que mencione que las
    llaves no se alinean exactamente.

    Why this matters
    ----------------
    Comparar modelos sobre conjuntos distintos de observaciones haría que
    las métricas fueran injustas o incluso engañosas. Por eso esta validación
    debe ser estricta.
    """
    benchmark_df = make_benchmark_regimes_df()
    ml_df = make_ml_forecasts_df().iloc[:-1].copy()
    regime_targets_df = make_regime_targets_df()

    with pytest.raises(ValueError, match="do not align exactly"):
        build_model_comparison_artifacts(
            benchmark_regimes_df=benchmark_df,
            ml_forecasts_df=ml_df,
            regime_targets_df=regime_targets_df,
            evaluation_version="v1",
            labels=("calm", "normal", "stress"),
        )


def test_build_model_comparison_artifacts_raises_when_required_columns_are_missing() -> None:
    """
    Verifica que la función falle cuando alguna entrada no cumple el esquema
    mínimo requerido.

    Test setup
    ----------
    Se elimina la columna `threshold_high` de `regime_targets_df`.

    Expected behavior
    -----------------
    La función debe lanzar `ValueError` indicando que faltan columnas requeridas.

    Why this matters
    ----------------
    El pipeline depende de estas columnas para:
    - validar estructura,
    - hacer joins,
    - derivar regímenes,
    - construir artefactos consistentes.

    Si falta una columna clave, el fallo debe ocurrir temprano y con un mensaje
    claro, en lugar de producir errores más ambiguos después.
    """
    benchmark_df = make_benchmark_regimes_df()
    ml_df = make_ml_forecasts_df()
    regime_targets_df = make_regime_targets_df().drop(columns=["threshold_high"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_model_comparison_artifacts(
            benchmark_regimes_df=benchmark_df,
            ml_forecasts_df=ml_df,
            regime_targets_df=regime_targets_df,
            evaluation_version="v1",
            labels=("calm", "normal", "stress"),
        )