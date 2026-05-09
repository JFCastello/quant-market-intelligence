from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.evaluation import build_structural_break_artifacts


def make_features_targets_with_regime_change(
    n_rows: int = 120,
    instrument_id: str = "spy_us",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye un par sintético de DataFrames (features y targets) con un
    cambio de régimen claramente marcado a mitad de la muestra.

    Propósito:
    Esta función sirve como generador de datos de prueba controlados para los
    tests de structural breaks. En vez de depender de datos reales del proyecto,
    crea una serie artificial donde el comportamiento cambia de forma abrupta
    en la mitad del periodo, lo que facilita verificar si el detector de
    quiebres es capaz de capturar ese cambio.

    Parámetros:
    - n_rows:
      número total de filas que tendrán los DataFrames generados.
    - instrument_id:
      identificador del instrumento que se repetirá en todas las filas.

    Retorna:
    - Una tupla:
        (features_df, targets_df)
      donde:
      * features_df contiene la señal de retornos
      * targets_df contiene la señal de volatilidad futura

    Idea del cambio de régimen:
    - Primera mitad:
        retornos pequeños y volatilidad futura baja
    - Segunda mitad:
        retornos mucho mayores y volatilidad futura más alta

    Eso genera una ruptura muy artificial, pero justamente útil para un test,
    porque el quiebre debería ser relativamente evidente.
    """

    # Genera un índice de fechas hábiles (business days) a partir de 2020-01-01.
    # Esto imita una serie financiera más realista que usar días calendario.
    dates = pd.bdate_range("2020-01-01", periods=n_rows)

    # Divide la muestra en dos mitades.
    # Si n_rows es impar, la segunda mitad recibe la fila sobrante.
    first_half = n_rows // 2
    second_half = n_rows - first_half

    # Construye la señal de retornos:
    # - primera mitad: retorno constante bajo
    # - segunda mitad: retorno constante mucho mayor
    #
    # El uso de np.full hace explícito que cada mitad tiene un valor plano.
    log_ret = np.concatenate(
        [
            np.full(first_half, 0.001),
            np.full(second_half, 0.020),
        ]
    )

    # Construye la señal objetivo de volatilidad futura:
    # - primera mitad: baja volatilidad
    # - segunda mitad: alta volatilidad
    #
    # Esto refuerza el cambio de régimen no solo en retornos, sino también
    # en la variable objetivo.
    future_rv = np.concatenate(
        [
            np.full(first_half, 0.05),
            np.full(second_half, 0.25),
        ]
    )

    # Crea el DataFrame de features.
    #
    # Columnas:
    # - instrument_id: identifica el activo
    # - date: fecha de la observación
    # - feature_version: metadato de versión
    # - log_ret_1d: señal de retornos usada como input
    features_df = pd.DataFrame(
        {
            "instrument_id": [instrument_id] * n_rows,
            "date": dates,
            "feature_version": ["v1"] * n_rows,
            "log_ret_1d": log_ret,
        }
    )

    # Crea el DataFrame de targets.
    #
    # Columnas:
    # - instrument_id: identifica el activo
    # - date: fecha de la observación
    # - target_version: metadato de versión
    # - future_rv_5d: señal objetivo de volatilidad futura
    targets_df = pd.DataFrame(
        {
            "instrument_id": [instrument_id] * n_rows,
            "date": dates,
            "target_version": ["v1"] * n_rows,
            "future_rv_5d": future_rv,
        }
    )

    return features_df, targets_df


def test_build_structural_break_artifacts_returns_expected_outputs() -> None:
    """
    Verifica el caso feliz principal del constructor de artefactos.

    Qué valida:
    1. Que la función retorne un signal_df con el número esperado de filas.
    2. Que el signal_df contenga tanto las señales originales como las
       columnas estandarizadas z_..., si así se configuró.
    3. Que el events_df tenga columnas clave esperadas.
    4. Que se detecte al menos un evento.
    5. Que los metadatos del método usado (algorithm, cost_model) queden
       correctamente registrados en los eventos.

    Intención del test:
    No intenta comprobar cada detalle fino del algoritmo, sino verificar que
    la salida general tenga la forma y metadatos esperados para un caso donde
    sí debería detectarse un cambio de régimen.
    """

    # Genera un dataset sintético con un cambio de régimen claro.
    features_df, targets_df = make_features_targets_with_regime_change()

    # Ejecuta la construcción completa de artefactos de structural breaks.
    #
    # Configuración relevante:
    # - use_joint_signal=True:
    #   la detección se hace considerando ambas señales conjuntamente.
    # - standardize_joint_signal=True:
    #   se espera que existan columnas z_... en la señal resultante.
    # - algorithm="pelt", cost_model="rbf":
    #   parámetros del método de detección.
    artifacts = build_structural_break_artifacts(
        features_df=features_df,
        targets_df=targets_df,
        break_version="v1",
        return_signal_column="log_ret_1d",
        volatility_signal_column="future_rv_5d",
        use_joint_signal=True,
        joint_signal_columns=("log_ret_1d", "future_rv_5d"),
        algorithm="pelt",
        cost_model="rbf",
        dropna=True,
        min_required_rows=60,
        standardize_joint_signal=True,
        penalty=1.0,
        min_size=10,
        jump=1,
    )

    # Extrae los dos outputs principales del objeto retornado.
    signal_df = artifacts.signal_df
    events_df = artifacts.events_df

    # Verifica que la señal final conserve todas las filas esperadas.
    assert len(signal_df) == 120

    # Verifica que el DataFrame de señal contenga:
    # - las columnas originales de señal
    # - las columnas estandarizadas asociadas
    assert {"log_ret_1d", "future_rv_5d", "z_log_ret_1d", "z_future_rv_5d"}.issubset(signal_df.columns)

    # Verifica presencia de columnas clave en la tabla de eventos.
    assert "instrument_id" in events_df.columns
    assert "break_date" in events_df.columns
    assert "method" in events_df.columns

    # Como el dataset tiene un cambio marcado, se espera al menos un evento.
    assert len(events_df) >= 1

    # El archivo de eventos debe corresponder a un solo instrumento.
    assert events_df["instrument_id"].nunique() == 1

    # También debe corresponder a un solo símbolo.
    assert events_df["symbol"].nunique() == 1

    # Verifica que el algoritmo y cost_model guardados en los eventos
    # sean exactamente los configurados al llamar la función.
    assert events_df["algorithm"].unique().tolist() == ["pelt"]
    assert events_df["cost_model"].unique().tolist() == ["rbf"]


def test_build_structural_break_artifacts_detects_break_dates_in_order() -> None:
    """
    Verifica consistencia temporal de los eventos detectados.

    Qué valida:
    1. Que las break_date estén ordenadas de forma ascendente.
    2. Que cada break_date coincida con next_segment_start_date.

    Justificación:
    En este pipeline, la fecha del quiebre se interpreta como el comienzo del
    nuevo segmento o régimen. Por tanto, ambas columnas deberían coincidir.
    Además, los eventos deben estar ordenados cronológicamente.
    """

    # Genera nuevamente un dataset sintético con cambio de régimen.
    features_df, targets_df = make_features_targets_with_regime_change()

    # Construye los artefactos usando la misma configuración del caso feliz.
    artifacts = build_structural_break_artifacts(
        features_df=features_df,
        targets_df=targets_df,
        break_version="v1",
        return_signal_column="log_ret_1d",
        volatility_signal_column="future_rv_5d",
        use_joint_signal=True,
        joint_signal_columns=("log_ret_1d", "future_rv_5d"),
        algorithm="pelt",
        cost_model="rbf",
        dropna=True,
        min_required_rows=60,
        standardize_joint_signal=True,
        penalty=1.0,
        min_size=10,
        jump=1,
    )

    events_df = artifacts.events_df

    # Convierte explícitamente la columna break_date a datetime y exige que
    # no haya errores de parseo.
    break_dates = pd.to_datetime(events_df["break_date"], errors="raise")

    # Verifica que los eventos estén ordenados cronológicamente.
    assert break_dates.is_monotonic_increasing

    # Verifica la convención del pipeline:
    # la fecha del quiebre coincide con el inicio del siguiente segmento.
    assert (pd.to_datetime(events_df["break_date"]) == pd.to_datetime(events_df["next_segment_start_date"])).all()


def test_build_structural_break_artifacts_raises_when_required_signal_column_is_missing() -> None:
    """
    Verifica que la función falle cuando falta una columna de señal obligatoria.

    Caso probado:
    - Se elimina "log_ret_1d" del DataFrame de features.
    - Luego se llama la función indicando que esa columna debería existir.

    Resultado esperado:
    - La función debe lanzar ValueError con un mensaje que indique
      "missing required columns".

    Importancia:
    Este test comprueba que la validación de inputs está bien implementada y
    que el error es explícito cuando faltan columnas esenciales.
    """

    # Genera datos sintéticos válidos.
    features_df, targets_df = make_features_targets_with_regime_change()

    # Elimina deliberadamente una columna requerida para forzar el error.
    features_df = features_df.drop(columns=["log_ret_1d"])

    # Verifica que la función lance la excepción correcta con un mensaje
    # coherente con el problema.
    with pytest.raises(ValueError, match="missing required columns"):
        build_structural_break_artifacts(
            features_df=features_df,
            targets_df=targets_df,
            return_signal_column="log_ret_1d",
            volatility_signal_column="future_rv_5d",
        )


def test_build_structural_break_artifacts_raises_when_rows_are_insufficient() -> None:
    """
    Verifica que la función falle cuando la cantidad de filas es menor que
    el mínimo requerido.

    Caso probado:
    - Se generan solo 40 filas.
    - Luego se exige min_required_rows=60.

    Resultado esperado:
    - La función debe lanzar ValueError con un mensaje asociado a
      "insufficient rows".

    Importancia:
    Un detector de quiebres necesita una longitud mínima de serie para que
    la segmentación tenga sentido. Este test confirma que ese contrato se
    haga cumplir.
    """

    # Genera una muestra deliberadamente demasiado corta.
    features_df, targets_df = make_features_targets_with_regime_change(n_rows=40)

    # Verifica que la función rechace el input por insuficiencia de filas.
    with pytest.raises(ValueError, match="insufficient rows"):
        build_structural_break_artifacts(
            features_df=features_df,
            targets_df=targets_df,
            return_signal_column="log_ret_1d",
            volatility_signal_column="future_rv_5d",
            min_required_rows=60,
        )


def test_build_structural_break_artifacts_raises_when_instrument_ids_do_not_match() -> None:
    """
    Verifica que la función falle cuando features y targets corresponden a
    instrumentos distintos.

    Caso probado:
    - features_df mantiene instrument_id="spy_us"
    - targets_df se modifica a instrument_id="tlt_us"

    Resultado esperado:
    - La función debe lanzar ValueError con un mensaje asociado a
      "instrument_id mismatch".

    Importancia:
    No tendría sentido construir una señal conjunta o alinear features y
    targets si ambos DataFrames representan instrumentos distintos.
    Este test garantiza que la función detecte esa inconsistencia.
    """

    # Genera datos sintéticos inicialmente consistentes.
    features_df, targets_df = make_features_targets_with_regime_change()

    # Introduce deliberadamente una inconsistencia entre features y targets.
    targets_df["instrument_id"] = "tlt_us"

    # Verifica que la función rechace el caso por desalineación de instrumento.
    with pytest.raises(ValueError, match="instrument_id mismatch"):
        build_structural_break_artifacts(
            features_df=features_df,
            targets_df=targets_df,
            return_signal_column="log_ret_1d",
            volatility_signal_column="future_rv_5d",
        )