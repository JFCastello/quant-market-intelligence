# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
import math

import numpy as np
import pandas as pd
import pytest  # Framework de pruebas unitarias

# Importamos las funciones del módulo de targets que vamos a probar
from quant_platform.targets import (
    adapt_targets_to_contract,
    build_continuous_targets,
    get_enabled_target_columns,
    validate_target_output,
)


# Configuración de prueba que simula los ajustes reales de la plataforma
# Se usa en todos los tests para asegurar consistencia y reproducibilidad
TEST_SETTINGS = {
    "targets": {
        "target_version": "v1",                # Versión del target
        "continuous_target": {
            "name": "future_rv_5d",             # Nombre de la columna del target continuo
            "enabled": True,                    # Target continuo activado
            "horizon_days": 5,                  # Ventana de 5 días hacia adelante
            "base_return_column": "log_ret_1d", # Columna con retornos logarítmicos diarios
            "annualization_factor": 252,        # Factor para anualizar volatilidad (días hábiles)
            "min_periods": 5,                   # Se requieren exactamente 5 días (allow_partial_window=False)
            "window_type": "fixed_forward",     # Ventana fija hacia adelante
            "allow_partial_window": False,      # No se permiten ventanas incompletas
        },
        "classification_target": {
            "name": "future_regime_5d",         # Nombre del target de clasificación (no usado en tests)
            "enabled": False,                   # Desactivado para estos tests
            "source_continuous_target": "future_rv_5d",
            "method": "quantile_bins",
            "labels": ["calm", "normal", "stress"],
            "n_classes": 3,
            "quantiles": [0.33, 0.66],
            "thresholds_source": "train_only",
        },
        "merge_keys": ["instrument_id", "date"],
        "source_layer": "features_context_v1",
    }
}


def make_target_source_df(n_rows: int = 20) -> pd.DataFrame:
    """
    Función auxiliar (fixture manual) para generar un DataFrame de entrada sintético
    que simula los datos de características de contexto necesarios para construir targets.

    Genera:
        - Fechas hábiles consecutivas (frecuencia 'B').
        - Una serie de retornos logarítmicos predefinida y determinista.
        - Valores constantes para 'instrument_id' y 'feature_version'.

    Args:
        n_rows: Número de filas a generar (por defecto 20).

    Returns:
        DataFrame con columnas: instrument_id, date, feature_version, log_ret_1d.
    """
    # Rango de fechas hábiles a partir del 1 de enero de 2020
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")

    # Retornos logarítmicos diarios sintéticos (valores fijos para reproducibilidad)
    log_ret_1d = pd.Series(
        [
            0.0100,
            -0.0050,
            0.0070,
            0.0120,
            -0.0080,
            0.0040,
            0.0090,
            -0.0030,
            0.0110,
            -0.0060,
            0.0050,
            0.0020,
            -0.0040,
            0.0080,
            0.0030,
            -0.0020,
            0.0060,
            -0.0010,
            0.0040,
            0.0070,
        ],
        dtype=float,
    )

    # Construcción del DataFrame
    return pd.DataFrame(
        {
            "instrument_id": "spy_us",          # Un solo instrumento para simplificar
            "date": dates,
            "feature_version": "v1",            # Versión fija
            "log_ret_1d": log_ret_1d,
        }
    )


def test_get_enabled_target_columns_matches_expected_contract() -> None:
    """
    Test 1: Verifica que la función `get_enabled_target_columns` devuelva
    exactamente las columnas esperadas según la configuración de prueba.

    En esta configuración solo está habilitado el target continuo, por lo que
    las columnas deben ser: instrument_id, date, target_version y future_rv_5d.
    """
    # Obtenemos las columnas habilitadas usando la configuración de prueba
    columns = get_enabled_target_columns(TEST_SETTINGS)

    # Lista esperada (target de clasificación desactivado)
    expected = [
        "instrument_id",
        "date",
        "target_version",
        "future_rv_5d",
    ]

    # Verificamos que coincidan exactamente
    assert columns == expected


def test_build_continuous_targets_adds_future_rv_5d_and_preserves_rows() -> None:
    """
    Test 2: Comprueba que la construcción del target continuo:
        - Añade la columna 'future_rv_5d' al DataFrame.
        - Mantiene el mismo número de filas que la entrada.
        - Asigna correctamente la versión del target ('v1') a todas las filas.
    """
    # DataFrame de entrada sintético
    df = make_target_source_df()

    # Construimos el target continuo
    target_df = build_continuous_targets(
        df=df,
        settings=TEST_SETTINGS,
    )

    # Verificamos que la nueva columna existe
    assert "future_rv_5d" in target_df.columns
    # El número de filas no debe cambiar
    assert len(target_df) == len(df)
    # Todas las filas deben tener la versión de target correcta
    assert target_df["target_version"].eq("v1").all()


def test_future_rv_5d_matches_manual_calculation_for_first_valid_row() -> None:
    """
    Test 3: Validación matemática del cálculo de volatilidad realizada futura.

    Para la primera fila (índice 0), se toman los retornos de los 5 días siguientes
    (índices 1 a 5), se calcula su desviación estándar muestral y se anualiza.
    Este valor debe coincidir exactamente con el calculado manualmente.
    """
    df = make_target_source_df()

    # Construimos el target
    target_df = build_continuous_targets(
        df=df,
        settings=TEST_SETTINGS,
    )

    # Extraemos los retornos futuros para la primera fila (índices 1 a 5 inclusive)
    future_returns = df.loc[1:5, "log_ret_1d"].to_numpy(dtype=float)
    # Cálculo manual: desviación estándar con ddof=1 * raíz(252)
    manual_value = np.std(future_returns, ddof=1) * math.sqrt(252)
    # Valor calculado por la función
    model_value = target_df.loc[0, "future_rv_5d"]

    # Comprobamos que son prácticamente iguales (pytest.approx para evitar errores de redondeo)
    assert model_value == pytest.approx(manual_value)


def test_last_five_rows_are_nan_when_partial_windows_not_allowed() -> None:
    """
    Test 4: Verifica que cuando no se permiten ventanas parciales
    (`allow_partial_window=False`), las últimas `horizon_days` filas (5 en este caso)
    tengan valor NaN en la columna del target, ya que no hay suficientes días futuros
    para completar la ventana requerida.
    """
    df = make_target_source_df()

    # Construimos el target
    target_df = build_continuous_targets(
        df=df,
        settings=TEST_SETTINGS,
    )

    # Las últimas 5 filas deben ser NaN
    assert target_df["future_rv_5d"].tail(5).isna().all()


def test_adapt_and_validate_target_output_pass_on_valid_dataset() -> None:
    """
    Test 5: Prueba de integración que verifica que el flujo completo de:
        1. Construcción de targets continuos.
        2. Adaptación al contrato (selección de columnas y validación Pydantic).
        3. Validación final exhaustiva.
    funcione correctamente con un dataset sintético válido.

    Asegura que:
        - El DataFrame adaptado tenga exactamente las columnas esperadas.
        - La función de validación no lance excepciones.
    """
    # DataFrame de entrada
    df = make_target_source_df()

    # Construcción del target continuo
    target_df = build_continuous_targets(
        df=df,
        settings=TEST_SETTINGS,
    )

    # Adaptación al contrato (filtra columnas y valida con Pydantic)
    contracted_df = adapt_targets_to_contract(
        df=target_df,
        settings=TEST_SETTINGS,
    )

    # Verificamos que las columnas coincidan con las habilitadas según configuración
    expected_columns = get_enabled_target_columns(TEST_SETTINGS)
    assert list(contracted_df.columns) == expected_columns

    # Validación final: no debe lanzar excepciones
    validate_target_output(
        df=contracted_df,
        settings=TEST_SETTINGS,
    )