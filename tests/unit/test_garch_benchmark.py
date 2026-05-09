from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.models.garch_benchmark import (
    _forecast_variance_path_from_observed_state,
    _normalize_split_df_for_benchmark,
    build_benchmark_input_df,
    build_garch_benchmark_forecasts_by_split,
    fit_garch_on_train_returns,
)


def make_test_benchmark_settings() -> dict:
    """Crea una configuración de prueba estándar para el benchmark GARCH."""
    return {
        "benchmark_name": "garch_11_student_t",
        "benchmark_version": "v1",
        "enabled": True,
        "mean_model": "zero",
        "vol_model": "garch",
        "p": 1,
        "o": 0,
        "q": 1,
        "distribution": "studentst",
        "input_price_column": "close",
        "return_type": "log",
        "return_column_name": "ret_1d",
        "fit_scale": 100.0,
        "annualization_factor": 252,
        "forecast_horizon_days": 5,
        "output_target_name": "future_rv_5d",
        "min_train_points": 20,
        "score_roles": ["validation", "test"],
        "enforce_positive_forecasts": True,
        "persist_fit_params": True,
    }


def test_build_benchmark_input_df_computes_scaled_log_returns() -> None:
    """
    Prueba que build_benchmark_input_df calcule correctamente los retornos logarítmicos
    escalados según fit_scale, y que mantenga las columnas esperadas.
    """
    settings = make_test_benchmark_settings()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
            "close": [100.0, 110.0, 121.0],
            "symbol": ["SPY", "SPY", "SPY"],
        }
    )

    result = build_benchmark_input_df(normalized_df=df, settings=settings)

    # Retornos esperados: log(P_t / P_{t-1}) * fit_scale (100)
    expected_r1 = np.log(110.0 / 100.0) * 100.0
    expected_r2 = np.log(121.0 / 110.0) * 100.0

    assert list(result.columns) == ["date", "close", "ret_1d", "symbol"]
    assert pd.isna(result.loc[0, "ret_1d"])   # Primer retorno es NaN
    assert np.isclose(result.loc[1, "ret_1d"], expected_r1)
    assert np.isclose(result.loc[2, "ret_1d"], expected_r2)


def test_forecast_variance_path_from_observed_state_has_expected_shape() -> None:
    """
    Verifica que _forecast_variance_path_from_observed_state genere una trayectoria
    de varianzas de la longitud correcta (horizon), con valores no negativos,
    y que el primer paso use el residuo observado y los siguientes la persistencia.
    """
    path = _forecast_variance_path_from_observed_state(
        epsilon_t=1.0,
        sigma2_t=2.0,
        omega=0.1,
        alpha_1=0.2,
        beta_1=0.7,
        horizon=5,
    )

    assert len(path) == 5
    assert np.all(path >= 0.0)
    # Primer paso: omega + alpha * epsilon^2 + beta * sigma2_t
    assert np.isclose(path[0], 0.1 + 0.2 * (1.0 ** 2) + 0.7 * 2.0)
    # Siguientes: omega + (alpha+beta) * varianza anterior
    assert np.isclose(path[1], 0.1 + (0.2 + 0.7) * path[0])


def test_normalize_split_df_for_benchmark_expands_interval_splits() -> None:
    """
    Prueba que _normalize_split_df_for_benchmark expanda correctamente los intervalos
    de train/validation/test en un DataFrame con una fila por fecha y rol asignado.
    """
    dates = pd.bdate_range("2022-01-03", periods=10)

    split_df = pd.DataFrame(
        {
            "split_id": ["split_001"],
            "train_start": [dates[0]],
            "train_end": [dates[3]],
            "validation_start": [dates[4]],
            "validation_end": [dates[6]],
            "test_start": [dates[7]],
            "test_end": [dates[9]],
            "train_rows": [4],
            "validation_rows": [3],
            "test_rows": [3],
            "regime_thresholds_source": ["train_only"],
        }
    )

    normalized = _normalize_split_df_for_benchmark(
        split_df=split_df,
        available_dates=pd.Series(dates),
    )

    assert set(normalized.columns) == {"date", "split_id", "dataset_role"}
    assert len(normalized) == 10
    assert (normalized["dataset_role"] == "train").sum() == 4
    assert (normalized["dataset_role"] == "validation").sum() == 3
    assert (normalized["dataset_role"] == "test").sum() == 3


def test_fit_garch_on_train_returns_returns_expected_metadata() -> None:
    """
    Comprueba que fit_garch_on_train_returns devuelva la estructura esperada:
    número de puntos, objeto result, parámetros y valores finitos.
    """
    settings = make_test_benchmark_settings()

    rng = np.random.default_rng(7)
    train_returns = pd.Series(rng.normal(loc=0.0, scale=1.0, size=100))

    fit_info = fit_garch_on_train_returns(
        train_returns=train_returns,
        settings=settings,
    )

    assert fit_info["n_train"] == 100
    assert "result" in fit_info
    assert "params" in fit_info
    assert np.isfinite(fit_info["sigma2_last"])
    assert np.isfinite(fit_info["epsilon_last"])
    assert np.isfinite(fit_info["params"]["omega"])
    assert np.isfinite(fit_info["params"]["alpha_1"])
    assert np.isfinite(fit_info["params"]["beta_1"])


def test_build_garch_benchmark_forecasts_by_split_returns_scored_rows_only() -> None:
    """
    Prueba end-to-end de build_garch_benchmark_forecasts_by_split:
    - Solo genera filas para validation y test (no train).
    - Los forecasts son positivos y finitos.
    - Los parámetros del modelo son constantes dentro de cada split.
    """
    settings = make_test_benchmark_settings()

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=80)

    # Simular retornos y precios
    returns = rng.normal(loc=0.0002, scale=0.01, size=len(dates))
    prices = 100.0 * np.exp(np.cumsum(returns))

    normalized_df = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "symbol": ["SPY"] * len(dates),
        }
    )

    split_df = pd.DataFrame(
        {
            "split_id": ["split_001"],
            "instrument_id": ["spy"],
            "train_start": [dates[0]],
            "train_end": [dates[39]],
            "validation_start": [dates[40]],
            "validation_end": [dates[59]],
            "test_start": [dates[60]],
            "test_end": [dates[79]],
            "train_rows": [40],
            "validation_rows": [20],
            "test_rows": [20],
            "regime_thresholds_source": ["train_only"],
        }
    )

    forecast_df = build_garch_benchmark_forecasts_by_split(
        normalized_df=normalized_df,
        split_df=split_df,
        settings=settings,
        symbol="SPY",
    )

    assert not forecast_df.empty
    assert set(forecast_df["dataset_role"].unique()) == {"validation", "test"}
    assert set(forecast_df["split_id"].unique()) == {"split_001"}
    assert (forecast_df["yhat_future_rv_5d"] > 0).all()
    assert np.isfinite(forecast_df["yhat_future_rv_5d"]).all()

    # Los parámetros deben ser constantes dentro del split
    assert forecast_df["omega"].nunique() == 1
    assert forecast_df["alpha_1"].nunique() == 1
    assert forecast_df["beta_1"].nunique() == 1
    assert forecast_df["n_train"].nunique() == 1