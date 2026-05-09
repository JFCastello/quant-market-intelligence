from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.models.benchmark_regime import (
    _assign_regime_labels,
    _build_thresholds_by_split,
    build_benchmark_regime_predictions,
)


def make_benchmark_regime_settings() -> dict:
    """Crea una configuración de prueba estándar para el mapeo de régimen del benchmark."""
    return {
        "enabled": True,
        "forecast_value_column": "yhat_future_rv_5d",
        "output_regime_column": "yhat_future_regime_5d",
        "target_continuous_column": "future_rv_5d",
        "target_regime_column": "future_regime_5d",
        "threshold_low_column": "threshold_low",
        "threshold_high_column": "threshold_high",
        "threshold_source_column": "regime_thresholds_source",
        "expected_threshold_source": "train_only",
        "calm_label": "calm",
        "normal_label": "normal",
        "stress_label": "stress",
    }


def test_assign_regime_labels_maps_values_correctly() -> None:
    """
    Prueba que _assign_regime_labels asigne correctamente las etiquetas:
    - calm: valor < threshold_low
    - normal: valor entre threshold_low y threshold_high (inclusive)
    - stress: valor > threshold_high
    """
    settings = make_benchmark_regime_settings()

    forecast_values = pd.Series([0.05, 0.10, 0.20])
    threshold_low = pd.Series([0.08, 0.08, 0.08])
    threshold_high = pd.Series([0.15, 0.15, 0.15])

    result = _assign_regime_labels(
        forecast_values=forecast_values,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        settings=settings,
    )

    assert result.tolist() == ["calm", "normal", "stress"]


def test_build_thresholds_by_split_extracts_constant_thresholds() -> None:
    """
    Verifica que _build_thresholds_by_split extraiga correctamente los umbrales
    por split, asumiendo que son constantes dentro de cada split.
    """
    settings = make_benchmark_regime_settings()

    regime_df = pd.DataFrame(
        {
            "split_id": ["split_001", "split_001", "split_002", "split_002"],
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-02-01", "2022-02-02"]),
            "dataset_role": ["validation", "test", "validation", "test"],
            "future_rv_5d": [0.10, 0.12, 0.20, 0.18],
            "future_regime_5d": ["normal", "normal", "stress", "stress"],
            "threshold_low": [0.08, 0.08, 0.10, 0.10],
            "threshold_high": [0.15, 0.15, 0.22, 0.22],
            "regime_thresholds_source": ["train_only", "train_only", "train_only", "train_only"],
        }
    )

    thresholds_df = _build_thresholds_by_split(
        regime_targets_by_split_df=regime_df,
        settings=settings,
    )

    assert set(thresholds_df.columns) == {
        "split_id",
        "threshold_low",
        "threshold_high",
        "regime_thresholds_source",
    }
    assert len(thresholds_df) == 2


def test_build_thresholds_by_split_raises_when_thresholds_vary_within_split() -> None:
    """
    Comprueba que _build_thresholds_by_split lance una excepción cuando los umbrales
    no son constantes dentro de un mismo split (threshold_low varía).
    """
    settings = make_benchmark_regime_settings()

    regime_df = pd.DataFrame(
        {
            "split_id": ["split_001", "split_001"],
            "date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "dataset_role": ["validation", "test"],
            "future_rv_5d": [0.10, 0.12],
            "future_regime_5d": ["normal", "normal"],
            "threshold_low": [0.08, 0.09],  # diferente dentro del mismo split
            "threshold_high": [0.15, 0.15],
            "regime_thresholds_source": ["train_only", "train_only"],
        }
    )

    with pytest.raises(ValueError, match="not constant within split_id"):
        _build_thresholds_by_split(
            regime_targets_by_split_df=regime_df,
            settings=settings,
        )


def test_build_benchmark_regime_predictions_merges_targets_and_assigns_labels() -> None:
    """
    Prueba end-to-end de build_benchmark_regime_predictions:
    - Fusiona forecasts del benchmark con targets reales y umbrales.
    - Asigna etiquetas de régimen a los forecasts.
    - Retorna un DataFrame enriquecido con todas las columnas esperadas.
    """
    settings = make_benchmark_regime_settings()

    # DataFrame de forecasts del benchmark (GARCH)
    benchmark_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
            "split_id": ["split_001", "split_001", "split_001"],
            "dataset_role": ["validation", "validation", "test"],
            "model_name": ["garch_11_student_t"] * 3,
            "benchmark_version": ["v1"] * 3,
            "forecast_horizon_days": [5] * 3,
            "output_target_name": ["future_rv_5d"] * 3,
            "yhat_future_rv_5d": [0.05, 0.10, 0.20],
            "train_start_date": pd.to_datetime(["2018-01-02"] * 3),
            "train_end_date": pd.to_datetime(["2021-12-31"] * 3),
            "n_train": [1007] * 3,
            "fit_status": ["converged"] * 3,
            "omega": [0.04] * 3,
            "alpha_1": [0.20] * 3,
            "beta_1": [0.75] * 3,
            "nu": [6.0] * 3,
        }
    )

    # DataFrame de targets de régimen (con umbrales por split)
    regime_df = pd.DataFrame(
        {
            "split_id": ["split_001", "split_001", "split_001"],
            "split_version": ["v1"] * 3,
            "instrument_id": ["spy_us"] * 3,
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
            "dataset_role": ["validation", "validation", "test"],
            "target_version": ["v1"] * 3,
            "future_rv_5d": [0.07, 0.12, 0.18],
            "future_regime_5d": ["calm", "normal", "stress"],
            "threshold_low": [0.08, 0.08, 0.08],
            "threshold_high": [0.15, 0.15, 0.15],
            "regime_thresholds_source": ["train_only", "train_only", "train_only"],
        }
    )

    result = build_benchmark_regime_predictions(
        benchmark_forecast_df=benchmark_df,
        regime_targets_by_split_df=regime_df,
        settings=settings,
    )

    assert len(result) == 3
    assert "yhat_future_regime_5d" in result.columns
    assert "future_rv_5d" in result.columns
    assert "future_regime_5d" in result.columns
    assert result["yhat_future_regime_5d"].tolist() == ["calm", "normal", "stress"]


def test_build_benchmark_regime_predictions_requires_finite_forecasts() -> None:
    """
    Verifica que la función falle si los forecasts contienen valores NaN,
    ya que no se pueden asignar etiquetas de régimen.
    """
    settings = make_benchmark_regime_settings()

    benchmark_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03"]),
            "split_id": ["split_001"],
            "dataset_role": ["validation"],
            "yhat_future_rv_5d": [float("nan")],  # forecast inválido
        }
    )

    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03"]),
            "split_id": ["split_001"],
            "dataset_role": ["validation"],
            "future_rv_5d": [0.10],
            "future_regime_5d": ["normal"],
            "threshold_low": [0.08],
            "threshold_high": [0.15],
            "regime_thresholds_source": ["train_only"],
        }
    )

    with pytest.raises(ValueError, match="contains NaN values"):
        build_benchmark_regime_predictions(
            benchmark_forecast_df=benchmark_df,
            regime_targets_by_split_df=regime_df,
            settings=settings,
        )