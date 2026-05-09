from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Verifica que el DataFrame contenga todas las columnas requeridas. Lanza error si falta alguna."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def _validate_benchmark_regime_settings(settings: Mapping[str, Any]) -> None:
    """
    Valida que la configuración para el mapeo de régimen del benchmark contenga
    todas las claves necesarias.
    """
    required_keys = {
        "forecast_value_column",
        "output_regime_column",
        "target_continuous_column",
        "target_regime_column",
        "threshold_low_column",
        "threshold_high_column",
        "threshold_source_column",
        "expected_threshold_source",
        "calm_label",
        "normal_label",
        "stress_label",
    }
    missing = required_keys - set(settings.keys())
    if missing:
        raise ValueError(f"Benchmark regime settings missing keys: {sorted(missing)}")


def _build_thresholds_by_split(
    regime_targets_by_split_df: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Extrae los umbrales (threshold_low, threshold_high) por split a partir del DataFrame
    de targets de régimen. Valida que sean constantes dentro de cada split,
    que estén ordenados (low < high) y que la fuente de umbrales coincida con la esperada.
    Retorna un DataFrame con una fila por split y las columnas de umbrales.
    """
    threshold_low_col = settings["threshold_low_column"]
    threshold_high_col = settings["threshold_high_column"]
    threshold_source_col = settings["threshold_source_column"]

    threshold_cols = {
        "split_id",
        threshold_low_col,
        threshold_high_col,
        threshold_source_col,
    }
    _require_columns(regime_targets_by_split_df, threshold_cols, "regime_targets_by_split_df")

    # Verificar que las columnas de umbral sean constantes por split
    grouped = (
        regime_targets_by_split_df.groupby("split_id", dropna=False)[
            [threshold_low_col, threshold_high_col, threshold_source_col]
        ]
        .nunique(dropna=False)
        .reset_index()
    )

    inconsistent = grouped.loc[
        (grouped[threshold_low_col] != 1)
        | (grouped[threshold_high_col] != 1)
        | (grouped[threshold_source_col] != 1)
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Threshold columns are not constant within split_id. "
            f"Inconsistent splits: {inconsistent['split_id'].tolist()}"
        )

    # Obtener los umbrales únicos por split
    thresholds_df = (
        regime_targets_by_split_df[
            ["split_id", threshold_low_col, threshold_high_col, threshold_source_col]
        ]
        .drop_duplicates()
        .sort_values("split_id")
        .reset_index(drop=True)
    )

    # Validar orden de umbrales (low debe ser menor que high)
    if (thresholds_df[threshold_low_col] >= thresholds_df[threshold_high_col]).any():
        bad_rows = thresholds_df.loc[
            thresholds_df[threshold_low_col] >= thresholds_df[threshold_high_col]
        ]
        raise ValueError(
            "Found split(s) with threshold_low >= threshold_high:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # Validar la fuente de umbrales (ej. 'train_only')
    expected_source = settings["expected_threshold_source"]
    actual_sources = set(thresholds_df[threshold_source_col].dropna().unique())
    if actual_sources != {expected_source}:
        raise ValueError(
            f"Unexpected threshold source values: {sorted(actual_sources)} "
            f"(expected only: {expected_source})"
        )

    return thresholds_df


def _assign_regime_labels(
    forecast_values: pd.Series,
    threshold_low: pd.Series,
    threshold_high: pd.Series,
    settings: Mapping[str, Any],
) -> pd.Series:
    """
    Asigna etiquetas de régimen (calm, normal, stress) a los valores pronosticados
    según los umbrales proporcionados. Lanza error si algún valor no se puede mapear.
    """
    calm_label = settings["calm_label"]
    normal_label = settings["normal_label"]
    stress_label = settings["stress_label"]

    output = pd.Series(index=forecast_values.index, dtype="object")

    calm_mask = forecast_values < threshold_low
    normal_mask = (forecast_values >= threshold_low) & (forecast_values <= threshold_high)
    stress_mask = forecast_values > threshold_high

    output.loc[calm_mask] = calm_label
    output.loc[normal_mask] = normal_label
    output.loc[stress_mask] = stress_label

    # Si quedan valores sin asignar (por NaN o fuera de rangos), reportar error
    if output.isna().any():
        bad_rows = pd.DataFrame(
            {
                "forecast": forecast_values,
                "threshold_low": threshold_low,
                "threshold_high": threshold_high,
                "assigned_label": output,
            }
        ).loc[output.isna()]
        raise ValueError(
            "Some benchmark forecasts could not be mapped to regime labels:\n"
            f"{bad_rows.head().to_string(index=False)}"
        )

    return output


def build_benchmark_regime_predictions(
    benchmark_forecast_df: pd.DataFrame,
    regime_targets_by_split_df: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Construye predicciones discretas de régimen a partir de los forecasts continuos
    del benchmark GARCH, utilizando los umbrales (threshold_low, threshold_high)
    materializados en regime_targets_by_split_df.

    Proceso:
    1. Valida configuraciones y columnas de entrada.
    2. Extrae umbrales por split (constantes dentro de cada split).
    3. Combina forecasts con los targets realizados (continuo y régimen real).
    4. Asigna etiquetas de régimen a cada forecast según los umbrales.
    5. Retorna un DataFrame enriquecido con la predicción de régimen del benchmark.

    Output incluye: todas las columnas del benchmark_forecast_df más
    target_continuous, target_regime, threshold_low, threshold_high,
    threshold_source y output_regime_column.
    """
    _validate_benchmark_regime_settings(settings)

    forecast_value_col = settings["forecast_value_column"]
    output_regime_col = settings["output_regime_column"]
    target_continuous_col = settings["target_continuous_column"]
    target_regime_col = settings["target_regime_column"]
    threshold_low_col = settings["threshold_low_column"]
    threshold_high_col = settings["threshold_high_column"]
    threshold_source_col = settings["threshold_source_column"]

    required_benchmark_cols = {
        "date",
        "split_id",
        "dataset_role",
        forecast_value_col,
    }
    required_regime_cols = {
        "date",
        "split_id",
        "dataset_role",
        target_continuous_col,
        target_regime_col,
        threshold_low_col,
        threshold_high_col,
        threshold_source_col,
    }

    _require_columns(benchmark_forecast_df, required_benchmark_cols, "benchmark_forecast_df")
    _require_columns(regime_targets_by_split_df, required_regime_cols, "regime_targets_by_split_df")

    benchmark_df = benchmark_forecast_df.copy()
    regime_df = regime_targets_by_split_df.copy()

    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], errors="raise")
    regime_df["date"] = pd.to_datetime(regime_df["date"], errors="raise")

    # Validar que los forecasts sean numéricos válidos
    if benchmark_df[forecast_value_col].isna().any():
        raise ValueError(f"{forecast_value_col} contains NaN values in benchmark_forecast_df")

    if not np.isfinite(benchmark_df[forecast_value_col]).all():
        raise ValueError(f"{forecast_value_col} contains non-finite values in benchmark_forecast_df")

    # Obtener umbrales por split
    thresholds_df = _build_thresholds_by_split(
        regime_targets_by_split_df=regime_df,
        settings=settings,
    )

    # Extraer los targets realizados (únicos por split/fecha/rol)
    realized_targets_df = regime_df[
        [
            "split_id",
            "date",
            "dataset_role",
            target_continuous_col,
            target_regime_col,
        ]
    ].drop_duplicates()

    # Unir forecasts con targets realizados
    merged = benchmark_df.merge(
        realized_targets_df,
        on=["split_id", "date", "dataset_role"],
        how="inner",
        validate="one_to_one",
    )

    # Agregar umbrales por split
    merged = merged.merge(
        thresholds_df,
        on="split_id",
        how="left",
        validate="many_to_one",
    )

    if merged.empty:
        raise ValueError("Merged benchmark regime output is empty")

    # Asignar etiquetas de régimen a los forecasts
    merged[output_regime_col] = _assign_regime_labels(
        forecast_values=merged[forecast_value_col],
        threshold_low=merged[threshold_low_col],
        threshold_high=merged[threshold_high_col],
        settings=settings,
    )

    merged = merged.sort_values(["split_id", "date"]).reset_index(drop=True)
    return merged