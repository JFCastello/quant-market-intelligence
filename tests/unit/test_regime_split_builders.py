from __future__ import annotations

import pandas as pd

from quant_platform.targets import (
    build_regime_targets_by_split,
    build_regime_targets_for_single_split,
    validate_regime_split_output,
)

# Configuración de prueba que incluye las secciones 'targets' y 'splits'
TEST_SETTINGS = {
    "targets": {
        "target_version": "v1",
        "continuous_target": {
            "name": "future_rv_5d",
            "enabled": True,
            "horizon_days": 5,
            "base_return_column": "log_ret_1d",
            "annualization_factor": 252,
            "min_periods": 5,
            "window_type": "fixed_forward",
            "allow_partial_window": False,
        },
        "classification_target": {
            "name": "future_regime_5d",
            "enabled": True,
            "source_continuous_target": "future_rv_5d",
            "method": "quantile_bins",
            "labels": ["calm", "normal", "stress"],
            "n_classes": 3,
            "quantiles": [0.33, 0.66],
            "thresholds_source": "train_only",
            "lower_bin_inclusive": True,
            "upper_bin_inclusive": False,
            "allow_missing_source": True,
        },
        "merge_keys": ["instrument_id", "date"],
        "source_layer": "targets_v1",
    },
    "splits": {
        "split_version": "v1",
        "method": "walk_forward_expanding",
        "date_column": "date",
        "group_key": "instrument_id",
        "train_years": 2,
        "validation_months": 3,
        "test_months": 3,
        "step_months": 3,
        "min_train_observations": 200,
        "allow_partial_last_fold": False,
        "require_validation": True,
        "regime_thresholds_source": "train_only",
        "output_format": "fold_table",
    },
}


def make_target_df(
    instrument_id: str = "spy_us",
    start_date: str = "2020-01-01",
    n_rows: int = 900,
) -> pd.DataFrame:
    """Crea un DataFrame de target continuo (future_rv_5d) con valores aleatorios y algunos nulos."""
    dates = pd.bdate_range(start=start_date, periods=n_rows)

    # Generar valores: la mayoría numéricos, algunos nulos cada 17 filas
    values = []
    for i in range(n_rows):
        if i % 17 == 0:
            values.append(None)
        else:
            values.append(0.05 + 0.0002 * i)

    return pd.DataFrame(
        {
            "instrument_id": [instrument_id] * len(dates),
            "date": dates,
            "target_version": ["v1"] * len(dates),
            "future_rv_5d": values,
        }
    )


def make_split_df(instrument_id: str = "spy_us") -> pd.DataFrame:
    """Crea un DataFrame de splits con dos pliegues (split_001 y split_002) para pruebas."""
    return pd.DataFrame(
        {
            "split_id": ["split_001", "split_002"],
            "split_version": ["v1", "v1"],
            "instrument_id": [instrument_id, instrument_id],
            "train_start": pd.to_datetime(["2020-01-01", "2020-04-01"]),
            "train_end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "validation_start": pd.to_datetime(["2020-04-01", "2020-07-01"]),
            "validation_end": pd.to_datetime(["2020-06-30", "2020-09-30"]),
            "test_start": pd.to_datetime(["2020-07-01", "2020-10-01"]),
            "test_end": pd.to_datetime(["2020-09-30", "2020-12-31"]),
            "regime_thresholds_source": ["train_only", "train_only"],
        }
    )


def test_build_regime_targets_for_single_split_returns_expected_columns() -> None:
    """
    Prueba que build_regime_targets_for_single_split devuelva un DataFrame
    con todas las columnas esperadas y que los metadatos contengan la información correcta.
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df, metadata = build_regime_targets_for_single_split(
        instrument_target_df=target_df,
        split_row=split_df.iloc[0],
        settings=TEST_SETTINGS,
    )

    expected_cols = {
        "split_id",
        "split_version",
        "instrument_id",
        "date",
        "dataset_role",
        "target_version",
        "future_rv_5d",
        "future_regime_5d",
        "threshold_low",
        "threshold_high",
        "regime_thresholds_source",
    }

    assert set(out_df.columns) == expected_cols
    assert metadata["regime_target_name"] == "future_regime_5d"
    assert metadata["source_continuous_target"] == "future_rv_5d"


def test_single_split_contains_train_validation_and_test_roles() -> None:
    """
    Verifica que para un split individual se generen filas con los tres roles:
    'train', 'validation' y 'test'.
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df, _ = build_regime_targets_for_single_split(
        instrument_target_df=target_df,
        split_row=split_df.iloc[0],
        settings=TEST_SETTINGS,
    )

    observed_roles = set(out_df["dataset_role"].dropna().unique().tolist())
    assert observed_roles == {"train", "validation", "test"}


def test_threshold_columns_are_constant_within_a_single_split() -> None:
    """
    Comprueba que dentro de un mismo split, los umbrales threshold_low y threshold_high
    sean constantes (calculados una vez con el conjunto de entrenamiento) y que estén ordenados.
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df, _ = build_regime_targets_for_single_split(
        instrument_target_df=target_df,
        split_row=split_df.iloc[0],
        settings=TEST_SETTINGS,
    )

    assert out_df["threshold_low"].nunique(dropna=False) == 1
    assert out_df["threshold_high"].nunique(dropna=False) == 1
    assert (out_df["threshold_low"] <= out_df["threshold_high"]).all()


def test_build_regime_targets_by_split_returns_multiple_splits() -> None:
    """
    Prueba que la función orquestadora build_regime_targets_by_split procese
    correctamente múltiples splits y devuelva los dos esperados.
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df = build_regime_targets_by_split(
        target_df=target_df,
        split_df=split_df,
        settings=TEST_SETTINGS,
    )

    observed_split_ids = out_df["split_id"].drop_duplicates().tolist()
    assert observed_split_ids == ["split_001", "split_002"]


def test_validate_regime_split_output_accepts_valid_output() -> None:
    """
    Verifica que validate_regime_split_output no lance excepción
    cuando recibe un DataFrame de régimen por split correctamente construido.
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df = build_regime_targets_by_split(
        target_df=target_df,
        split_df=split_df,
        settings=TEST_SETTINGS,
    )

    # No debe lanzar excepción
    validate_regime_split_output(
        regime_split_df=out_df,
        settings=TEST_SETTINGS,
    )


def test_build_regime_targets_by_split_preserves_missing_source_as_missing_label() -> None:
    """
    Comprueba que cuando el target continuo (future_rv_5d) tiene valores nulos,
    la etiqueta de régimen correspondiente también sea nula (según allow_missing_source=True).
    """
    target_df = make_target_df()
    split_df = make_split_df()

    out_df = build_regime_targets_by_split(
        target_df=target_df,
        split_df=split_df,
        settings=TEST_SETTINGS,
    )

    missing_source_mask = out_df["future_rv_5d"].isna()
    assert missing_source_mask.any()

    assert out_df.loc[missing_source_mask, "future_regime_5d"].isna().all()