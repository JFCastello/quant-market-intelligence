from __future__ import annotations

import pandas as pd

from quant_platform.evaluation import (
    build_walk_forward_splits,
    validate_split_output,
)

# Configuración de prueba para splits walk‑forward
TEST_SETTINGS = {
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
    }
}


def make_split_source_df(
    instrument_id: str = "spy_us",
    start_date: str = "2018-01-01",
    n_rows: int = 1800,
) -> pd.DataFrame:
    """Crea un DataFrame de prueba con fechas hábiles y datos dummy para un instrumento."""
    dates = pd.bdate_range(start=start_date, periods=n_rows)

    return pd.DataFrame(
        {
            "instrument_id": [instrument_id] * len(dates),
            "date": dates,
            "future_rv_5d": [0.10] * len(dates),  # columna adicional irrelevante para splits
        }
    )


def test_build_walk_forward_splits_returns_non_empty_dataframe() -> None:
    """
    Prueba que build_walk_forward_splits genere un DataFrame no vacío
    y que contenga las columnas esenciales.
    """
    df = make_split_source_df()

    split_df = build_walk_forward_splits(
        df=df,
        settings=TEST_SETTINGS,
    )

    assert not split_df.empty
    assert "split_id" in split_df.columns
    assert "instrument_id" in split_df.columns
    assert "train_start" in split_df.columns
    assert "test_end" in split_df.columns


def test_split_ids_are_sequential_for_single_instrument() -> None:
    """
    Verifica que los identificadores de split sean secuenciales (split_001, split_002, ...)
    para un único instrumento.
    """
    df = make_split_source_df()

    split_df = build_walk_forward_splits(
        df=df,
        settings=TEST_SETTINGS,
    )

    observed = split_df["split_id"].tolist()
    expected = [f"split_{i:03d}" for i in range(1, len(observed) + 1)]

    assert observed == expected


def test_train_validation_test_are_temporally_ordered() -> None:
    """
    Comprueba que las ventanas de entrenamiento, validación y prueba
    estén correctamente ordenadas en el tiempo:
    train_start <= train_end < validation_start <= validation_end < test_start <= test_end
    """
    df = make_split_source_df()

    split_df = build_walk_forward_splits(
        df=df,
        settings=TEST_SETTINGS,
    )

    for row in split_df.itertuples(index=False):
        assert row.train_start <= row.train_end
        assert row.train_end < row.validation_start
        assert row.validation_start <= row.validation_end
        assert row.validation_end < row.test_start
        assert row.test_start <= row.test_end


def test_train_rows_are_non_decreasing_across_folds() -> None:
    """
    Verifica que el número de filas de entrenamiento nunca disminuya
    a medida que avanzan los folds (expanding window).
    """
    df = make_split_source_df()

    split_df = build_walk_forward_splits(
        df=df,
        settings=TEST_SETTINGS,
    )

    diffs = split_df["train_rows"].diff().dropna()
    assert (diffs >= 0).all()


def test_validate_split_output_accepts_valid_split_dataframe() -> None:
    """
    Prueba que validate_split_output no lance excepción
    cuando recibe un DataFrame de splits válido.
    """
    df = make_split_source_df()

    split_df = build_walk_forward_splits(
        df=df,
        settings=TEST_SETTINGS,
    )

    # No debe lanzar excepción
    validate_split_output(
        split_df=split_df,
        settings=TEST_SETTINGS,
    )


def test_build_walk_forward_splits_supports_multiple_instruments() -> None:
    """
    Verifica que la función soporte múltiples instrumentos:
    - Genera splits para cada instrumento.
    - Los split_id son secuenciales por instrumento.
    """
    df_spy = make_split_source_df(instrument_id="spy_us", start_date="2018-01-01", n_rows=1800)
    df_tlt = make_split_source_df(instrument_id="tlt_us", start_date="2018-01-01", n_rows=1800)

    universe_df = pd.concat([df_spy, df_tlt], ignore_index=True)

    split_df = build_walk_forward_splits(
        df=universe_df,
        settings=TEST_SETTINGS,
    )

    # Ambos instrumentos están presentes
    observed_ids = set(split_df["instrument_id"].unique().tolist())
    assert observed_ids == {"spy_us", "tlt_us"}

    # Los split_ids son secuenciales dentro de cada instrumento
    for instrument_id, g in split_df.groupby("instrument_id", sort=True):
        observed = g["split_id"].tolist()
        expected = [f"split_{i:03d}" for i in range(1, len(observed) + 1)]
        assert observed == expected