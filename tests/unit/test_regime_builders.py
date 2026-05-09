from __future__ import annotations

import pandas as pd
import pytest

# Importaciones de las funciones a probar del módulo targets
from quant_platform.targets import (
    apply_regime_labels,
    attach_regime_target_to_dataframe,
    build_regime_target_series,
    compute_quantile_thresholds,
    validate_regime_output_series,
    validate_regime_thresholds_metadata,
)

# Configuración de prueba que simula la estructura real de settings
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
    }
}


def make_train_source_series() -> pd.Series:
    """Crea una serie de entrenamiento con valores conocidos para calcular cuantiles."""
    return pd.Series(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        name="future_rv_5d",
        dtype=float,
    )


def make_apply_source_series() -> pd.Series:
    """Crea una serie de aplicación (incluye un valor nulo al final) para probar la asignación."""
    return pd.Series(
        [0.10, 0.25, 0.35, 0.70, None],
        name="future_rv_5d",
        dtype=float,
    )

def make_base_target_df() -> pd.DataFrame:
    """Construye un DataFrame base con las columnas mínimas necesarias para las pruebas."""
    return pd.DataFrame(
        {
            "instrument_id": ["spy_us"] * 5,
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "target_version": ["v1"] * 5,
            "future_rv_5d": [0.10, 0.25, 0.45, 0.70, None],
        }
    )


def test_compute_quantile_thresholds_returns_expected_metadata() -> None:
    """
    Prueba que compute_quantile_thresholds genere los metadatos esperados:
    - Campos correctos (source, nombre, método, etiquetas, cuantiles, etc.)
    - Los cuantiles q1 y q2 son números y están ordenados.
    - Los metadatos pasan la validación.
    """
    train_series = make_train_source_series()

    metadata = compute_quantile_thresholds(
        train_source_series=train_series,
        settings=TEST_SETTINGS,
    )

    # Verificar valores literales de la configuración
    assert metadata["source_continuous_target"] == "future_rv_5d"
    assert metadata["regime_target_name"] == "future_regime_5d"
    assert metadata["method"] == "quantile_bins"
    assert metadata["labels"] == ["calm", "normal", "stress"]
    assert metadata["quantiles"] == [0.33, 0.66]
    assert metadata["thresholds_source"] == "train_only"

    # Verificar tipo y orden de los umbrales calculados
    assert isinstance(metadata["q1"], float)
    assert isinstance(metadata["q2"], float)
    assert metadata["q1"] <= metadata["q2"]

    # Confirmar que los metadatos son consistentes con la configuración
    validate_regime_thresholds_metadata(metadata, TEST_SETTINGS)


def test_apply_regime_labels_assigns_expected_classes_and_preserves_nan() -> None:
    """
    Prueba que apply_regime_labels asigne correctamente las etiquetas según los umbrales:
    - Calm para valores <= q1
    - Normal entre q1 y q2
    - Stress para > q2
    - Los valores nulos se mantienen como NA (allow_missing_source=True)
    """
    train_series = make_train_source_series()
    source_series = make_apply_source_series()

    metadata = compute_quantile_thresholds(
        train_source_series=train_series,
        settings=TEST_SETTINGS,
    )

    regime_series = apply_regime_labels(
        source_series=source_series,
        thresholds_metadata=metadata,
        settings=TEST_SETTINGS,
    )

    # Comprobaciones puntuales según los valores de source_series
    assert regime_series.iloc[0] == "calm"      # 0.10 <= q1
    assert regime_series.iloc[1] == "calm"      # 0.25 <= q1 (aprox)
    assert regime_series.iloc[2] == "normal"    # q1 < 0.45 <= q2
    assert regime_series.iloc[3] == "stress"    # 0.70 > q2
    assert pd.isna(regime_series.iloc[4])       # None se mantiene como NA

    # La serie generada solo contiene etiquetas válidas
    validate_regime_output_series(regime_series, TEST_SETTINGS)


def test_build_regime_target_series_returns_series_and_metadata() -> None:
    """
    Prueba que build_regime_target_series orquesta correctamente el cálculo de umbrales
    y la aplicación de etiquetas, devolviendo tanto la serie como los metadatos.
    """
    train_series = make_train_source_series()
    source_series = make_apply_source_series()

    regime_series, metadata = build_regime_target_series(
        source_series=source_series,
        train_source_series=train_series,
        settings=TEST_SETTINGS,
    )

    # La serie resultante tiene el mismo largo que la fuente
    assert len(regime_series) == len(source_series)

    # Metadatos clave presentes
    assert metadata["regime_target_name"] == "future_regime_5d"
    assert metadata["source_continuous_target"] == "future_rv_5d"

    # Ambos, metadatos y serie, son válidos según las reglas de negocio
    validate_regime_thresholds_metadata(metadata, TEST_SETTINGS)
    validate_regime_output_series(regime_series, TEST_SETTINGS)


def test_attach_regime_target_to_dataframe_adds_expected_column() -> None:
    """
    Prueba que attach_regime_target_to_dataframe agregue la columna de régimen
    al DataFrame original sin modificar las otras columnas y respetando los índices.
    """
    base_df = make_base_target_df()
    train_series = make_train_source_series()

    # Construir la serie de régimen a partir de la columna 'future_rv_5d' del DataFrame
    regime_series, _ = build_regime_target_series(
        source_series=base_df["future_rv_5d"],
        train_source_series=train_series,
        settings=TEST_SETTINGS,
    )

    out_df = attach_regime_target_to_dataframe(
        df=base_df,
        regime_series=regime_series,
        settings=TEST_SETTINGS,
    )

    # La nueva columna existe y tiene el nombre esperado
    assert "future_regime_5d" in out_df.columns
    assert len(out_df) == len(base_df)

    # Verificar algunos valores (coinciden con los de test_apply_regime_labels)
    assert out_df["future_regime_5d"].iloc[0] == "calm"
    assert pd.isna(out_df["future_regime_5d"].iloc[-1])


def test_validate_regime_output_series_rejects_invalid_labels() -> None:
    """
    Prueba que validate_regime_output_series lance una excepción cuando la serie
    contiene etiquetas no definidas en la configuración (por ejemplo, 'panic').
    """
    invalid_series = pd.Series(["calm", "panic", "stress"], dtype="object")

    with pytest.raises(ValueError, match="invalid labels"):
        validate_regime_output_series(invalid_series, TEST_SETTINGS)