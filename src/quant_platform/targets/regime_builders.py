from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _targets_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae la sección 'targets' de la configuración.
    Lanza un error si no existe.
    """
    if "targets" not in settings:
        raise KeyError("Missing `targets` section in settings.")
    return settings["targets"]


def _classification_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae la subsección 'classification_target' dentro de 'targets'.
    Útil para obtener la configuración específica del régimen de clasificación.
    """
    targets_cfg = _targets_cfg(settings)

    if "classification_target" not in targets_cfg:
        raise KeyError("Missing `classification_target` section in `targets`.")

    return targets_cfg["classification_target"]


def get_regime_target_name(settings: dict[str, Any]) -> str:
    """
    Devuelve el nombre de la columna objetivo (régimen) definido en la configuración.
    """
    return _classification_cfg(settings)["name"]


def validate_regime_source_series(
    source_series: pd.Series,
    settings: dict[str, Any],
) -> None:
    """
    Valida que la serie fuente (por ejemplo, un rendimiento o volatilidad) sea apta
    para calcular los umbrales de régimen. Comprueba:
    - Que no esté vacía.
    - Que sus valores sean numéricos (convierte a numérico, errores -> NaN).
    - Que no contenga valores infinitos.
    - Que después de eliminar NaN queden al menos un valor.
    """
    cfg = _classification_cfg(settings)
    source_name = cfg["source_continuous_target"]

    if source_series.empty:
        raise ValueError("Regime source series is empty.")

    # Convertir a numérico; los no numéricos se vuelven NaN
    numeric_series = pd.to_numeric(source_series, errors="coerce")

    # Si hay más NaN después de la conversión, es porque habían no numéricos
    if numeric_series.isna().sum() > source_series.isna().sum():
        raise ValueError(
            f"Regime source series `{source_name}` contains non-numeric values."
        )

    # Detectar infinitos
    inf_mask = np.isinf(numeric_series.to_numpy(dtype=float, copy=True))
    if inf_mask.any():
        raise ValueError(
            f"Regime source series `{source_name}` contains `inf` or `-inf` values."
        )

    # Al menos un valor no nulo debe existir
    if numeric_series.dropna().empty:
        raise ValueError(
            f"Regime source series `{source_name}` has no non-null values to compute thresholds."
        )


def compute_quantile_thresholds(
    train_source_series: pd.Series,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Calcula los umbrales (cuantiles) a partir de la serie fuente de entrenamiento.
    Solo soporta el método 'quantile_bins' y exactamente 3 clases (bajo, normal, estrés).
    Retorna un diccionario con metadatos y los valores de los dos cuantiles (q1, q2).
    """
    cfg = _classification_cfg(settings)

    if not cfg["enabled"]:
        raise ValueError("classification_target is disabled in settings.")

    if cfg["method"] != "quantile_bins":
        raise NotImplementedError(
            f"Unsupported classification_target method: {cfg['method']}"
        )

    # Validar que la serie fuente sea correcta
    validate_regime_source_series(train_source_series, settings)

    labels = cfg["labels"]
    quantiles = cfg["quantiles"]
    n_classes = cfg["n_classes"]

    # Consistencia de la configuración
    if len(labels) != n_classes:
        raise ValueError(
            f"`labels` length ({len(labels)}) must match `n_classes` ({n_classes})."
        )

    if n_classes != 3:
        raise NotImplementedError(
            "This initial implementation supports exactly 3 classes."
        )

    if len(quantiles) != 2:
        raise ValueError(
            "This initial implementation expects exactly 2 quantiles for 3 classes."
        )

    # Limpiar NaN y calcular cuantiles
    clean_series = pd.to_numeric(train_source_series, errors="coerce").dropna()

    q1 = float(clean_series.quantile(quantiles[0]))
    q2 = float(clean_series.quantile(quantiles[1]))

    if q1 > q2:
        raise ValueError(
            f"Computed thresholds are not ordered: q1={q1}, q2={q2}."
        )

    # Almacenar metadatos útiles para aplicar la misma transformación después
    metadata = {
        "source_continuous_target": cfg["source_continuous_target"],
        "regime_target_name": cfg["name"],
        "method": cfg["method"],
        "labels": list(labels),
        "quantiles": list(quantiles),
        "thresholds_source": cfg["thresholds_source"],
        "lower_bin_inclusive": cfg["lower_bin_inclusive"],
        "upper_bin_inclusive": cfg["upper_bin_inclusive"],
        "allow_missing_source": cfg["allow_missing_source"],
        "q1": q1,
        "q2": q2,
    }

    return metadata


def apply_regime_labels(
    source_series: pd.Series,
    thresholds_metadata: dict[str, Any],
    settings: dict[str, Any],
) -> pd.Series:
    """
    Aplica las etiquetas de régimen a una serie fuente usando los umbrales precalculados.
    Soporta valores faltantes según la configuración 'allow_missing_source'.
    Retorna una serie con las etiquetas (calm, normal, stress) según corresponda.
    """
    cfg = _classification_cfg(settings)

    if not cfg["enabled"]:
        raise ValueError("classification_target is disabled in settings.")

    # Asegurar que los metadatos sean consistentes con la configuración
    validate_regime_thresholds_metadata(thresholds_metadata, settings)

    labels = thresholds_metadata["labels"]
    q1 = thresholds_metadata["q1"]
    q2 = thresholds_metadata["q2"]
    allow_missing_source = thresholds_metadata["allow_missing_source"]

    label_calm = labels[0]
    label_normal = labels[1]
    label_stress = labels[2]

    # Convertir a numérico; los no numéricos se vuelven NaN
    numeric_series = pd.to_numeric(source_series, errors="coerce")

    out = pd.Series(index=source_series.index, dtype="object")

    # Manejo de valores ausentes
    missing_mask = numeric_series.isna()
    if missing_mask.any():
        if allow_missing_source:
            out.loc[missing_mask] = pd.NA
        else:
            raise ValueError(
                "Source series contains missing values but `allow_missing_source` is false."
            )

    # Asignar etiquetas según los umbrales
    calm_mask = numeric_series <= q1
    normal_mask = (numeric_series > q1) & (numeric_series <= q2)
    stress_mask = numeric_series > q2

    out.loc[calm_mask.fillna(False)] = label_calm
    out.loc[normal_mask.fillna(False)] = label_normal
    out.loc[stress_mask.fillna(False)] = label_stress

    return out


def build_regime_target_series(
    source_series: pd.Series,
    train_source_series: pd.Series,
    settings: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Construye la serie objetivo de régimen en dos pasos:
    1. Calcula los umbrales usando la serie de entrenamiento.
    2. Aplica las etiquetas a la serie fuente (puede ser la misma o diferente).
    Retorna la serie con etiquetas y los metadatos de umbrales.
    """
    thresholds_metadata = compute_quantile_thresholds(
        train_source_series=train_source_series,
        settings=settings,
    )

    regime_series = apply_regime_labels(
        source_series=source_series,
        thresholds_metadata=thresholds_metadata,
        settings=settings,
    )

    return regime_series, thresholds_metadata


def attach_regime_target_to_dataframe(
    df: pd.DataFrame,
    regime_series: pd.Series,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Agrega la serie de régimen como una nueva columna al DataFrame.
    Verifica que las longitudes coincidan y devuelve una copia del DataFrame.
    """
    cfg = _classification_cfg(settings)
    output_name = cfg["name"]

    if len(df) != len(regime_series):
        raise ValueError(
            "Length mismatch between dataframe and regime_series."
        )

    out = df.copy()
    out[output_name] = regime_series.values
    return out


def validate_regime_thresholds_metadata(
    thresholds_metadata: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    """
    Valida que los metadatos de umbrales sean completos y consistentes con la configuración.
    Comprueba:
    - Presencia de todas las claves esperadas.
    - Coincidencia de valores con la configuración (nombre, método, etiquetas, etc.).
    - Que q1 y q2 sean numéricos, no NaN, y estén ordenados (q1 <= q2).
    """
    cfg = _classification_cfg(settings)

    required_keys = {
        "source_continuous_target",
        "regime_target_name",
        "method",
        "labels",
        "quantiles",
        "thresholds_source",
        "lower_bin_inclusive",
        "upper_bin_inclusive",
        "allow_missing_source",
        "q1",
        "q2",
    }

    missing = required_keys - set(thresholds_metadata.keys())
    if missing:
        raise ValueError(
            f"Threshold metadata is missing required keys: {sorted(missing)}"
        )

    # Comparaciones con la configuración original
    if thresholds_metadata["regime_target_name"] != cfg["name"]:
        raise ValueError(
            "Threshold metadata `regime_target_name` does not match settings."
        )

    if thresholds_metadata["source_continuous_target"] != cfg["source_continuous_target"]:
        raise ValueError(
            "Threshold metadata `source_continuous_target` does not match settings."
        )

    if thresholds_metadata["method"] != cfg["method"]:
        raise ValueError("Threshold metadata `method` does not match settings.")

    if list(thresholds_metadata["labels"]) != list(cfg["labels"]):
        raise ValueError("Threshold metadata `labels` do not match settings.")

    if list(thresholds_metadata["quantiles"]) != list(cfg["quantiles"]):
        raise ValueError("Threshold metadata `quantiles` do not match settings.")

    if thresholds_metadata["thresholds_source"] != cfg["thresholds_source"]:
        raise ValueError(
            "Threshold metadata `thresholds_source` does not match settings."
        )

    if thresholds_metadata["lower_bin_inclusive"] != cfg["lower_bin_inclusive"]:
        raise ValueError(
            "Threshold metadata `lower_bin_inclusive` does not match settings."
        )

    if thresholds_metadata["upper_bin_inclusive"] != cfg["upper_bin_inclusive"]:
        raise ValueError(
            "Threshold metadata `upper_bin_inclusive` does not match settings."
        )

    if thresholds_metadata["allow_missing_source"] != cfg["allow_missing_source"]:
        raise ValueError(
            "Threshold metadata `allow_missing_source` does not match settings."
        )

    q1 = thresholds_metadata["q1"]
    q2 = thresholds_metadata["q2"]

    if not isinstance(q1, (int, float)) or not isinstance(q2, (int, float)):
        raise ValueError("Threshold metadata `q1` and `q2` must be numeric.")

    if np.isnan(q1) or np.isnan(q2):
        raise ValueError("Threshold metadata `q1` and `q2` cannot be NaN.")

    if q1 > q2:
        raise ValueError(
            f"Threshold metadata is not ordered: q1={q1}, q2={q2}."
        )


def validate_regime_output_series(
    regime_series: pd.Series,
    settings: dict[str, Any],
) -> None:
    """
    Verifica que la serie de régimen generada contenga únicamente las etiquetas
    permitidas según la configuración (ignorando valores nulos).
    """
    cfg = _classification_cfg(settings)
    allowed_labels = set(cfg["labels"])

    observed = set(regime_series.dropna().unique().tolist())
    invalid_labels = observed - allowed_labels

    if invalid_labels:
        raise ValueError(
            f"Regime output contains invalid labels: {sorted(invalid_labels)}"
        )