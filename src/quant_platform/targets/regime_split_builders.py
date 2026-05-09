from __future__ import annotations

from typing import Any

import pandas as pd

from quant_platform.targets.regime_builders import (
    apply_regime_labels,
    compute_quantile_thresholds,
    validate_regime_output_series,
)


def _targets_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """Extrae la sección 'targets' de la configuración. Lanza KeyError si no existe."""
    if "targets" not in settings:
        raise KeyError("Missing `targets` section in settings.")
    return settings["targets"]


def _classification_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """Extrae la subsección 'classification_target' dentro de 'targets'."""
    targets_cfg = _targets_cfg(settings)
    if "classification_target" not in targets_cfg:
        raise KeyError("Missing `classification_target` section in `targets`.")
    return targets_cfg["classification_target"]


def _splits_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """Extrae la sección 'splits' de la configuración."""
    if "splits" not in settings:
        raise KeyError("Missing `splits` section in settings.")
    return settings["splits"]


def get_regime_split_output_columns(settings: dict[str, Any]) -> list[str]:
    """
    Devuelve la lista de columnas que debe tener el DataFrame de salida
    para los targets de régimen por split.
    """
    classification_cfg = _classification_cfg(settings)
    regime_name = classification_cfg["name"]
    continuous_name = classification_cfg["source_continuous_target"]

    return [
        "split_id",
        "split_version",
        "instrument_id",
        "date",
        "dataset_role",
        "target_version",
        continuous_name,
        regime_name,
        "threshold_low",
        "threshold_high",
        "regime_thresholds_source",
    ]


def validate_regime_split_inputs(
    target_df: pd.DataFrame,
    split_df: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    """
    Valida que los DataFrames de entrada (target continuo y splits) tengan
    las columnas necesarias, no estén vacíos y contengan fechas e identificadores válidos.
    """
    classification_cfg = _classification_cfg(settings)
    continuous_name = classification_cfg["source_continuous_target"]

    # Verificar columnas requeridas en target_df
    target_required = {"instrument_id", "date", "target_version", continuous_name}
    missing_target = sorted(target_required - set(target_df.columns))
    if missing_target:
        raise ValueError(
            f"Target dataframe is missing required columns: {missing_target}"
        )

    # Verificar columnas requeridas en split_df
    split_required = {
        "split_id",
        "split_version",
        "instrument_id",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "regime_thresholds_source",
    }
    missing_split = sorted(split_required - set(split_df.columns))
    if missing_split:
        raise ValueError(
            f"Split dataframe is missing required columns: {missing_split}"
        )

    if target_df.empty:
        raise ValueError("Target dataframe is empty.")

    if split_df.empty:
        raise ValueError("Split dataframe is empty.")

    # Validar nulos en identificadores
    if target_df["instrument_id"].isna().any():
        raise ValueError("Target dataframe contains null `instrument_id` values.")

    if split_df["instrument_id"].isna().any():
        raise ValueError("Split dataframe contains null `instrument_id` values.")

    # Validar fechas en target_df
    if target_df["date"].isna().any():
        raise ValueError("Target dataframe contains null `date` values.")

    date_parsed = pd.to_datetime(target_df["date"], errors="coerce")
    if date_parsed.isna().any():
        raise ValueError("Target dataframe contains unparseable `date` values.")

    # Validar fechas en split_df (validation_start/end pueden ser nulos)
    for col in [
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ]:
        parsed = pd.to_datetime(split_df[col], errors="coerce")
        if col in {"validation_start", "validation_end"}:
            continue  # pueden ser nulos si no se requiere validación
        if parsed.isna().any():
            raise ValueError(f"Split dataframe contains invalid `{col}` values.")


def _prepare_target_df(target_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame de targets: convierte fechas a datetime y ordena por instrumento y fecha."""
    out = target_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return out


def _prepare_split_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame de splits: convierte columnas de fechas a datetime y ordena."""
    out = split_df.copy()

    for col in [
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out = out.sort_values(["instrument_id", "train_start", "test_start"]).reset_index(
        drop=True
    )
    return out


def build_dataset_role_series_for_split(
    dates: pd.Series,
    split_row: pd.Series,
    require_validation: bool,
) -> pd.Series:
    """
    Construye una serie con el rol ('train', 'validation', 'test') para cada fecha
    dentro de un split específico, según las ventanas definidas en split_row.
    """
    roles = pd.Series(pd.NA, index=dates.index, dtype="object")

    # Asignar entrenamiento
    train_mask = (dates >= split_row["train_start"]) & (dates <= split_row["train_end"])
    roles.loc[train_mask] = "train"

    # Asignar validación solo si está requerida y las fechas existen
    if require_validation:
        validation_mask = (
            (dates >= split_row["validation_start"])
            & (dates <= split_row["validation_end"])
        )
        roles.loc[validation_mask] = "validation"

    # Asignar prueba
    test_mask = (dates >= split_row["test_start"]) & (dates <= split_row["test_end"])
    roles.loc[test_mask] = "test"

    return roles


def build_regime_targets_for_single_split(
    instrument_target_df: pd.DataFrame,
    split_row: pd.Series,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Construye los targets de régimen para un instrumento y un split específicos:
    1. Determina el rol (train/validation/test) de cada fecha.
    2. Estima umbrales usando solo el conjunto de entrenamiento.
    3. Aplica etiquetas a todas las filas del split.
    4. Retorna un DataFrame con los resultados y los metadatos de umbrales.
    """
    classification_cfg = _classification_cfg(settings)
    splits_cfg = _splits_cfg(settings)

    continuous_name = classification_cfg["source_continuous_target"]
    regime_name = classification_cfg["name"]
    require_validation = splits_cfg["require_validation"]

    # Preparar datos del instrumento
    instrument_df = instrument_target_df.copy()
    instrument_df["date"] = pd.to_datetime(instrument_df["date"], errors="raise")
    instrument_df = instrument_df.sort_values("date").reset_index(drop=True)

    # Asignar roles (train/validation/test)
    roles = build_dataset_role_series_for_split(
        dates=instrument_df["date"],
        split_row=split_row,
        require_validation=require_validation,
    )

    # Filtrar solo las filas que pertenecen a este split (las que tienen rol)
    fold_df = instrument_df.loc[roles.notna()].copy()
    fold_df["dataset_role"] = roles.loc[roles.notna()].values

    if fold_df.empty:
        raise ValueError(
            f"Split `{split_row['split_id']}` produced no rows for instrument "
            f"`{split_row['instrument_id']}`."
        )

    # Extraer serie de entrenamiento para calcular umbrales
    train_source = fold_df.loc[
        fold_df["dataset_role"] == "train",
        continuous_name,
    ]

    if train_source.empty:
        raise ValueError(
            f"Split `{split_row['split_id']}` has no train rows for threshold estimation."
        )

    # Calcular umbrales (cuantiles) usando solo entrenamiento
    thresholds_metadata = compute_quantile_thresholds(
        train_source_series=train_source,
        settings=settings,
    )

    # Aplicar etiquetas a todas las filas del split (train, validation, test)
    regime_series = apply_regime_labels(
        source_series=fold_df[continuous_name],
        thresholds_metadata=thresholds_metadata,
        settings=settings,
    )

    validate_regime_output_series(
        regime_series=regime_series,
        settings=settings,
    )

    # Construir DataFrame de salida con las columnas esperadas
    out = fold_df[
        ["instrument_id", "date", "target_version", continuous_name, "dataset_role"]
    ].copy()

    out.insert(0, "split_id", split_row["split_id"])
    out.insert(1, "split_version", split_row["split_version"])
    out[regime_name] = regime_series.values
    out["threshold_low"] = thresholds_metadata["q1"]
    out["threshold_high"] = thresholds_metadata["q2"]
    out["regime_thresholds_source"] = split_row["regime_thresholds_source"]

    out = out[get_regime_split_output_columns(settings)]

    return out.reset_index(drop=True), thresholds_metadata


def build_regime_targets_by_split(
    target_df: pd.DataFrame,
    split_df: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Orquestador principal: construye los targets de régimen para todos los splits
    y todos los instrumentos.
    - Valida entradas.
    - Prepara DataFrames.
    - Para cada instrumento y cada split, llama a build_regime_targets_for_single_split.
    - Concatena los resultados y los ordena.
    """
    validate_regime_split_inputs(
        target_df=target_df,
        split_df=split_df,
        settings=settings,
    )

    prepared_targets = _prepare_target_df(target_df)
    prepared_splits = _prepare_split_df(split_df)

    outputs: list[pd.DataFrame] = []

    # Procesar cada instrumento por separado
    for instrument_id, instrument_splits in prepared_splits.groupby(
        "instrument_id",
        sort=True,
    ):
        instrument_targets = prepared_targets.loc[
            prepared_targets["instrument_id"] == instrument_id
        ].copy()

        if instrument_targets.empty:
            raise ValueError(
                f"No target rows found for instrument `{instrument_id}`."
            )

        # Procesar cada split del instrumento
        for _, split_row in instrument_splits.iterrows():
            split_output_df, _ = build_regime_targets_for_single_split(
                instrument_target_df=instrument_targets,
                split_row=split_row,
                settings=settings,
            )
            outputs.append(split_output_df)

    if not outputs:
        return pd.DataFrame(columns=get_regime_split_output_columns(settings))

    out = pd.concat(outputs, ignore_index=True)
    out = out.sort_values(["instrument_id", "split_id", "date"]).reset_index(drop=True)
    return out


def validate_regime_split_output(
    regime_split_df: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    """
    Valida el DataFrame resultante de build_regime_targets_by_split:
    - Columnas esperadas.
    - Sin valores nulos en columnas clave.
    - Fechas correctas.
    - Roles permitidos (train/validation/test).
    - Coherencia de umbrales (threshold_low <= threshold_high).
    - Etiquetas de régimen válidas.
    - Sin duplicados por (split_id, instrument_id, date).
    """
    classification_cfg = _classification_cfg(settings)
    splits_cfg = _splits_cfg(settings)

    regime_name = classification_cfg["name"]
    expected_cols = get_regime_split_output_columns(settings)

    missing_cols = [col for col in expected_cols if col not in regime_split_df.columns]
    if missing_cols:
        raise ValueError(
            f"Regime split output is missing expected columns: {missing_cols}"
        )

    if regime_split_df.empty:
        raise ValueError("Regime split output dataframe is empty.")

    # Verificar nulos en columnas obligatorias
    if regime_split_df["split_id"].isna().any():
        raise ValueError("Regime split output contains null `split_id` values.")

    if regime_split_df["instrument_id"].isna().any():
        raise ValueError("Regime split output contains null `instrument_id` values.")

    if regime_split_df["date"].isna().any():
        raise ValueError("Regime split output contains null `date` values.")

    parsed_dates = pd.to_datetime(regime_split_df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Regime split output contains invalid `date` values.")

    # Validar roles
    allowed_roles = {"train", "validation", "test"}
    observed_roles = set(regime_split_df["dataset_role"].dropna().unique().tolist())
    invalid_roles = observed_roles - allowed_roles
    if invalid_roles:
        raise ValueError(
            f"Regime split output contains invalid dataset_role values: {sorted(invalid_roles)}"
        )

    if splits_cfg["require_validation"]:
        if "validation" not in observed_roles:
            raise ValueError(
                "Regime split output is missing `validation` rows while validation is required."
            )

    # Validar fuente de umbrales
    expected_source = splits_cfg["regime_thresholds_source"]
    if not (regime_split_df["regime_thresholds_source"] == expected_source).all():
        raise ValueError(
            "Regime split output contains invalid `regime_thresholds_source` values."
        )

    # Validar umbrales
    if regime_split_df["threshold_low"].isna().any():
        raise ValueError("Regime split output contains null `threshold_low` values.")

    if regime_split_df["threshold_high"].isna().any():
        raise ValueError("Regime split output contains null `threshold_high` values.")

    bad_threshold_order = regime_split_df["threshold_low"] > regime_split_df["threshold_high"]
    if bad_threshold_order.any():
        raise ValueError(
            "Regime split output contains rows with threshold_low > threshold_high."
        )

    # Validar etiquetas de régimen
    validate_regime_output_series(
        regime_series=regime_split_df[regime_name],
        settings=settings,
    )

    # Validar unicidad
    if regime_split_df.duplicated(subset=["split_id", "instrument_id", "date"]).any():
        raise ValueError(
            "Regime split output contains duplicate split_id/instrument_id/date rows."
        )