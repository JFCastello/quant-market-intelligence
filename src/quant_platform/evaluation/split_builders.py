from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.tseries.offsets import DateOffset

from quant_platform.schemas import SplitRecord


@dataclass(frozen=True)
class FoldBoundaries:
    """
    Contenedor inmutable que define los límites de fecha de un pliegue (fold)
    en una estrategia de walk‑forward.
    """
    split_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _splits_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """Extrae la sección 'splits' de la configuración. Lanza KeyError si no existe."""
    if "splits" not in settings:
        raise KeyError("Missing `splits` section in settings.")
    return settings["splits"]


def validate_split_source_input(df: pd.DataFrame, settings: dict[str, Any]) -> None:
    """
    Valida que el DataFrame de entrada para generar splits sea correcto:
    - Contiene las columnas obligatorias (group_key, date_column).
    - No está vacío.
    - No tiene valores nulos en las columnas clave.
    - La columna de fechas es parseable.
    - No hay filas duplicadas por (instrumento, fecha).
    """
    cfg = _splits_cfg(settings)

    date_col = cfg["date_column"]
    group_key = cfg["group_key"]

    required_cols = {group_key, date_col}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Split source input is missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Split source input dataframe is empty.")

    if df[group_key].isna().any():
        raise ValueError(f"Split source input contains null `{group_key}` values.")

    if df[date_col].isna().any():
        raise ValueError(f"Split source input contains null `{date_col}` values.")

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"Split source input contains unparseable `{date_col}` values.")

    duplicated = df.duplicated(subset=[group_key, date_col])
    if duplicated.any():
        raise ValueError(
            f"Split source input contains duplicate `{group_key}`/`{date_col}` rows."
        )


def _prepare_split_source(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """
    Prepara el DataFrame de entrada:
    - Convierte la columna de fechas a datetime.
    - Ordena por instrumento y fecha.
    - Reinicia el índice.
    """
    cfg = _splits_cfg(settings)
    date_col = cfg["date_column"]
    group_key = cfg["group_key"]

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    out = out.sort_values([group_key, date_col]).reset_index(drop=True)
    return out


def build_fold_boundaries_for_instrument(
    instrument_df: pd.DataFrame,
    settings: dict[str, Any],
) -> list[FoldBoundaries]:
    """
    Construye la lista de pliegues (FoldBoundaries) para un único instrumento.
    Soporta únicamente el método 'walk_forward_expanding'.
    Genera ventanas de entrenamiento, validación (opcional) y prueba con desplazamiento (step).
    """
    cfg = _splits_cfg(settings)

    if cfg["method"] != "walk_forward_expanding":
        raise NotImplementedError(f"Unsupported split method: {cfg['method']}")

    date_col = cfg["date_column"]
    require_validation = cfg["require_validation"]
    allow_partial_last_fold = cfg["allow_partial_last_fold"]

    train_years = cfg["train_years"]
    validation_months = cfg["validation_months"]
    test_months = cfg["test_months"]
    step_months = cfg["step_months"]

    # Obtener fechas únicas y ordenadas del instrumento
    dates = pd.Series(pd.to_datetime(instrument_df[date_col], errors="raise")).sort_values()
    dates = dates.drop_duplicates().reset_index(drop=True)

    if dates.empty:
        return []

    dataset_start = dates.iloc[0]
    dataset_end = dates.iloc[-1]

    train_start = dataset_start
    train_end = train_start + DateOffset(years=train_years) - DateOffset(days=1)

    folds: list[FoldBoundaries] = []
    split_idx = 1

    while True:
        # Definir ventana de validación si está habilitada
        validation_start = train_end + DateOffset(days=1) if require_validation else None
        validation_end = (
            validation_start + DateOffset(months=validation_months) - DateOffset(days=1)
            if require_validation and validation_start is not None
            else None
        )

        # Definir ventana de prueba (justo después de validación o entrenamiento)
        test_start = (
            validation_end + DateOffset(days=1)
            if require_validation and validation_end is not None
            else train_end + DateOffset(days=1)
        )
        test_end = test_start + DateOffset(months=test_months) - DateOffset(days=1)

        # Si no se permiten pliegues parciales y la prueba excede el dataset, terminar
        if not allow_partial_last_fold and test_end > dataset_end:
            break

        fold = FoldBoundaries(
            split_id=f"split_{split_idx:03d}",
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=min(test_end, dataset_end),
        )
        folds.append(fold)

        split_idx += 1
        train_end = train_end + DateOffset(months=step_months)

        if train_end >= dataset_end:
            break

    return folds


def materialize_split_records(
    instrument_df: pd.DataFrame,
    fold_boundaries: list[FoldBoundaries],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Convierte cada FoldBoundaries en un registro SplitRecord (modelo Pydantic) y
    devuelve un DataFrame con todos los splits para un instrumento.
    Calcula el número de filas reales dentro de cada ventana.
    """
    cfg = _splits_cfg(settings)
    date_col = cfg["date_column"]
    group_key = cfg["group_key"]

    # Verificar que el DataFrame corresponda a un solo instrumento
    instrument_ids = instrument_df[group_key].dropna().unique().tolist()
    if len(instrument_ids) != 1:
        raise ValueError(
            "materialize_split_records expects data for exactly one instrument."
        )

    instrument_id = instrument_ids[0]
    dates = pd.to_datetime(instrument_df[date_col], errors="raise")

    rows: list[dict[str, Any]] = []

    for fold in fold_boundaries:
        # Máscaras para cada conjunto
        train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
        validation_mask = (
            (dates >= fold.validation_start) & (dates <= fold.validation_end)
            if fold.validation_start is not None and fold.validation_end is not None
            else pd.Series(False, index=instrument_df.index)
        )
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)

        train_rows = int(train_mask.sum())
        validation_rows = int(validation_mask.sum())
        test_rows = int(test_mask.sum())

        record = SplitRecord(
            split_version=cfg["split_version"],
            split_id=fold.split_id,
            instrument_id=instrument_id,
            train_start=fold.train_start.date(),
            train_end=fold.train_end.date(),
            validation_start=fold.validation_start.date()
            if fold.validation_start is not None
            else None,
            validation_end=fold.validation_end.date()
            if fold.validation_end is not None
            else None,
            test_start=fold.test_start.date(),
            test_end=fold.test_end.date(),
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            regime_thresholds_source=cfg["regime_thresholds_source"],
        ).model_dump()

        rows.append(record)

    split_df = pd.DataFrame(rows)
    # Convertir columnas de fecha a datetime para consistencia
    if not split_df.empty:
        for col in [
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ]:
            split_df[col] = pd.to_datetime(split_df[col], errors="coerce")

    return split_df


def build_walk_forward_splits(
    df: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Orquestador principal: genera splits walk‑forward para todos los instrumentos.
    - Valida la entrada.
    - Prepara los datos.
    - Para cada instrumento, construye pliegues y los materializa.
    - Filtra según min_train_observations y require_validation.
    - Concatena todos los splits en un solo DataFrame.
    """
    validate_split_source_input(df, settings)
    prepared = _prepare_split_source(df, settings)

    cfg = _splits_cfg(settings)
    group_key = cfg["group_key"]
    min_train_observations = cfg["min_train_observations"]
    require_validation = cfg["require_validation"]

    all_split_frames: list[pd.DataFrame] = []

    for instrument_id, instrument_df in prepared.groupby(group_key, sort=True):
        folds = build_fold_boundaries_for_instrument(
            instrument_df=instrument_df,
            settings=settings,
        )

        split_df = materialize_split_records(
            instrument_df=instrument_df,
            fold_boundaries=folds,
            settings=settings,
        )

        if split_df.empty:
            continue

        # Aplicar filtros de calidad
        split_df = split_df.loc[split_df["train_rows"] >= min_train_observations].copy()

        if require_validation:
            split_df = split_df.loc[split_df["validation_rows"].fillna(0) > 0].copy()

        split_df = split_df.loc[split_df["test_rows"] > 0].copy()

        if not split_df.empty:
            all_split_frames.append(split_df.reset_index(drop=True))

    # Si no hay splits, devolver un DataFrame vacío con la estructura esperada
    if not all_split_frames:
        return pd.DataFrame(
            columns=[
                "split_version",
                "split_id",
                "instrument_id",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
                "train_rows",
                "validation_rows",
                "test_rows",
                "regime_thresholds_source",
            ]
        )

    out = pd.concat(all_split_frames, ignore_index=True)
    out = out.sort_values(["instrument_id", "train_start", "test_start"]).reset_index(drop=True)
    return out


def validate_split_output(split_df: pd.DataFrame, settings: dict[str, Any]) -> None:
    """
    Valida que el DataFrame resultante de los splits cumpla con todas las reglas:
    - Columnas esperadas presentes.
    - Sin valores nulos en columnas críticas.
    - Coherencia de versiones y fuentes de umbrales.
    - Fechas válidas y correctamente ordenadas (train < validation < test).
    - Recuentos de filas positivos.
    """
    cfg = _splits_cfg(settings)
    require_validation = cfg["require_validation"]

    expected_cols = [
        "split_version",
        "split_id",
        "instrument_id",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "train_rows",
        "validation_rows",
        "test_rows",
        "regime_thresholds_source",
    ]

    missing_cols = [col for col in expected_cols if col not in split_df.columns]
    if missing_cols:
        raise ValueError(f"Split output is missing expected columns: {missing_cols}")

    if split_df.empty:
        raise ValueError("Split output dataframe is empty.")

    if split_df["split_version"].isna().any():
        raise ValueError("Split output contains null `split_version` values.")

    if split_df["split_id"].isna().any():
        raise ValueError("Split output contains null `split_id` values.")

    if split_df["instrument_id"].isna().any():
        raise ValueError("Split output contains null `instrument_id` values.")

    expected_version = cfg["split_version"]
    if not (split_df["split_version"] == expected_version).all():
        raise ValueError(
            f"Split output contains split_version different from `{expected_version}`."
        )

    if not (split_df["regime_thresholds_source"] == cfg["regime_thresholds_source"]).all():
        raise ValueError("Split output contains invalid `regime_thresholds_source` values.")

    # Validar parseo de fechas
    for col in [
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ]:
        parsed = pd.to_datetime(split_df[col], errors="coerce")
        if col in {"validation_start", "validation_end"} and not require_validation:
            continue
        if parsed.isna().any():
            raise ValueError(f"Split output contains invalid dates in `{col}`.")

    # Validar relaciones temporales y recuentos por cada fila
    for row in split_df.itertuples(index=False):
        if row.train_start > row.train_end:
            raise ValueError(f"Invalid train interval in {row.split_id} / {row.instrument_id}.")

        if require_validation:
            if row.validation_start is None or row.validation_end is None:
                raise ValueError(
                    f"Missing validation interval in {row.split_id} / {row.instrument_id}."
                )
            if row.train_end >= row.validation_start:
                raise ValueError(
                    f"Train must end before validation starts in {row.split_id} / {row.instrument_id}."
                )
            if row.validation_start > row.validation_end:
                raise ValueError(
                    f"Invalid validation interval in {row.split_id} / {row.instrument_id}."
                )
            if row.validation_end >= row.test_start:
                raise ValueError(
                    f"Validation must end before test starts in {row.split_id} / {row.instrument_id}."
                )
        else:
            if row.train_end >= row.test_start:
                raise ValueError(
                    f"Train must end before test starts in {row.split_id} / {row.instrument_id}."
                )

        if row.test_start > row.test_end:
            raise ValueError(f"Invalid test interval in {row.split_id} / {row.instrument_id}.")

        if row.train_rows <= 0:
            raise ValueError(f"Non-positive train_rows in {row.split_id} / {row.instrument_id}.")
        if require_validation and (row.validation_rows is None or row.validation_rows <= 0):
            raise ValueError(
                f"Non-positive validation_rows in {row.split_id} / {row.instrument_id}."
            )
        if row.test_rows <= 0:
            raise ValueError(f"Non-positive test_rows in {row.split_id} / {row.instrument_id}.")