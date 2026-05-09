from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from xgboost import XGBRegressor


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Verifica que el DataFrame contenga todas las columnas requeridas. Lanza error si falta alguna."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def _validate_ml_regressor_settings(settings: Mapping[str, Any]) -> None:
    """Valida que la configuración del regresor ML contenga todas las claves necesarias."""
    required_keys = {
        "model_name",
        "model_version",
        "feature_source",
        "target_column",
        "score_roles",
        "objective",
        "eval_metric",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "random_state",
        "tree_method",
        "n_jobs",
        "early_stopping_rounds",
        "persist_models",
        "persist_feature_columns",
        "allow_native_missing_values",
    }
    missing = required_keys - set(settings.keys())
    if missing:
        raise ValueError(f"ML regressor settings missing keys: {sorted(missing)}")


def _resolve_first_existing_column(
    columns: set[str],
    candidates: list[str],
    logical_name: str,
) -> str:
    """
    Busca la primera columna de una lista de candidatos que exista en el conjunto de columnas.
    Útil para manejar variaciones en los nombres de columnas entre versiones de splits.
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not resolve split column for '{logical_name}'. "
        f"Candidates tried: {candidates}. "
        f"Available columns: {sorted(columns)}"
    )


def _expand_interval_split_records_to_daily_roles(
    split_df: pd.DataFrame,
    available_dates: pd.Series,
) -> pd.DataFrame:
    """
    Expande registros de splits definidos por intervalos (train_start, train_end, etc.)
    en un DataFrame con una fila por fecha y su rol (train/validation/test).
    Utiliza un calendario de fechas disponibles y asigna roles según los intervalos.
    """
    _require_columns(split_df, {"split_id"}, "split_df")

    columns = set(split_df.columns)

    # Resolver nombres de columnas flexibles
    train_start_col = _resolve_first_existing_column(
        columns,
        ["train_start_date", "train_start", "train_date_start"],
        "train_start",
    )
    train_end_col = _resolve_first_existing_column(
        columns,
        ["train_end_date", "train_end", "train_date_end"],
        "train_end",
    )
    validation_start_col = _resolve_first_existing_column(
        columns,
        ["validation_start_date", "validation_start", "val_start_date", "val_start"],
        "validation_start",
    )
    validation_end_col = _resolve_first_existing_column(
        columns,
        ["validation_end_date", "validation_end", "val_end_date", "val_end"],
        "validation_end",
    )
    test_start_col = _resolve_first_existing_column(
        columns,
        ["test_start_date", "test_start"],
        "test_start",
    )
    test_end_col = _resolve_first_existing_column(
        columns,
        ["test_end_date", "test_end"],
        "test_end",
    )

    # Crear calendario base con todas las fechas disponibles
    calendar = pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(available_dates, errors="raise"))
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        }
    )

    expanded_parts: list[pd.DataFrame] = []

    for row in split_df.itertuples(index=False):
        split_id = getattr(row, "split_id")

        train_start = pd.to_datetime(getattr(row, train_start_col), errors="raise")
        train_end = pd.to_datetime(getattr(row, train_end_col), errors="raise")
        validation_start = pd.to_datetime(getattr(row, validation_start_col), errors="raise")
        validation_end = pd.to_datetime(getattr(row, validation_end_col), errors="raise")
        test_start = pd.to_datetime(getattr(row, test_start_col), errors="raise")
        test_end = pd.to_datetime(getattr(row, test_end_col), errors="raise")

        split_calendar = calendar.copy()
        split_calendar["split_id"] = split_id
        split_calendar["dataset_role"] = pd.NA

        # Asignar roles según intervalos
        train_mask = split_calendar["date"].between(train_start, train_end, inclusive="both")
        validation_mask = split_calendar["date"].between(validation_start, validation_end, inclusive="both")
        test_mask = split_calendar["date"].between(test_start, test_end, inclusive="both")

        split_calendar.loc[train_mask, "dataset_role"] = "train"
        split_calendar.loc[validation_mask, "dataset_role"] = "validation"
        split_calendar.loc[test_mask, "dataset_role"] = "test"

        # Filtrar solo fechas que pertenecen a algún rol
        split_calendar = split_calendar.loc[
            split_calendar["dataset_role"].notna(),
            ["date", "split_id", "dataset_role"],
        ].copy()

        expanded_parts.append(split_calendar)

    if not expanded_parts:
        return pd.DataFrame(columns=["date", "split_id", "dataset_role"])

    expanded_df = pd.concat(expanded_parts, ignore_index=True)
    expanded_df["date"] = pd.to_datetime(expanded_df["date"], errors="raise")

    return expanded_df.sort_values(["split_id", "date"]).reset_index(drop=True)


def _normalize_split_df(
    split_df: pd.DataFrame,
    available_dates: pd.Series,
) -> pd.DataFrame:
    """
    Normaliza el DataFrame de splits a un formato estándar con columnas:
    date, split_id, dataset_role.
    Si ya tiene el formato diario, lo usa; si no, expande intervalos.
    """
    daily_required = {"date", "split_id", "dataset_role"}

    if daily_required.issubset(split_df.columns):
        out = split_df.loc[:, ["date", "split_id", "dataset_role"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        return out.sort_values(["split_id", "date"]).reset_index(drop=True)

    return _expand_interval_split_records_to_daily_roles(
        split_df=split_df,
        available_dates=available_dates,
    )


def build_ml_regressor_input_panel(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Combina features y targets en un panel unificado para modelado.
    - Valida columnas requeridas.
    - Fusiona por instrument_id y date.
    - Identifica automáticamente las columnas de features (todas excepto metadatos).
    Retorna:
      - panel_df: DataFrame con features y target unidos
      - feature_columns: lista de nombres de columnas de features
    """
    _validate_ml_regressor_settings(settings)

    target_col = settings["target_column"]

    _require_columns(features_df, {"instrument_id", "date"}, "features_df")
    _require_columns(targets_df, {"instrument_id", "date", target_col}, "targets_df")

    features = features_df.copy()
    targets = targets_df.copy()

    features["date"] = pd.to_datetime(features["date"], errors="raise")
    targets["date"] = pd.to_datetime(targets["date"], errors="raise")

    panel_df = features.merge(
        targets[["instrument_id", "date", target_col]],
        on=["instrument_id", "date"],
        how="inner",
        validate="one_to_one",
    )

    metadata_cols = {"instrument_id", "date", "feature_version", target_col}
    feature_columns = [col for col in panel_df.columns if col not in metadata_cols]

    if not feature_columns:
        raise ValueError("No feature columns were found for ml_regressor input panel")

    panel_df = panel_df.sort_values("date").reset_index(drop=True)
    return panel_df, feature_columns


def build_xgboost_regressor_forecasts_by_split(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    split_df: pd.DataFrame,
    settings: Mapping[str, Any],
    symbol: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Entrena un modelo XGBoost Regressor por cada split temporal.
    - Para cada split: entrena con train, valida con validation, predice sobre validation/test.
    - Retorna:
        - forecast_df: DataFrame con predicciones OOS (yhat_future_rv_5d) y targets reales.
        - split_models: lista de diccionarios con los modelos entrenados y metadatos para persistencia.
    """
    _validate_ml_regressor_settings(settings)

    # Construir panel de features + target
    panel_df, feature_columns = build_ml_regressor_input_panel(
        features_df=features_df,
        targets_df=targets_df,
        settings=settings,
    )

    # Normalizar splits (formato diario con roles)
    split_df_prepared = _normalize_split_df(
        split_df=split_df,
        available_dates=panel_df["date"],
    )

    # Unir panel con roles de split
    merged = panel_df.merge(
        split_df_prepared,
        on="date",
        how="inner",
        validate="one_to_many",
    )

    merged["date"] = pd.to_datetime(merged["date"], errors="raise")
    merged = merged.sort_values(["split_id", "date"]).reset_index(drop=True)

    target_col = settings["target_column"]
    score_roles = set(settings["score_roles"])

    # Determinar símbolo (si no se pasa, se extrae de instrument_id)
    symbol_value = symbol
    if symbol_value is None and "instrument_id" in merged.columns and merged["instrument_id"].notna().any():
        symbol_value = str(merged["instrument_id"].dropna().iloc[0])

    forecast_records: list[dict[str, Any]] = []
    split_models: list[dict[str, Any]] = []

    # Procesar cada split independientemente
    for split_id, split_slice in merged.groupby("split_id", sort=True):
        split_slice = split_slice.sort_values("date").reset_index(drop=True)

        train_df = split_slice.loc[split_slice["dataset_role"] == "train"].copy()
        validation_df = split_slice.loc[split_slice["dataset_role"] == "validation"].copy()
        score_df = split_slice.loc[split_slice["dataset_role"].isin(score_roles)].copy()

        # Eliminar filas con target nulo
        train_df = train_df.loc[train_df[target_col].notna()].copy()
        validation_df = validation_df.loc[validation_df[target_col].notna()].copy()
        score_df = score_df.loc[score_df[target_col].notna()].copy()

        if train_df.empty:
            raise ValueError(f"Split {split_id} has no non-null train target rows.")
        if validation_df.empty:
            raise ValueError(f"Split {split_id} has no non-null validation target rows.")

        x_train = train_df[feature_columns]
        y_train = train_df[target_col]

        x_validation = validation_df[feature_columns]
        y_validation = validation_df[target_col]

        # Configurar y entrenar XGBoost
        model = XGBRegressor(
            objective=settings["objective"],
            eval_metric=settings["eval_metric"],
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            gamma=float(settings["gamma"]),
            random_state=int(settings["random_state"]),
            tree_method=settings["tree_method"],
            n_jobs=int(settings["n_jobs"]),
            early_stopping_rounds=int(settings["early_stopping_rounds"]),
        )

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )

        best_iteration = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)

        train_start_date = pd.to_datetime(train_df["date"].min())
        train_end_date = pd.to_datetime(train_df["date"].max())

        # Predecir sobre validation y test (score_roles)
        x_score = score_df[feature_columns]
        yhat_score = model.predict(x_score)

        score_df = score_df.copy()
        score_df["yhat_future_rv_5d"] = yhat_score

        # Guardar registros de forecast
        for row in score_df.itertuples(index=False):
            forecast_records.append(
                {
                    "symbol": symbol_value,
                    "date": pd.to_datetime(row.date),
                    "split_id": row.split_id,
                    "dataset_role": row.dataset_role,
                    "model_name": settings["model_name"],
                    "model_version": settings["model_version"],
                    "target_column": target_col,
                    "yhat_future_rv_5d": float(row.yhat_future_rv_5d),
                    "future_rv_5d": float(getattr(row, target_col)),
                    "train_start_date": train_start_date,
                    "train_end_date": train_end_date,
                    "n_train": int(len(train_df)),
                    "n_validation": int(len(validation_df)),
                    "n_score": int(len(score_df)),
                    "feature_count": int(len(feature_columns)),
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                }
            )

        # Almacenar modelo y metadatos para posible persistencia
        split_models.append(
            {
                "split_id": split_id,
                "model": model,
                "feature_columns": feature_columns,
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "n_train": int(len(train_df)),
                "n_validation": int(len(validation_df)),
                "best_iteration": best_iteration,
                "best_score": best_score,
            }
        )

    forecast_df = pd.DataFrame.from_records(forecast_records)

    if forecast_df.empty:
        return forecast_df, split_models

    forecast_df = forecast_df.sort_values(["split_id", "date"]).reset_index(drop=True)
    return forecast_df, split_models