from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.services.settings import load_settings


# Conjunto de columnas obligatorias en el DataFrame de forecasts
REQUIRED_FORECAST_COLUMNS = {
    "symbol",
    "date",
    "split_id",
    "dataset_role",
    "model_name",
    "benchmark_version",
    "forecast_horizon_days",
    "output_target_name",
    "yhat_future_rv_5d",
    "train_start_date",
    "train_end_date",
    "n_train",
    "fit_status",
    "omega",
    "alpha_1",
    "beta_1",
    "nu",
}


def _find_single_parquet(directory: Path) -> Path:
    """
    Busca el primer archivo Parquet dentro de un directorio.
    Si hay múltiples, advierte y usa el primero (orden alfabético).
    Lanza excepción si no encuentra ninguno.
    """
    files = sorted(directory.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {directory}")

    if len(files) > 1:
        print(f"[WARN] Multiple parquet files found in {directory}. Using: {files[0].name}")

    return files[0]


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Verifica que el DataFrame contenga todas las columnas requeridas. Lanza error si falta alguna."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def _validate_symbol_forecast_file(
    symbol: str,
    forecast_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    """
    Valida el archivo de forecasts GARCH para un símbolo:
    - Columnas requeridas presentes.
    - Sin valores nulos en columnas clave.
    - Roles permitidos (validation, test) – no debe incluir 'train'.
    - Sin duplicados (split_id, date).
    - Forecasts finitos y positivos.
    - Estado de convergencia del modelo.
    - Coincidencia de splits entre forecasts y metadatos.
    - Cantidad de filas por split coincide con la esperada.
    - Fechas dentro de las ventanas definidas.
    - Parámetros constantes dentro de cada split.
    """
    _require_columns(forecast_df, REQUIRED_FORECAST_COLUMNS, "forecast_df")

    if forecast_df.empty:
        raise ValueError(f"{symbol}: forecast_df is empty")

    forecast_df = forecast_df.copy()
    split_df = split_df.copy()

    # Convertir columnas de fecha a datetime
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="raise")
    forecast_df["train_start_date"] = pd.to_datetime(forecast_df["train_start_date"], errors="raise")
    forecast_df["train_end_date"] = pd.to_datetime(forecast_df["train_end_date"], errors="raise")

    for col in [
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ]:
        split_df[col] = pd.to_datetime(split_df[col], errors="raise")

    # Validar que el símbolo sea consistente
    if set(forecast_df["symbol"].dropna().unique()) != {symbol}:
        raise ValueError(
            f"{symbol}: forecast_df contains unexpected symbol values: "
            f"{sorted(forecast_df['symbol'].dropna().unique().tolist())}"
        )

    # Los roles deben ser solo 'validation' o 'test' (el entrenamiento no se pronostica)
    allowed_roles = {"validation", "test"}
    actual_roles = set(forecast_df["dataset_role"].dropna().unique())
    if not actual_roles.issubset(allowed_roles):
        raise ValueError(f"{symbol}: dataset_role contains invalid values: {sorted(actual_roles)}")

    if forecast_df["dataset_role"].isna().any():
        raise ValueError(f"{symbol}: dataset_role contains NaN values")

    if forecast_df["split_id"].isna().any():
        raise ValueError(f"{symbol}: split_id contains NaN values")

    # Verificar duplicados por split y fecha
    if forecast_df.duplicated(subset=["split_id", "date"]).any():
        duplicated_rows = forecast_df.loc[forecast_df.duplicated(subset=["split_id", "date"], keep=False)]
        raise ValueError(
            f"{symbol}: duplicated (split_id, date) rows found:\n{duplicated_rows.head().to_string(index=False)}"
        )

    # Validar que los forecasts sean numéricos, finitos y positivos
    if not np.isfinite(forecast_df["yhat_future_rv_5d"]).all():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains non-finite values")

    if (forecast_df["yhat_future_rv_5d"] <= 0).any():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains non-positive values")

    # Validar estado de convergencia
    if forecast_df["fit_status"].isna().any():
        raise ValueError(f"{symbol}: fit_status contains NaN values")

    non_converged = forecast_df.loc[forecast_df["fit_status"] != "converged"]
    if not non_converged.empty:
        raise ValueError(
            f"{symbol}: found non-converged fits:\n"
            f"{non_converged[['split_id', 'fit_status']].drop_duplicates().to_string(index=False)}"
        )

    # Comparar splits presentes en forecasts vs metadatos
    unique_splits_forecast = set(forecast_df["split_id"].unique())
    unique_splits_meta = set(split_df["split_id"].unique())
    if unique_splits_forecast != unique_splits_meta:
        raise ValueError(
            f"{symbol}: split_id mismatch between forecasts and split metadata. "
            f"forecast={sorted(unique_splits_forecast)} meta={sorted(unique_splits_meta)}"
        )

    # Validaciones específicas por split
    for split_row in split_df.itertuples(index=False):
        split_id = split_row.split_id

        split_forecast_df = forecast_df.loc[forecast_df["split_id"] == split_id].copy()
        if split_forecast_df.empty:
            raise ValueError(f"{symbol} {split_id}: no forecast rows found")

        # Verificar cantidad de filas de validación y prueba
        expected_validation_rows = int(split_row.validation_rows)
        expected_test_rows = int(split_row.test_rows)

        actual_validation_rows = int((split_forecast_df["dataset_role"] == "validation").sum())
        actual_test_rows = int((split_forecast_df["dataset_role"] == "test").sum())

        if actual_validation_rows != expected_validation_rows:
            raise ValueError(
                f"{symbol} {split_id}: validation row mismatch. "
                f"expected={expected_validation_rows} actual={actual_validation_rows}"
            )

        if actual_test_rows != expected_test_rows:
            raise ValueError(
                f"{symbol} {split_id}: test row mismatch. "
                f"expected={expected_test_rows} actual={actual_test_rows}"
            )

        # Verificar que las fechas estén dentro de las ventanas definidas
        validation_dates = split_forecast_df.loc[
            split_forecast_df["dataset_role"] == "validation", "date"
        ]
        test_dates = split_forecast_df.loc[
            split_forecast_df["dataset_role"] == "test", "date"
        ]

        if not validation_dates.empty:
            if validation_dates.min() < split_row.validation_start or validation_dates.max() > split_row.validation_end:
                raise ValueError(
                    f"{symbol} {split_id}: validation dates fall outside validation window"
                )

        if not test_dates.empty:
            if test_dates.min() < split_row.test_start or test_dates.max() > split_row.test_end:
                raise ValueError(
                    f"{symbol} {split_id}: test dates fall outside test window"
                )

        # Verificar que los parámetros del modelo sean constantes dentro del split
        for param_col in ["omega", "alpha_1", "beta_1", "nu", "n_train", "train_start_date", "train_end_date"]:
            if split_forecast_df[param_col].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{symbol} {split_id}: parameter/metadata column '{param_col}' is not constant within split"
                )


def main() -> int:
    """
    Orquestador de validación de calidad de los forecasts GARCH.
    - Carga configuración.
    - Para cada símbolo del universo:
      - Localiza el archivo de forecasts y el de splits.
      - Valida el archivo de forecasts contra los splits.
    - Termina con código de salida 0 si todo está bien.
    """
    settings = load_settings()

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]
    evaluations_root = Path(settings["paths"]["evaluations_path"])

    forecast_root = evaluations_root / "benchmark_forecasts"
    splits_root = evaluations_root / "splits"

    print("GARCH BENCHMARK QUALITY CHECKS")

    for symbol in universe:
        symbol_lower = symbol.lower()

        forecast_path = _find_single_parquet(forecast_root / symbol_lower)
        split_path = _find_single_parquet(splits_root / symbol_lower)

        forecast_df = pd.read_parquet(forecast_path)
        split_df = pd.read_parquet(split_path)

        _validate_symbol_forecast_file(
            symbol=symbol,
            forecast_df=forecast_df,
            split_df=split_df,
        )

        print(f"[GARCH BENCHMARK] OK -> {forecast_path}")

    print("GARCH BENCHMARK QUALITY CHECKS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())