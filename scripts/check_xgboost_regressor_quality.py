from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.services.settings import load_settings


# Conjunto de columnas obligatorias en el DataFrame de forecasts del XGBoost
REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "split_id",
    "dataset_role",
    "model_name",
    "model_version",
    "target_column",
    "yhat_future_rv_5d",
    "future_rv_5d",
    "train_start_date",
    "train_end_date",
    "n_train",
    "n_validation",
    "n_score",
    "feature_count",
    "best_iteration",
    "best_score",
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


def _validate_symbol_forecast_file(symbol: str, forecast_df: pd.DataFrame) -> None:
    """
    Valida el archivo de forecasts del XGBoost para un símbolo:
    - Columnas requeridas presentes.
    - Sin valores nulos en columnas clave.
    - Sin duplicados (split_id, date).
    - Roles permitidos (validation, test).
    - Forecasts y targets reales válidos (positivos, finitos).
    - Nombres de modelo, versión y target consistentes.
    - Metadatos (feature_count, n_train, etc.) positivos.
    - Parámetros constantes dentro de cada split.
    """
    _require_columns(forecast_df, REQUIRED_COLUMNS, "forecast_df")

    if forecast_df.empty:
        raise ValueError(f"{symbol}: forecast_df is empty")

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["train_start_date"] = pd.to_datetime(df["train_start_date"], errors="raise")
    df["train_end_date"] = pd.to_datetime(df["train_end_date"], errors="raise")

    # Validar consistencia del símbolo
    if set(df["symbol"].dropna().unique()) != {symbol}:
        raise ValueError(
            f"{symbol}: unexpected symbol values in output: {sorted(df['symbol'].dropna().unique().tolist())}"
        )

    if df["date"].isna().any():
        raise ValueError(f"{symbol}: date contains NaN values")

    # Verificar duplicados por split y fecha
    if df.duplicated(subset=["split_id", "date"]).any():
        duplicated_rows = df.loc[df.duplicated(subset=["split_id", "date"], keep=False)]
        raise ValueError(
            f"{symbol}: duplicated (split_id, date) rows found:\n"
            f"{duplicated_rows.head().to_string(index=False)}"
        )

    # Los roles deben ser solo 'validation' o 'test'
    allowed_roles = {"validation", "test"}
    actual_roles = set(df["dataset_role"].dropna().unique())
    if not actual_roles.issubset(allowed_roles):
        raise ValueError(f"{symbol}: invalid dataset_role values found: {sorted(actual_roles)}")

    if df["dataset_role"].isna().any():
        raise ValueError(f"{symbol}: dataset_role contains NaN values")

    # Validar forecasts y targets reales
    if df["yhat_future_rv_5d"].isna().any():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains NaN values")

    if df["future_rv_5d"].isna().any():
        raise ValueError(f"{symbol}: future_rv_5d contains NaN values")

    if (df["yhat_future_rv_5d"] <= 0).any():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains non-positive values")

    if (df["future_rv_5d"] <= 0).any():
        raise ValueError(f"{symbol}: future_rv_5d contains non-positive values")

    # Validar metadatos constantes esperados
    if set(df["model_name"].dropna().unique()) != {"xgboost_regressor"}:
        raise ValueError(f"{symbol}: unexpected model_name values found")

    if set(df["model_version"].dropna().unique()) != {"v1"}:
        raise ValueError(f"{symbol}: unexpected model_version values found")

    if set(df["target_column"].dropna().unique()) != {"future_rv_5d"}:
        raise ValueError(f"{symbol}: unexpected target_column values found")

    if (df["feature_count"] <= 0).any():
        raise ValueError(f"{symbol}: feature_count contains non-positive values")

    if (df["n_train"] <= 0).any():
        raise ValueError(f"{symbol}: n_train contains non-positive values")

    if (df["n_validation"] <= 0).any():
        raise ValueError(f"{symbol}: n_validation contains non-positive values")

    if (df["n_score"] <= 0).any():
        raise ValueError(f"{symbol}: n_score contains non-positive values")

    # Validar que ciertas columnas sean constantes dentro de cada split
    for split_id, split_slice in df.groupby("split_id", sort=True):
        for constant_col in [
            "model_name",
            "model_version",
            "target_column",
            "train_start_date",
            "train_end_date",
            "n_train",
            "n_validation",
            "n_score",
            "feature_count",
            "best_iteration",
            "best_score",
        ]:
            if split_slice[constant_col].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{symbol} {split_id}: column '{constant_col}' is not constant within split"
                )


def main() -> int:
    """
    Orquestador de validación de calidad de los forecasts del XGBoost.
    - Carga configuración.
    - Para cada símbolo del universo:
      - Localiza el archivo Parquet de forecasts.
      - Valida su estructura y contenido.
    - Termina con código de salida 0 si todo está bien.
    """
    settings = load_settings()

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]
    evaluations_root = Path(settings["paths"]["evaluations_path"])
    ml_forecasts_root = evaluations_root / "ml_forecasts"

    print("XGBOOST REGRESSOR QUALITY CHECKS")

    for symbol in universe:
        symbol_lower = symbol.lower()
        output_path = _find_single_parquet(ml_forecasts_root / symbol_lower)

        forecast_df = pd.read_parquet(output_path)
        _validate_symbol_forecast_file(symbol=symbol, forecast_df=forecast_df)

        print(f"[XGBOOST REGRESSOR] OK -> {output_path}")

    print("XGBOOST REGRESSOR QUALITY CHECKS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())