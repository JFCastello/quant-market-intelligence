from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.services.settings import load_settings


# Conjunto de columnas obligatorias en el DataFrame de predicciones de régimen del benchmark
REQUIRED_COLUMNS = {
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
    "future_rv_5d",
    "future_regime_5d",
    "threshold_low",
    "threshold_high",
    "regime_thresholds_source",
    "yhat_future_regime_5d",
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


def _validate_symbol_file(symbol: str, benchmark_regime_df: pd.DataFrame) -> None:
    """
    Valida el archivo de predicciones de régimen del benchmark para un símbolo:
    - Columnas requeridas presentes.
    - Sin valores nulos en columnas clave.
    - Sin duplicados (split_id, date).
    - Roles permitidos (validation, test).
    - Fuente de umbrales correcta (train_only).
    - Umbrales ordenados (threshold_low < threshold_high).
    - Forecasts continuos y targets reales válidos (positivos, finitos).
    - Etiquetas de régimen dentro de las permitidas (calm, normal, stress).
    - Parámetros del modelo constantes dentro de cada split.
    """
    _require_columns(benchmark_regime_df, REQUIRED_COLUMNS, "benchmark_regime_df")

    if benchmark_regime_df.empty:
        raise ValueError(f"{symbol}: benchmark_regime_df is empty")

    df = benchmark_regime_df.copy()
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

    # Validar fuente de umbrales (debe ser 'train_only')
    expected_threshold_source = "train_only"
    actual_threshold_sources = set(df["regime_thresholds_source"].dropna().unique())
    if actual_threshold_sources != {expected_threshold_source}:
        raise ValueError(
            f"{symbol}: unexpected threshold source values: {sorted(actual_threshold_sources)}"
        )

    # Validar orden de umbrales
    if (df["threshold_low"] >= df["threshold_high"]).any():
        bad_rows = df.loc[df["threshold_low"] >= df["threshold_high"]]
        raise ValueError(
            f"{symbol}: found rows with threshold_low >= threshold_high:\n"
            f"{bad_rows.head().to_string(index=False)}"
        )

    # Validar forecasts continuos
    if df["yhat_future_rv_5d"].isna().any():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains NaN values")

    if (df["yhat_future_rv_5d"] <= 0).any():
        raise ValueError(f"{symbol}: yhat_future_rv_5d contains non-positive values")

    # Validar target real continuo
    if df["future_rv_5d"].isna().any():
        raise ValueError(f"{symbol}: future_rv_5d contains NaN values")

    # Validar etiquetas de régimen
    allowed_regimes = {"calm", "normal", "stress"}

    predicted_regimes = set(df["yhat_future_regime_5d"].dropna().unique())
    if not predicted_regimes.issubset(allowed_regimes):
        raise ValueError(
            f"{symbol}: invalid predicted regime labels found: {sorted(predicted_regimes)}"
        )

    actual_regimes = set(df["future_regime_5d"].dropna().unique())
    if not actual_regimes.issubset(allowed_regimes):
        raise ValueError(
            f"{symbol}: invalid actual regime labels found: {sorted(actual_regimes)}"
        )

    if df["yhat_future_regime_5d"].isna().any():
        raise ValueError(f"{symbol}: yhat_future_regime_5d contains NaN values")

    if df["future_regime_5d"].isna().any():
        raise ValueError(f"{symbol}: future_regime_5d contains NaN values")

    # Validar que los parámetros y metadatos sean constantes dentro de cada split
    for split_id, split_slice in df.groupby("split_id", sort=True):
        for constant_col in [
            "threshold_low",
            "threshold_high",
            "regime_thresholds_source",
            "omega",
            "alpha_1",
            "beta_1",
            "nu",
            "n_train",
            "train_start_date",
            "train_end_date",
        ]:
            if split_slice[constant_col].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{symbol} {split_id}: column '{constant_col}' is not constant within split"
                )


def main() -> int:
    """
    Orquestador de validación de calidad de las predicciones de régimen del benchmark.
    - Carga configuración.
    - Para cada símbolo del universo:
      - Localiza el archivo Parquet de predicciones de régimen.
      - Valida su estructura y contenido.
    - Termina con código de salida 0 si todo está bien.
    """
    settings = load_settings()

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]
    evaluations_root = Path(settings["paths"]["evaluations_path"])
    benchmark_regimes_root = evaluations_root / "benchmark_regimes"

    print("BENCHMARK REGIME QUALITY CHECKS")

    for symbol in universe:
        symbol_lower = symbol.lower()
        output_path = _find_single_parquet(benchmark_regimes_root / symbol_lower)

        benchmark_regime_df = pd.read_parquet(output_path)
        _validate_symbol_file(symbol=symbol, benchmark_regime_df=benchmark_regime_df)

        print(f"[BENCHMARK REGIME] OK -> {output_path}")

    print("BENCHMARK REGIME QUALITY CHECKS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())