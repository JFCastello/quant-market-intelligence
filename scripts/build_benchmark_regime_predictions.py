from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.models import build_benchmark_regime_predictions
from quant_platform.services.settings import load_settings


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


def main() -> int:
    """
    Orquestador principal para construir predicciones de régimen a partir de forecasts
    del benchmark GARCH.
    
    Proceso:
    1. Carga la configuración global.
    2. Itera sobre cada símbolo del universo.
    3. Localiza los archivos Parquet de forecasts del benchmark y de targets de régimen.
    4. Construye las predicciones de régimen del benchmark usando los umbrales
       materializados en los targets de régimen.
    5. Guarda el resultado en artifacts/evaluations/benchmark_regimes/
    6. Muestra información de seguimiento.
    """
    settings = load_settings()
    benchmark_regime_settings = settings["benchmark_regime"]

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]

    evaluations_root = Path(settings["paths"]["evaluations_path"])
    benchmark_forecasts_root = evaluations_root / "benchmark_forecasts"
    regime_targets_root = evaluations_root / "regime_targets"
    output_root = evaluations_root / "benchmark_regimes"

    output_root.mkdir(parents=True, exist_ok=True)

    print("BENCHMARK REGIME BUILD")
    print(f"[INFO] universe = {universe}")
    print(f"[INFO] output_root = {output_root}")

    for symbol in universe:
        symbol_lower = symbol.lower()

        # Localizar archivos de entrada
        benchmark_path = _find_single_parquet(benchmark_forecasts_root / symbol_lower)
        regime_path = _find_single_parquet(regime_targets_root / symbol_lower)

        print(f"\n[INFO] symbol = {symbol}")
        print(f"[INFO] benchmark_path = {benchmark_path}")
        print(f"[INFO] regime_path    = {regime_path}")

        # Cargar DataFrames
        benchmark_df = pd.read_parquet(benchmark_path)
        regime_df = pd.read_parquet(regime_path)

        # Construir predicciones de régimen del benchmark
        benchmark_regime_df = build_benchmark_regime_predictions(
            benchmark_forecast_df=benchmark_df,
            regime_targets_by_split_df=regime_df,
            settings=benchmark_regime_settings,
        )

        if benchmark_regime_df.empty:
            raise ValueError(f"Benchmark regime build returned empty output for symbol={symbol}")

        # Obtener rango de fechas para el nombre del archivo
        benchmark_regime_df["date"] = pd.to_datetime(benchmark_regime_df["date"], errors="raise")
        min_date = benchmark_regime_df["date"].min().date()
        max_date = benchmark_regime_df["date"].max().date()

        # Crear directorio de salida para el símbolo
        symbol_output_dir = output_root / symbol_lower
        symbol_output_dir.mkdir(parents=True, exist_ok=True)

        # Construir nombre de archivo y guardar
        output_filename = f"{symbol_lower}_{min_date}_{max_date}_benchmark_regimes_v1.parquet"
        output_path = symbol_output_dir / output_filename

        benchmark_regime_df.to_parquet(output_path, index=False)

        # Información de seguimiento
        print(
            f"[OK] {symbol} -> {output_path} | "
            f"rows={len(benchmark_regime_df)} | "
            f"splits={benchmark_regime_df['split_id'].nunique()} | "
            f"predicted_regimes={benchmark_regime_df['yhat_future_regime_5d'].value_counts(dropna=False).to_dict()}"
        )

    print("\nBENCHMARK REGIME BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())