from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.models import build_garch_benchmark_forecasts_by_split
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
    Orquestador principal para generar forecasts del benchmark GARCH.
    - Carga la configuración.
    - Itera sobre cada símbolo del universo.
    - Localiza los archivos de precios normalizados y de splits.
    - Construye los forecasts respetando los splits (train/validation/test).
    - Guarda los resultados en Parquet con nombre descriptivo.
    - Retorna 0 si todo es exitoso.
    """
    settings = load_settings()
    benchmark_settings = settings["benchmark"]

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]

    normalized_root = Path(settings["paths"]["normalized_path"])
    evaluations_root = Path(settings["paths"]["evaluations_path"])
    output_root = evaluations_root / "benchmark_forecasts"

    output_root.mkdir(parents=True, exist_ok=True)

    print("GARCH BENCHMARK BUILD")
    print(f"[INFO] universe = {universe}")
    print(f"[INFO] output_root = {output_root}")

    for symbol in universe:
        symbol_lower = symbol.lower()

        # Directorios específicos del símbolo
        normalized_dir = normalized_root / symbol_lower
        splits_dir = evaluations_root / "splits" / symbol_lower
        symbol_output_dir = output_root / symbol_lower
        symbol_output_dir.mkdir(parents=True, exist_ok=True)

        # Localizar archivos Parquet de entrada
        normalized_path = _find_single_parquet(normalized_dir)
        split_path = _find_single_parquet(splits_dir)

        print(f"\n[INFO] symbol = {symbol}")
        print(f"[INFO] normalized_path = {normalized_path}")
        print(f"[INFO] split_path      = {split_path}")

        # Cargar DataFrames
        normalized_df = pd.read_parquet(normalized_path)
        split_df = pd.read_parquet(split_path)

        # Generar forecasts GARCH por split
        forecast_df = build_garch_benchmark_forecasts_by_split(
            normalized_df=normalized_df,
            split_df=split_df,
            settings=benchmark_settings,
            symbol=symbol,
        )

        if forecast_df.empty:
            raise ValueError(f"Benchmark forecast build returned empty output for symbol={symbol}")

        # Obtener rango de fechas para el nombre del archivo
        min_date = pd.to_datetime(normalized_df["date"], errors="raise").min().date()
        max_date = pd.to_datetime(normalized_df["date"], errors="raise").max().date()

        # Construir nombre de salida descriptivo
        output_filename = (
            f"{symbol_lower}_{min_date}_{max_date}_"
            f"{benchmark_settings['benchmark_name']}_"
            f"{benchmark_settings['benchmark_version']}.parquet"
        )
        output_path = symbol_output_dir / output_filename

        # Guardar resultados
        forecast_df.to_parquet(output_path, index=False)

        # Información de seguimiento
        print(
            f"[OK] {symbol} -> {output_path} | "
            f"rows={len(forecast_df)} | "
            f"splits={forecast_df['split_id'].nunique()} | "
            f"roles={forecast_df['dataset_role'].value_counts(dropna=False).to_dict()}"
        )

    print("\nGARCH BENCHMARK BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())