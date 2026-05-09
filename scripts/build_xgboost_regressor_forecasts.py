from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.models import build_xgboost_regressor_forecasts_by_split
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


def _save_split_models(
    split_models: list[dict],
    output_dir: Path,
    symbol_lower: str,
    model_name: str,
    model_version: str,
) -> None:
    """
    Guarda los modelos entrenados por split en formato XGBoost JSON,
    junto con las columnas de features y metadatos asociados.
    Cada split genera tres archivos:
    - modelo: .json
    - lista de features: .json
    - metadatos (fechas, tamaños, etc.): .json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_model in split_models:
        split_id = split_model["split_id"]
        model = split_model["model"]

        # Rutas de archivos
        model_path = output_dir / f"{symbol_lower}_{split_id}_{model_name}_{model_version}.json"
        feature_columns_path = output_dir / f"{symbol_lower}_{split_id}_{model_name}_{model_version}_feature_columns.json"
        metadata_path = output_dir / f"{symbol_lower}_{split_id}_{model_name}_{model_version}_metadata.json"

        # Guardar modelo XGBoost
        model.save_model(model_path)

        # Guardar lista de features usadas
        feature_columns_path.write_text(
            json.dumps(split_model["feature_columns"], indent=2),
            encoding="utf-8",
        )

        # Guardar metadatos relevantes
        metadata = {
            "split_id": split_model["split_id"],
            "train_start_date": str(split_model["train_start_date"]),
            "train_end_date": str(split_model["train_end_date"]),
            "n_train": split_model["n_train"],
            "n_validation": split_model["n_validation"],
            "best_iteration": split_model["best_iteration"],
            "best_score": split_model["best_score"],
            "feature_count": len(split_model["feature_columns"]),
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    """
    Orquestador principal para construir forecasts de XGBoost por split.
    - Carga configuración.
    - Itera sobre cada símbolo del universo.
    - Localiza los archivos de features, targets y splits.
    - Entrena modelos XGBoost por split y genera predicciones OOS.
    - Guarda forecasts en Parquet y (opcionalmente) modelos en JSON.
    - Muestra información de seguimiento.
    """
    settings = load_settings()
    ml_settings = settings["ml_regressor"]

    universe = [str(symbol).upper() for symbol in settings["data"]["universe"]]

    feature_source = ml_settings["feature_source"]
    features_root = Path("data") / feature_source
    targets_root = Path(settings["paths"]["targets_path"])
    evaluations_root = Path(settings["paths"]["evaluations_path"])
    models_root = Path(settings["paths"]["models_path"])

    splits_root = evaluations_root / "splits"
    forecasts_root = evaluations_root / "ml_forecasts"
    model_output_root = models_root / ml_settings["model_name"]

    # Crear directorios de salida si no existen
    forecasts_root.mkdir(parents=True, exist_ok=True)
    model_output_root.mkdir(parents=True, exist_ok=True)

    print("XGBOOST REGRESSOR BUILD")
    print(f"[INFO] universe = {universe}")
    print(f"[INFO] features_root = {features_root}")
    print(f"[INFO] forecasts_root = {forecasts_root}")
    print(f"[INFO] model_output_root = {model_output_root}")

    for symbol in universe:
        symbol_lower = symbol.lower()

        # Localizar archivos de entrada
        features_path = _find_single_parquet(features_root / symbol_lower)
        targets_path = _find_single_parquet(targets_root / symbol_lower)
        splits_path = _find_single_parquet(splits_root / symbol_lower)

        print(f"\n[INFO] symbol = {symbol}")
        print(f"[INFO] features_path = {features_path}")
        print(f"[INFO] targets_path  = {targets_path}")
        print(f"[INFO] splits_path   = {splits_path}")

        # Cargar DataFrames
        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        splits_df = pd.read_parquet(splits_path)

        # Generar forecasts y modelos por split
        forecast_df, split_models = build_xgboost_regressor_forecasts_by_split(
            features_df=features_df,
            targets_df=targets_df,
            split_df=splits_df,
            settings=ml_settings,
            symbol=symbol,
        )

        if forecast_df.empty:
            raise ValueError(f"ML regressor forecast build returned empty output for symbol={symbol}")

        # Obtener rango de fechas para el nombre del archivo
        forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="raise")
        min_date = forecast_df["date"].min().date()
        max_date = forecast_df["date"].max().date()

        # Guardar forecasts
        symbol_forecast_dir = forecasts_root / symbol_lower
        symbol_forecast_dir.mkdir(parents=True, exist_ok=True)

        forecast_filename = (
            f"{symbol_lower}_{min_date}_{max_date}_"
            f"{ml_settings['model_name']}_{ml_settings['model_version']}.parquet"
        )
        forecast_path = symbol_forecast_dir / forecast_filename
        forecast_df.to_parquet(forecast_path, index=False)

        # Persistir modelos si está habilitado
        if ml_settings["persist_models"]:
            symbol_model_dir = model_output_root / symbol_lower
            _save_split_models(
                split_models=split_models,
                output_dir=symbol_model_dir,
                symbol_lower=symbol_lower,
                model_name=ml_settings["model_name"],
                model_version=ml_settings["model_version"],
            )

        # Información de seguimiento
        print(
            f"[OK] {symbol} -> {forecast_path} | "
            f"rows={len(forecast_df)} | "
            f"splits={forecast_df['split_id'].nunique()} | "
            f"roles={forecast_df['dataset_role'].value_counts(dropna=False).to_dict()}"
        )

    print("\nXGBOOST REGRESSOR BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())