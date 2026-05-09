from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.features import (
    adapt_features_to_contract,
    build_base_features,
    validate_feature_output,
)
from quant_platform.services.settings import load_settings

def ensure_directory(path: Path) -> None:
    # Crea el directorio (y padres si no existen) sin error si ya existe.
    path.mkdir(parents=True, exist_ok=True)

def discover_latest_normalized_file(symbol: str, normalized_root: Path) -> Path:
    # Busca el archivo de datos normalizados más reciente para un símbolo.
    #   -Asume que dentro de normalized_root/symbol/ hay archivos *_daily_bars.parquet.
    #   -Ordena por nombre (que debe incluir fecha o versión) y toma el último (el más nuevo).
    #   -Lanza error si no encuentra nada.
    symbol_dir = normalized_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Normalized directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    candidates = sorted(symbol_dir.glob("*_daily_bars.parquet"))

    if not candidates:
        raise FileNotFoundError(
            f"No normalized parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    return candidates[-1]


def build_feature_output_path(    
    symbol: str,
    normalized_file: Path,
    features_root: Path,
    feature_version: str,
) -> Path:
    # Construye la ruta donde se guardará el archivo de features.
    #   -Crea el directorio de salida (features_root/symbol/).
    #   -El nombre del archivo se basa en el del normalized, reemplazando _daily_bars por _features_{feature_version}.
    
    symbol_dir = features_root / symbol.lower()
    ensure_directory(symbol_dir)

    stem = normalized_file.stem
    if stem.endswith("_daily_bars"):
        feature_stem = stem.replace("_daily_bars", f"_features_{feature_version}")
    else:
        feature_stem = f"{stem}_features_{feature_version}"

    return symbol_dir / f"{feature_stem}.parquet"

# Orquestador del archivo
def main() -> None:
    #1. Localizar el normalized file más reciente.
    #2. Cargar normalized_df con pd.read_parquet.
    #3. Construir features.
    #4. Adaptar al contrato 
    #6. Validar output 
    #7. Generar ruta de salida y guardar en Parquet (sin índice)
    #8. Imprimir métricas: número de filas, rango de fechas, conteo de nulos por columna.
    
    settings = load_settings()

    symbols = settings["data"]["universe"]
    normalized_root = Path(settings["paths"]["normalized_path"])
    features_root = Path(settings["paths"]["features_path"])
    feature_version = settings["features"]["feature_version"]

    print("=" * 80)
    print("BUILD FEATURES")
    print("=" * 80)
    print(f"normalized_root={normalized_root}")
    print(f"features_root={features_root}")
    print(f"feature_version={feature_version}")
    print(f"symbols={symbols}")
    print("-" * 80)

    for symbol in symbols:
        normalized_file = discover_latest_normalized_file(
            symbol=symbol,
            normalized_root=normalized_root,
        )

        normalized_df = pd.read_parquet(normalized_file)

        feature_df = build_base_features(
            df=normalized_df,
            settings=settings,
        )

        feature_df = adapt_features_to_contract(
            df=feature_df,
            settings=settings,
        )

        validate_feature_output(
            df=feature_df,
            settings=settings,
        )

        output_file = build_feature_output_path(
            symbol=symbol,
            normalized_file=normalized_file,
            features_root=features_root,
            feature_version=feature_version,
        )

        feature_df.to_parquet(output_file, index=False)

        print(f"symbol={symbol}")
        print(f"input_file={normalized_file}")
        print(f"output_file={output_file}")
        print(f"rows={len(feature_df)}")
        print(f"min_date={feature_df['date'].min()}")
        print(f"max_date={feature_df['date'].max()}")
        print("nan_counts=")
        print(feature_df.isna().sum())
        print("-" * 80)

    print("FEATURE BUILD: PASS")


if __name__ == "__main__":
    main()