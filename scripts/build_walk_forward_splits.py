from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.evaluation import (
    build_walk_forward_splits,
    validate_split_output,
)
from quant_platform.services.settings import load_settings


def ensure_directory(path: Path) -> None:
    """Crea el directorio (y sus padres) si no existe. No falla si ya existe."""
    path.mkdir(parents=True, exist_ok=True)


def discover_latest_target_file(symbol: str, targets_root: Path) -> Path:
    """
    Encuentra el archivo Parquet más reciente (por orden alfabético/numérico)
    para un símbolo dado dentro del directorio raíz de targets.
    Asume que los nombres de archivo siguen un orden creciente (ej: por fecha).
    """
    symbol_dir = targets_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Target directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    # Listar todos los archivos .parquet y ordenarlos alfabéticamente
    candidates = sorted(symbol_dir.glob("*.parquet"))

    if not candidates:
        raise FileNotFoundError(
            f"No target parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    # El último en orden alfabético es el más reciente (convención de nombres)
    return candidates[-1]


def build_split_output_path(
    symbol: str,
    source_file: Path,
    output_root: Path,
    split_version: str,
) -> Path:
    """
    Construye la ruta completa donde se guardará el archivo de splits para un símbolo.
    La estructura es: output_root/symbol_lower/nombre_base_splits_version.parquet
    """
    symbol_dir = output_root / symbol.lower()
    ensure_directory(symbol_dir)

    stem = source_file.stem  # Nombre del archivo fuente sin extensión
    output_stem = f"{stem}_splits_{split_version}"
    return symbol_dir / f"{output_stem}.parquet"


def main() -> None:
    """
    Orquestador principal:
    1. Carga la configuración global.
    2. Itera sobre cada símbolo del universo.
    3. Descubre el archivo de targets más reciente.
    4. Construye los splits walk‑forward usando la configuración.
    5. Valida el resultado.
    6. Guarda el DataFrame de splits en formato Parquet.
    7. Muestra información de depuración.
    """
    settings = load_settings()

    symbols = settings["data"]["universe"]
    targets_root = Path(settings["paths"]["targets_path"])
    split_version = settings["splits"]["split_version"]
    output_root = Path("artifacts/evaluations/splits")

    # Cabecera informativa
    print("=" * 80)
    print("BUILD WALK-FORWARD SPLITS")
    print("=" * 80)
    print(f"targets_root={targets_root}")
    print(f"output_root={output_root}")
    print(f"split_version={split_version}")
    print(f"symbols={symbols}")
    print("-" * 80)

    for symbol in symbols:
        # 1. Localizar el archivo fuente de targets para este símbolo
        source_file = discover_latest_target_file(
            symbol=symbol,
            targets_root=targets_root,
        )

        # 2. Cargar el DataFrame desde Parquet
        df = pd.read_parquet(source_file)

        # 3. Generar los splits walk‑forward (función del módulo evaluation)
        split_df = build_walk_forward_splits(
            df=df,
            settings=settings,
        )

        # 4. Validar la estructura y consistencia del DataFrame de splits
        validate_split_output(
            split_df=split_df,
            settings=settings,
        )

        # 5. Definir la ruta de salida y guardar el resultado
        output_file = build_split_output_path(
            symbol=symbol,
            source_file=source_file,
            output_root=output_root,
            split_version=split_version,
        )

        split_df.to_parquet(output_file, index=False)

        # 6. Mostrar resumen del proceso
        print(f"symbol={symbol}")
        print(f"input_file={source_file}")
        print(f"output_file={output_file}")
        print(f"rows={len(split_df)}")

        if not split_df.empty:
            print(f"first_split={split_df.iloc[0]['split_id']}")
            print(f"last_split={split_df.iloc[-1]['split_id']}")
            print("head=")
            print(split_df.head())
        else:
            print("WARNING: split_df is empty")

        print("-" * 80)

    print("WALK-FORWARD SPLIT BUILD: PASS")


if __name__ == "__main__":
    main()