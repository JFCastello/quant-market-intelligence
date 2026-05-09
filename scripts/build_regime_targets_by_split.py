from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.targets import (
    build_regime_targets_by_split,
    validate_regime_split_output,
)
from quant_platform.services.settings import load_settings


def ensure_directory(path: Path) -> None:
    """Crea el directorio (y sus padres) si no existe. No falla si ya existe."""
    path.mkdir(parents=True, exist_ok=True)


def discover_latest_target_file(symbol: str, targets_root: Path) -> Path:
    """
    Encuentra el archivo Parquet más reciente de targets continuos para un símbolo.
    Busca en targets_root/symbol_lower/ y devuelve el último archivo .parquet
    según orden alfabético (convención de nombres con timestamp o secuencia).
    """
    symbol_dir = targets_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Target directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    candidates = sorted(symbol_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No target parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    return candidates[-1]


def discover_latest_split_file(symbol: str, splits_root: Path) -> Path:
    """
    Encuentra el archivo Parquet más reciente de splits para un símbolo.
    Busca en splits_root/symbol_lower/ y devuelve el último .parquet.
    """
    symbol_dir = splits_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    candidates = sorted(symbol_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No split parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    return candidates[-1]


def build_regime_split_output_path(
    symbol: str,
    source_target_file: Path,
    output_root: Path,
) -> Path:
    """
    Construye la ruta de salida para el archivo de régimen por split.
    Estructura: output_root/symbol_lower/{nombre_target}_regime_by_split_v1.parquet
    """
    symbol_dir = output_root / symbol.lower()
    ensure_directory(symbol_dir)

    stem = source_target_file.stem  # nombre base del archivo de targets
    output_stem = f"{stem}_regime_by_split_v1"
    return symbol_dir / f"{output_stem}.parquet"


def main() -> None:
    """
    Orquestador principal:
    1. Carga la configuración.
    2. Para cada símbolo del universo:
       - Descubre el archivo de targets y el de splits más recientes.
       - Lee los DataFrames.
       - Construye los targets de régimen por split usando la lógica del módulo targets.
       - Valida el resultado.
       - Guarda el resultado en Parquet.
       - Muestra información de depuración.
    """
    settings = load_settings()

    symbols = settings["data"]["universe"]
    targets_root = Path(settings["paths"]["targets_path"])
    splits_root = Path("artifacts/evaluations/splits")
    output_root = Path("artifacts/evaluations/regime_targets")

    print("=" * 80)
    print("BUILD REGIME TARGETS BY SPLIT")
    print("=" * 80)
    print(f"targets_root={targets_root}")
    print(f"splits_root={splits_root}")
    print(f"output_root={output_root}")
    print(f"symbols={symbols}")
    print("-" * 80)

    for symbol in symbols:
        # 1. Localizar archivos fuente
        target_file = discover_latest_target_file(
            symbol=symbol,
            targets_root=targets_root,
        )
        split_file = discover_latest_split_file(
            symbol=symbol,
            splits_root=splits_root,
        )

        # 2. Cargar DataFrames
        target_df = pd.read_parquet(target_file)
        split_df = pd.read_parquet(split_file)

        # 3. Construir targets de régimen respetando los splits
        regime_split_df = build_regime_targets_by_split(
            target_df=target_df,
            split_df=split_df,
            settings=settings,
        )

        # 4. Validar estructura y consistencia
        validate_regime_split_output(
            regime_split_df=regime_split_df,
            settings=settings,
        )

        # 5. Guardar resultado
        output_file = build_regime_split_output_path(
            symbol=symbol,
            source_target_file=target_file,
            output_root=output_root,
        )
        regime_split_df.to_parquet(output_file, index=False)

        # 6. Información de seguimiento
        print(f"symbol={symbol}")
        print(f"target_file={target_file}")
        print(f"split_file={split_file}")
        print(f"output_file={output_file}")
        print(f"rows={len(regime_split_df)}")

        if not regime_split_df.empty:
            print(f"first_split={regime_split_df.iloc[0]['split_id']}")
            print(f"last_split={regime_split_df.iloc[-1]['split_id']}")
            print("head=")
            print(regime_split_df.head())
        else:
            print("WARNING: regime_split_df is empty")

        print("-" * 80)

    print("REGIME TARGETS BY SPLIT BUILD: PASS")


if __name__ == "__main__":
    main()