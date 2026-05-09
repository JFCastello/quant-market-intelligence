# Archivo: build_context_features.py
# Propósito: Construir features de contexto (relaciones entre activos) para un universo
#            de símbolos, combinando las features base de cada uno, enriqueciéndolas
#            con información de mercado relativa, y guardando el resultado por símbolo.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.features import (
    build_context_enriched_features,
    get_enabled_context_feature_columns,
    get_enabled_feature_columns,
)
from quant_platform.schemas import FeatureRow
from quant_platform.services.settings import load_settings


# ----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE ARCHIVOS Y RUTAS
# ----------------------------------------------------------------------------

def ensure_directory(path: Path) -> None:
    """Crea el directorio (y sus padres) si no existe, sin lanzar error si ya existe."""
    path.mkdir(parents=True, exist_ok=True)


def discover_latest_base_feature_file(symbol: str, features_root: Path) -> Path:
    """
    Busca el archivo de features base más reciente para un símbolo.
    Se asume que los archivos se nombran como '*_features_v1.parquet'.
    Devuelve el último en orden alfabético (normalmente el más nuevo por versión/fecha).
    """
    symbol_dir = features_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Base feature directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    # Filtra archivos que terminan con '_features_v1.parquet' (versión base)
    candidates = sorted(symbol_dir.glob("*_features_v1.parquet"))

    if not candidates:
        raise FileNotFoundError(
            f"No base feature parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    return candidates[-1]   # el más reciente (por nombre)


def build_context_output_path(
    symbol: str,
    base_feature_file: Path,
    context_features_root: Path,
    context_tag: str,
) -> Path:
    """
    Construye la ruta de salida para el archivo de features enriquecidas con contexto.
    Ejemplo: data/features_context/spy/spy_features_v1_context_v1.parquet
    """
    symbol_dir = context_features_root / symbol.lower()
    ensure_directory(symbol_dir)

    stem = base_feature_file.stem                     # ej. "spy_features_v1"
    output_stem = f"{stem}_{context_tag}"             # añade el tag de contexto
    return symbol_dir / f"{output_stem}.parquet"


def instrument_id_to_symbol(instrument_id: str) -> str:
    """
    Extrae el símbolo base de un instrument_id.
    Asume formato como "spy_us_2020" -> "spy" (parte antes del primer '_').
    """
    return instrument_id.split("_")[0].lower()


# ----------------------------------------------------------------------------
# ADAPTACIÓN AL CONTRATO (VALIDACIÓN Y TIPADO ESTRICTO)
# ----------------------------------------------------------------------------

def adapt_context_enriched_to_contract(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    """
    Filtra el DataFrame enriquecido para que contenga solo las columnas esperadas
    (features base + features de contexto) y valida cada fila con el esquema FeatureRow.
    """
    base_columns = get_enabled_feature_columns(settings)           # columnas base
    context_columns = get_enabled_context_feature_columns(settings) # columnas de contexto

    final_columns = base_columns + context_columns

    # Verifica que todas las columnas esperadas existan
    missing = [col for col in final_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Context-enriched dataframe is missing expected contract columns: {missing}"
        )

    contracted = df[final_columns].copy()

    # Convierte cada fila a dict y luego valida con el modelo Pydantic FeatureRow
    records = contracted.to_dict(orient="records")
    validated_records = [FeatureRow(**record).model_dump() for record in records]

    validated_df = pd.DataFrame(validated_records)
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="raise")

    return validated_df


# ----------------------------------------------------------------------------
# ORQUESTADOR PRINCIPAL
# ----------------------------------------------------------------------------

def main() -> None:
    """Carga configuración, lee features base de todos los símbolos, construye
    contexto a nivel de universo, y guarda el resultado en archivos separados por símbolo."""
    settings = load_settings()

    # Parámetros desde settings
    symbols = settings["data"]["universe"]                           # lista de símbolos
    base_features_root = Path(settings["paths"]["features_path"])    # donde están las features base
    context_features_root = Path(
        settings["paths"].get("context_features_path", "data/features_context")
    )
    context_tag = settings.get("context_features", {}).get("tag", "context_v1")

    # Impresión informativa
    print("=" * 80)
    print("BUILD CONTEXT FEATURES")
    print("=" * 80)
    print(f"base_features_root={base_features_root}")
    print(f"context_features_root={context_features_root}")
    print(f"context_tag={context_tag}")
    print(f"symbols={symbols}")
    print("-" * 80)

    # --------------------------------------------------------------------
    # 1. Descubrir archivos base y cargarlos en un único DataFrame (universo)
    # --------------------------------------------------------------------
    base_feature_files: dict[str, Path] = {}   # mapea símbolo -> ruta del archivo base
    dfs: list[pd.DataFrame] = []               # lista para concatenar

    for symbol in symbols:
        path = discover_latest_base_feature_file(
            symbol=symbol,
            features_root=base_features_root,
        )
        base_feature_files[symbol] = path
        dfs.append(pd.read_parquet(path))

    universe_df = pd.concat(dfs, ignore_index=True)   # DataFrame con datos de todos los símbolos

    # --------------------------------------------------------------------
    # 2. Construir features de contexto sobre el universo completo
    # --------------------------------------------------------------------
    context_df = build_context_enriched_features(
        universe_features_df=universe_df,
        settings=settings,
    )

    # 3. Adaptar al contrato (validación + tipos)
    context_df = adapt_context_enriched_to_contract(
        df=context_df,
        settings=settings,
    )

    # --------------------------------------------------------------------
    # 4. Mostrar estadísticas generales del universo enriquecido
    # --------------------------------------------------------------------
    ctx_cols = [col for col in context_df.columns if col.startswith("ctx_")]

    print("universe_summary")
    print(f"rows={len(context_df)}")
    print(f"min_date={context_df['date'].min()}")
    print(f"max_date={context_df['date'].max()}")
    print("context_columns=")
    for col in ctx_cols:
        print(f"  - {col}")
    print("-" * 80)

    # --------------------------------------------------------------------
    # 5. Dividir por instrument_id (símbolo real) y guardar cada uno por separado
    # --------------------------------------------------------------------
    for instrument_id, group_df in context_df.groupby("instrument_id", sort=True):
        symbol = instrument_id_to_symbol(instrument_id).upper()

        # Verifica que el símbolo extraído corresponda a alguno del universo original
        if symbol not in base_feature_files:
            raise ValueError(
                f"Could not map instrument_id `{instrument_id}` back to universe symbol."
            )

        # Ruta de salida para este símbolo
        output_file = build_context_output_path(
            symbol=symbol,
            base_feature_file=base_feature_files[symbol],
            context_features_root=context_features_root,
            context_tag=context_tag,
        )

        # Ordena por fecha y guarda
        group_df = group_df.sort_values("date").reset_index(drop=True)
        group_df.to_parquet(output_file, index=False)

        # Impresión de métricas por símbolo
        print(f"symbol={symbol}")
        print(f"instrument_id={instrument_id}")
        print(f"input_file={base_feature_files[symbol]}")
        print(f"output_file={output_file}")
        print(f"rows={len(group_df)}")
        print(f"min_date={group_df['date'].min()}")
        print(f"max_date={group_df['date'].max()}")
        print("ctx_nan_counts=")
        print(group_df[ctx_cols].isna().sum())
        print("-" * 80)

    print("CONTEXT FEATURE BUILD: PASS")


if __name__ == "__main__":
    main()