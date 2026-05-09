# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Importamos funciones de utilidad y módulos específicos de la plataforma
from quant_platform.services.settings import load_settings          # Carga la configuración global
from quant_platform.targets import (                               # Funciones para construir y validar targets
    adapt_targets_to_contract,
    build_continuous_targets,
    validate_target_output,
)


def ensure_directory(path: Path) -> None:
    """
    Crea el directorio especificado (y sus padres) si no existen.
    No lanza error si el directorio ya existe.
    """
    path.mkdir(parents=True, exist_ok=True)


def discover_latest_context_feature_file(symbol: str, context_features_root: Path) -> Path:
    """
    Dado un símbolo (ticker) y la ruta raíz de características de contexto,
    encuentra el archivo Parquet más reciente que corresponda a ese símbolo
    y tenga el sufijo '_context_v1.parquet'.

    Se asume una estructura de directorios:
        context_features_root / <symbol.lower()> / <archivos>.parquet

    Lanza FileNotFoundError si el directorio no existe o no hay archivos candidatos.
    """
    # Normalizamos el símbolo a minúsculas para la estructura de carpetas
    symbol_dir = context_features_root / symbol.lower()

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"Context feature directory not found for symbol `{symbol}`: {symbol_dir}"
        )

    # Buscamos todos los archivos que terminen con '_context_v1.parquet'
    candidates = sorted(symbol_dir.glob("*_context_v1.parquet"))

    if not candidates:
        raise FileNotFoundError(
            f"No context feature parquet files found for symbol `{symbol}` in {symbol_dir}"
        )

    # Devolvemos el último en orden alfabético (asumiendo que la fecha está en el nombre)
    return candidates[-1]


def build_target_output_path(
    symbol: str,
    source_file: Path,
    targets_root: Path,
    target_version: str,
) -> Path:
    """
    Construye la ruta de salida para el archivo de targets generado.

    - El directorio será: targets_root / <symbol.lower()> /
    - El nombre del archivo se deriva del nombre del archivo fuente,
      reemplazando '_context_v1' por '_targets_<target_version>'
      o añadiendo el sufijo si no se encuentra el patrón.
    - La extensión siempre será '.parquet'.
    - Crea el directorio de salida si no existe.
    """
    # Aseguramos que exista la carpeta del símbolo dentro de targets_root
    symbol_dir = targets_root / symbol.lower()
    ensure_directory(symbol_dir)

    # Extraemos el nombre base del archivo sin extensión
    stem = source_file.stem

    # Reemplazamos '_context_v1' por '_targets_<version>' o simplemente añadimos el sufijo
    if "_context_v1" in stem:
        output_stem = stem.replace("_context_v1", f"_targets_{target_version}")
    else:
        output_stem = f"{stem}_targets_{target_version}"

    # Devolvemos la ruta completa con extensión .parquet
    return symbol_dir / f"{output_stem}.parquet"


def main() -> None:
    """
    Función principal que ejecuta el pipeline completo de generación de targets.

    Pasos:
    1. Cargar configuración global.
    2. Leer la lista de símbolos (universo) y rutas de datos.
    3. Para cada símbolo:
        a. Descubrir el archivo de características de contexto más reciente.
        b. Leerlo como DataFrame.
        c. Construir el target continuo (volatilidad realizada futura).
        d. Adaptar el DataFrame al contrato (validación con Pydantic y selección de columnas).
        e. Validar exhaustivamente el output.
        f. Guardar el resultado en formato Parquet.
        g. Imprimir estadísticas de control.
    4. Finalizar con mensaje de éxito.
    """
    # Cargamos la configuración desde archivo (YAML/JSON según implementación de load_settings)
    settings = load_settings()

    # Extraemos parámetros relevantes de la configuración
    symbols = settings["data"]["universe"]                         # Lista de símbolos a procesar
    context_features_root = Path(
        settings["paths"].get("context_features_path", "data/features_context")
    )                                                              # Ruta base de features de contexto
    targets_root = Path(settings["paths"]["targets_path"])         # Ruta donde guardar los targets
    target_version = settings["targets"]["target_version"]         # Versión del target (ej: 'v1')

    # Imprimimos información de contexto para el usuario/operador
    print("=" * 80)
    print("BUILD TARGETS")
    print("=" * 80)
    print(f"context_features_root={context_features_root}")
    print(f"targets_root={targets_root}")
    print(f"target_version={target_version}")
    print(f"symbols={symbols}")
    print("-" * 80)

    # Iteramos sobre cada símbolo en el universo de activos
    for symbol in symbols:
        # 1. Encontrar el archivo fuente más reciente para este símbolo
        source_file = discover_latest_context_feature_file(
            symbol=symbol,
            context_features_root=context_features_root,
        )

        # 2. Leer los datos de entrada (features de contexto)
        source_df = pd.read_parquet(source_file)

        # 3. Construir el DataFrame de targets continuos
        #    (internamente valida entrada, ordena, calcula volatilidad futura, añade target_version)
        target_df = build_continuous_targets(
            df=source_df,
            settings=settings,
        )

        # 4. Adaptar al contrato definido (selecciona columnas habilitadas y valida con Pydantic)
        target_df = adapt_targets_to_contract(
            df=target_df,
            settings=settings,
        )

        # 5. Validación final exhaustiva (estructura, tipos, contenido, orden, etc.)
        validate_target_output(
            df=target_df,
            settings=settings,
        )

        # 6. Construir ruta de salida
        output_file = build_target_output_path(
            symbol=symbol,
            source_file=source_file,
            targets_root=targets_root,
            target_version=target_version,
        )

        # 7. Guardar el DataFrame como archivo Parquet (sin índice)
        target_df.to_parquet(output_file, index=False)

        # 8. Imprimir resumen del procesamiento para este símbolo
        print(f"symbol={symbol}")
        print(f"input_file={source_file}")
        print(f"output_file={output_file}")
        print(f"rows={len(target_df)}")
        print(f"min_date={target_df['date'].min()}")
        print(f"max_date={target_df['date'].max()}")
        print("nan_counts=")
        print(target_df.isna().sum())      # Cantidad de nulos por columna
        print("-" * 80)

    # Mensaje final de éxito
    print("TARGET BUILD: PASS")


# Punto de entrada del script cuando se ejecuta directamente
if __name__ == "__main__":
    main()