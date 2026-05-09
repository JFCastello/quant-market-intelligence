# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Importamos funciones de utilidad de la plataforma
from quant_platform.targets import get_enabled_target_columns  # Obtiene las columnas esperadas según config
from quant_platform.services.settings import load_settings      # Carga la configuración global


def discover_target_files(targets_root: Path, symbols: list[str]) -> list[Path]:
    """
    Descubre todos los archivos Parquet de targets existentes para los símbolos proporcionados.

    Recorre cada símbolo, verifica si existe su directorio correspondiente
    (targets_root / <symbol.lower()>) y añade a la lista todos los archivos .parquet
    encontrados en orden alfabético.

    Args:
        targets_root: Ruta raíz donde se almacenan los targets.
        symbols: Lista de símbolos (tickers) que forman parte del universo.

    Returns:
        Lista de rutas (Path) a los archivos de targets descubiertos.
    """
    files: list[Path] = []

    for symbol in symbols:
        symbol_dir = targets_root / symbol.lower()
        if symbol_dir.exists():
            # Usamos glob para encontrar todos los .parquet y los ordenamos
            files.extend(sorted(symbol_dir.glob("*.parquet")))

    return files


def check_target_file(path: Path, settings: dict) -> list[str]:
    """
    Realiza un conjunto de validaciones de calidad sobre un archivo de targets individual.

    Las comprobaciones incluyen:
        - Existencia del archivo.
        - Presencia de todas las columnas esperadas según la configuración.
        - DataFrame no vacío.
        - Ausencia de nulos en columnas clave (instrument_id, date, target_version).
        - Coincidencia de target_version con el valor esperado en settings.
        - Ausencia de duplicados por instrument_id/date.
        - Fechas parseables y correctamente ordenadas.
        - Validaciones específicas para la columna 'future_rv_5d' (si existe):
            - Valores numéricos, sin infinitos, no negativos.
        - Que el archivo contenga exactamente un único instrument_id (por convención de organización).

    Args:
        path: Ruta al archivo Parquet a validar.
        settings: Diccionario de configuración global.

    Returns:
        Lista de cadenas describiendo los problemas encontrados. Vacía si todo está correcto.
    """
    issues: list[str] = []

    # 1. Verificar existencia del archivo
    if not path.exists():
        return [f"[TARGETS] Missing file: {path}"]

    # 2. Leer el DataFrame
    df = pd.read_parquet(path)

    # 3. Obtener columnas esperadas según la configuración de targets activos
    expected_columns = get_enabled_target_columns(settings)

    # 4. Columnas faltantes
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"[TARGETS] Missing columns in {path}: {missing_cols}")
        return issues  # Si faltan columnas, no tiene sentido continuar con más validaciones

    # 5. DataFrame vacío
    if df.empty:
        issues.append(f"[TARGETS] Empty dataframe in {path}")
        return issues

    # 6. Nulos en columnas obligatorias
    if df["instrument_id"].isna().any():
        issues.append(f"[TARGETS] Null instrument_id values in {path}")

    if df["date"].isna().any():
        issues.append(f"[TARGETS] Null date values in {path}")

    if df["target_version"].isna().any():
        issues.append(f"[TARGETS] Null target_version values in {path}")

    # 7. Versión del target consistente con configuración
    expected_version = settings["targets"]["target_version"]
    if not (df["target_version"] == expected_version).all():
        issues.append(
            f"[TARGETS] target_version different from `{expected_version}` in {path}"
        )

    # 8. Duplicados en la clave natural (instrument_id, date)
    if df.duplicated(subset=["instrument_id", "date"]).any():
        issues.append(f"[TARGETS] Duplicate instrument_id/date rows in {path}")

    # 9. Fechas parseables y ordenamiento
    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().any():
        issues.append(f"[TARGETS] Unparseable date values in {path}")
    else:
        # Verificar que el DataFrame esté ordenado por instrument_id y date
        sorted_dates = df.sort_values(["instrument_id", "date"])["date"].reset_index(drop=True)
        current_dates = df["date"].reset_index(drop=True)
        if not current_dates.equals(sorted_dates):
            issues.append(f"[TARGETS] Dates are not sorted by instrument_id/date in {path}")

    # 10. Validaciones específicas para el target continuo 'future_rv_5d' (volatilidad realizada 5 días)
    if "future_rv_5d" in df.columns:
        # Intentar convertir a numérico, forzando errores a NaN
        target_series = pd.to_numeric(df["future_rv_5d"], errors="coerce")

        # Comprobar si hay valores no numéricos (comparando nulos antes y después)
        if target_series.isna().sum() > df["future_rv_5d"].isna().sum():
            issues.append(f"[TARGETS] Non-numeric values found in `future_rv_5d` of {path}")

        # Comprobar valores infinitos
        inf_mask = np.isinf(target_series.to_numpy(dtype=float, copy=True))
        if inf_mask.any():
            issues.append(f"[TARGETS] inf/-inf values found in `future_rv_5d` of {path}")

        # La volatilidad no puede ser negativa
        if (target_series.dropna() < 0).any():
            issues.append(f"[TARGETS] Negative values found in `future_rv_5d` of {path}")

    # 11. Verificar que el archivo contenga datos de un único instrumento (por diseño de almacenamiento)
    unique_instruments = df["instrument_id"].dropna().unique().tolist()
    if len(unique_instruments) != 1:
        issues.append(
            f"[TARGETS] Expected exactly one instrument_id per file in {path}, found {unique_instruments}"
        )

    return issues


def main() -> None:
    """
    Función principal que ejecuta el control de calidad sobre todos los archivos de targets.

    Pasos:
    1. Cargar configuración.
    2. Descubrir todos los archivos de targets existentes para el universo definido.
    3. Para cada archivo, ejecutar las validaciones definidas en check_target_file().
    4. Si se encuentran incidencias, imprimirlas y terminar con código de error (sys.exit(1)).
    5. Si todo está correcto, imprimir confirmación y lista de archivos validados.
    """
    # Cargar configuración global
    settings = load_settings()

    # Obtener rutas y símbolos de la configuración
    targets_root = Path(settings["paths"]["targets_path"])
    symbols = settings["data"]["universe"]

    # Descubrir archivos de targets existentes
    target_files = discover_target_files(
        targets_root=targets_root,
        symbols=symbols,
    )

    issues: list[str] = []

    # Si no se encuentra ningún archivo, se considera un fallo
    if not target_files:
        issues.append(f"[TARGETS] No target parquet files discovered under {targets_root}")

    # Validar cada archivo encontrado
    for path in target_files:
        issues.extend(check_target_file(path, settings))

    # Evaluar resultados
    if issues:
        print("TARGET QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)  # Salida con código de error para integración en CI/CD

    # Éxito: todos los archivos pasaron las validaciones
    print("TARGET QUALITY CHECKS: PASS")
    for path in target_files:
        print(f"[TARGETS] OK -> {path}")


# Punto de entrada del script cuando se ejecuta directamente
if __name__ == "__main__":
    main()