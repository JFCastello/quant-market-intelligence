from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quant_platform.features import get_enabled_feature_columns
from quant_platform.services.settings import load_settings


def discover_feature_files(features_root: Path, symbols: list[str]) -> list[Path]:
    """Busca todos los archivos Parquet de features para los símbolos dados."""
    files: list[Path] = []

    for symbol in symbols:
        symbol_dir = features_root / symbol.lower()          # directorio del símbolo
        if symbol_dir.exists():
            # Agrega todos los .parquet ordenados (por nombre)
            files.extend(sorted(symbol_dir.glob("*.parquet")))

    return files


def check_feature_file(path: Path, settings: dict) -> list[str]:
    """Valida un archivo de features: columnas, nulos, duplicados, orden, valores numéricos, etc."""
    issues: list[str] = []

    if not path.exists():
        return [f"[FEATURES] Missing file: {path}"]

    df = pd.read_parquet(path)
    expected_columns = get_enabled_feature_columns(settings)   # columnas que debería tener

    # 1. Columnas esperadas
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"[FEATURES] Missing columns in {path}: {missing_cols}")
        return issues   # si faltan columnas, no seguir (evita errores posteriores)

    # 2. DataFrame vacío
    if df.empty:
        issues.append(f"[FEATURES] Empty dataframe in {path}")
        return issues

    # 3. Nulos en columnas clave
    if df["instrument_id"].isna().any():
        issues.append(f"[FEATURES] Null instrument_id values in {path}")

    if df["date"].isna().any():
        issues.append(f"[FEATURES] Null date values in {path}")

    if df["feature_version"].isna().any():
        issues.append(f"[FEATURES] Null feature_version values in {path}")

    # 4. Versión de features correcta
    expected_version = settings["features"]["feature_version"]
    if not (df["feature_version"] == expected_version).all():
        issues.append(
            f"[FEATURES] feature_version different from `{expected_version}` in {path}"
        )

    # 5. Duplicados por instrumento/fecha
    if df.duplicated(subset=["instrument_id", "date"]).any():
        issues.append(f"[FEATURES] Duplicate instrument_id/date rows in {path}")

    # 6. Fechas parseables y orden correcto
    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().any():
        issues.append(f"[FEATURES] Unparseable date values in {path}")
    else:
        sorted_dates = df.sort_values(["instrument_id", "date"])["date"].reset_index(drop=True)
        current_dates = df["date"].reset_index(drop=True)
        if not current_dates.equals(sorted_dates):
            issues.append(f"[FEATURES] Dates are not sorted by instrument_id/date in {path}")

    # 7. Columnas numéricas: deben ser numéricas, sin inf
    numeric_cols = [
        col for col in expected_columns
        if col not in {"instrument_id", "date", "feature_version"}
    ]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")
        # Si hay más NaN después de coercionar, es que había no numéricos
        if numeric_series.isna().sum() > df[col].isna().sum():
            issues.append(f"[FEATURES] Non-numeric values found in `{col}` of {path}")

        inf_mask = np.isinf(numeric_series.to_numpy(dtype=float, copy=True))
        if inf_mask.any():
            issues.append(f"[FEATURES] inf/-inf values found in `{col}` of {path}")

    # 8. Validaciones específicas por tipo de feature
    # Volatilidad y ATR no deben ser negativos
    vol_cols = [col for col in numeric_cols if col.startswith("vol_")]
    for col in vol_cols:
        if (pd.to_numeric(df[col], errors="coerce").dropna() < 0).any():
            issues.append(f"[FEATURES] Negative values found in `{col}` of {path}")

    atr_cols = [col for col in numeric_cols if col.startswith("atr_")]
    for col in atr_cols:
        if (pd.to_numeric(df[col], errors="coerce").dropna() < 0).any():
            issues.append(f"[FEATURES] Negative values found in `{col}` of {path}")

    # Drawdowns no deben ser positivos (deben ser ≤ 0)
    drawdown_cols = [col for col in numeric_cols if col.startswith("drawdown_")]
    for col in drawdown_cols:
        if (pd.to_numeric(df[col], errors="coerce").dropna() > 1e-12).any():
            issues.append(f"[FEATURES] Positive values found in `{col}` of {path}")

    # Rango high-low (hl_range) no negativo
    if "hl_range" in df.columns:
        if (pd.to_numeric(df["hl_range"], errors="coerce").dropna() < 0).any():
            issues.append(f"[FEATURES] Negative values found in `hl_range` of {path}")

    return issues


def main() -> None:
    """Orquestador: carga settings, descubre archivos de features y los valida uno a uno."""
    settings = load_settings()                                 # carga configuración (YAML)

    features_root = Path(settings["paths"]["features_path"])   # directorio raíz de features
    symbols = settings["data"]["universe"]                     # lista de símbolos

    # Descubre todos los archivos Parquet de features
    feature_files = discover_feature_files(
        features_root=features_root,
        symbols=symbols,
    )

    issues: list[str] = []

    if not feature_files:
        issues.append(f"[FEATURES] No feature parquet files discovered under {features_root}")

    # Valida cada archivo y acumula problemas
    for path in feature_files:
        issues.extend(check_feature_file(path, settings))

    # Reporte final
    if issues:
        print("FEATURE QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)          # salida con error

    print("FEATURE QUALITY CHECKS: PASS")
    for path in feature_files:
        print(f"[FEATURES] OK -> {path}")


if __name__ == "__main__":
    main()