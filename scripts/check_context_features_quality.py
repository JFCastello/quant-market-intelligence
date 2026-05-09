# Archivo: check_context_feature_quality.py
# Propósito: Validar la calidad de los archivos Parquet que contienen features de contexto.
#            Similar al validador de features base, pero con reglas específicas para contexto:
#            - Comprueba columnas base + columnas de contexto
#            - Verifica rangos de valores (correlaciones en [-1,1], volatilidades >=0)
#            - Asegura que el archivo contenga exactamente un instrument_id
#            - Para el rol de equity_proxy, la autocorrelación debe ser NaN

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quant_platform.features import (
    get_enabled_context_feature_columns,
    get_enabled_feature_columns,
)
from quant_platform.services.settings import load_settings


# ----------------------------------------------------------------------------
# DESCUBRIMIENTO DE ARCHIVOS DE CONTEXTO
# ----------------------------------------------------------------------------

def discover_context_feature_files(
    context_features_root: Path,
    symbols: list[str],
) -> list[Path]:
    """Busca todos los archivos Parquet de features de contexto para los símbolos dados."""
    files: list[Path] = []

    for symbol in symbols:
        symbol_dir = context_features_root / symbol.lower()
        if symbol_dir.exists():
            files.extend(sorted(symbol_dir.glob("*.parquet")))

    return files


def instrument_id_to_symbol(instrument_id: str) -> str:
    """Convierte un instrument_id (ej. 'spy_us') al símbolo base ('SPY')."""
    return instrument_id.split("_")[0].upper()


# ----------------------------------------------------------------------------
# VALIDACIÓN DE UN ARCHIVO DE CONTEXTO
# ----------------------------------------------------------------------------

def check_context_feature_file(path: Path, settings: dict) -> list[str]:
    """
    Valida un archivo de features de contexto.
    Retorna una lista de problemas encontrados (vacía si todo está correcto).
    """
    issues: list[str] = []

    # 1. El archivo debe existir
    if not path.exists():
        return [f"[CTX] Missing file: {path}"]

    df = pd.read_parquet(path)

    # 2. Columnas esperadas = columnas base + columnas de contexto
    expected_base_columns = get_enabled_feature_columns(settings)
    expected_context_columns = get_enabled_context_feature_columns(settings)
    expected_columns = expected_base_columns + expected_context_columns

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"[CTX] Missing columns in {path}: {missing_cols}")
        return issues   # No seguir si faltan columnas esenciales

    # 3. No vacío
    if df.empty:
        issues.append(f"[CTX] Empty dataframe in {path}")
        return issues

    # 4. Nulos en columnas clave
    if df["instrument_id"].isna().any():
        issues.append(f"[CTX] Null instrument_id values in {path}")

    if df["date"].isna().any():
        issues.append(f"[CTX] Null date values in {path}")

    if df["feature_version"].isna().any():
        issues.append(f"[CTX] Null feature_version values in {path}")

    # 5. Versión correcta
    expected_version = settings["features"]["feature_version"]
    if not (df["feature_version"] == expected_version).all():
        issues.append(
            f"[CTX] feature_version different from `{expected_version}` in {path}"
        )

    # 6. Sin duplicados por instrumento/fecha
    if df.duplicated(subset=["instrument_id", "date"]).any():
        issues.append(f"[CTX] Duplicate instrument_id/date rows in {path}")

    # 7. Fechas parseables y orden correcto
    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().any():
        issues.append(f"[CTX] Unparseable date values in {path}")
    else:
        sorted_dates = df.sort_values(["instrument_id", "date"])["date"].reset_index(drop=True)
        current_dates = df["date"].reset_index(drop=True)
        if not current_dates.equals(sorted_dates):
            issues.append(f"[CTX] Dates are not sorted by instrument_id/date in {path}")

    # 8. Validación de columnas numéricas (tipos numéricos, sin infinitos)
    numeric_cols = [
        col for col in expected_columns
        if col not in {"instrument_id", "date", "feature_version"}
    ]

    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        # Si hay más NaN después de coercionar, es que había no numéricos
        if numeric_series.isna().sum() > df[col].isna().sum():
            issues.append(f"[CTX] Non-numeric values found in `{col}` of {path}")

        inf_mask = np.isinf(numeric_series.to_numpy(dtype=float, copy=True))
        if inf_mask.any():
            issues.append(f"[CTX] inf/-inf values found in `{col}` of {path}")

    # 9. Validaciones específicas de negatividad en volatilidades base
    base_vol_cols = [col for col in expected_base_columns if col.startswith("vol_")]
    for col in base_vol_cols:
        if (pd.to_numeric(df[col], errors="coerce").dropna() < 0).any():
            issues.append(f"[CTX] Negative values found in base volatility column `{col}` of {path}")

    # 10. Volatilidades de contexto (contienen "_vol_") también no negativas
    ctx_vol_cols = [col for col in expected_context_columns if "_vol_" in col]
    for col in ctx_vol_cols:
        if (pd.to_numeric(df[col], errors="coerce").dropna() < 0).any():
            issues.append(f"[CTX] Negative values found in context volatility column `{col}` of {path}")

    # 11. Volatilidad relativa específica (no negativa)
    if "ctx_rel_vol_20d_vs_equity_proxy" in df.columns:
        rel_vol = pd.to_numeric(df["ctx_rel_vol_20d_vs_equity_proxy"], errors="coerce").dropna()
        if (rel_vol < 0).any():
            issues.append(
                f"[CTX] Negative values found in `ctx_rel_vol_20d_vs_equity_proxy` of {path}"
            )

    # 12. Correlación dentro de [-1, 1]
    if "ctx_corr_20d_vs_equity_proxy" in df.columns:
        corr = pd.to_numeric(df["ctx_corr_20d_vs_equity_proxy"], errors="coerce").dropna()
        if ((corr < -1.0) | (corr > 1.0)).any():
            issues.append(
                f"[CTX] Values outside [-1, 1] found in `ctx_corr_20d_vs_equity_proxy` of {path}"
            )

    # 13. Spreads específicos sin infinitos
    if "ctx_equity_duration_ret_5d_spread" in df.columns:
        spread = pd.to_numeric(df["ctx_equity_duration_ret_5d_spread"], errors="coerce")
        if np.isinf(spread.to_numpy(dtype=float, copy=True)).any():
            issues.append(f"[CTX] inf/-inf values found in `ctx_equity_duration_ret_5d_spread` of {path}")

    if "ctx_credit_duration_ret_5d_spread" in df.columns:
        spread = pd.to_numeric(df["ctx_credit_duration_ret_5d_spread"], errors="coerce")
        if np.isinf(spread.to_numpy(dtype=float, copy=True)).any():
            issues.append(f"[CTX] inf/-inf values found in `ctx_credit_duration_ret_5d_spread` of {path}")

    # 14. Cada archivo debe contener exactamente un instrument_id
    unique_instruments = df["instrument_id"].dropna().unique().tolist()
    if len(unique_instruments) != 1:
        issues.append(
            f"[CTX] Expected exactly one instrument_id per file in {path}, found {unique_instruments}"
        )
    else:
        instrument_id = unique_instruments[0]
        symbol = instrument_id_to_symbol(instrument_id)

        role_map = settings["context_features"]["role_map"]
        equity_proxy_symbol = role_map["equity_proxy"].upper()

        # 15. Para el propio equity proxy, la columna de correlación debe ser toda NaN
        #    (evita autocorrelación espuria)
        if "ctx_corr_20d_vs_equity_proxy" in df.columns and symbol == equity_proxy_symbol:
            non_null_corr = df["ctx_corr_20d_vs_equity_proxy"].notna().any()
            if non_null_corr:
                issues.append(
                    f"[CTX] Expected all NaN in `ctx_corr_20d_vs_equity_proxy` for equity proxy file {path}"
                )

    return issues


# ----------------------------------------------------------------------------
# ORQUESTADOR PRINCIPAL
# ----------------------------------------------------------------------------

def main() -> None:
    """Carga settings, descubre archivos de contexto y ejecuta validaciones."""
    settings = load_settings()

    context_features_root = Path(
        settings["paths"].get("context_features_path", "data/features_context")
    )
    symbols = settings["data"]["universe"]

    # Descubre todos los archivos Parquet de contexto
    context_feature_files = discover_context_feature_files(
        context_features_root=context_features_root,
        symbols=symbols,
    )

    issues: list[str] = []

    if not context_feature_files:
        issues.append(
            f"[CTX] No context feature parquet files discovered under {context_features_root}"
        )

    # Valida cada archivo y acumula problemas
    for path in context_feature_files:
        issues.extend(check_context_feature_file(path, settings))

    # Reporte final
    if issues:
        print("CONTEXT FEATURE QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)   # Salida con error

    print("CONTEXT FEATURE QUALITY CHECKS: PASS")
    for path in context_feature_files:
        print(f"[CTX] OK -> {path}")


if __name__ == "__main__":
    main()