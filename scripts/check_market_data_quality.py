from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

#-opcion 1, <castello>-
# RAW_FILES = [
#     Path("data/raw/spy/spy_2026-02-01_2026-03-01_raw.json"),
#     Path("data/raw/tlt/tlt_2026-02-01_2026-03-01_raw.json"),
# ]
# 
# NORMALIZED_FILES = [
#     Path("data/normalized/spy/spy_2026-02-01_2026-03-01_daily_bars.parquet"),
#     Path("data/normalized/tlt/tlt_2026-02-01_2026-03-01_daily_bars.parquet"),
# ]

#-opcion 2, <castello>-
def discover_raw_files() -> list[Path]:
    return sorted(Path("data/raw").glob("*/*.json"))


def discover_normalized_files() -> list[Path]:
    return sorted(Path("data/normalized").glob("*/*.parquet"))

RAW_FILES        = discover_raw_files()         #-por orden, estas dos intrucciones no deberian estar aca (funciones 
NORMALIZED_FILES = discover_normalized_files()  #  agrupadas al inicio del doc i,e solo funciones es lo ideal 
                                                #  y no entre mezclar con otras cosas), pero el igual poner esto aca, lo 
                                                #  consideramos mas claro como propuesta o en sus intenciones, por asi 
                                                #  decirlo... asi se comenten o no esas dos lineas, <castello>-

#-opcion 3: una funcion que admita como argumentos que direcciones se quiere mirar, si no se especifica cuales
#  entonces se mira todas por defecto dentro del archivo y ya, <castello>-

def check_raw_file(path: Path) -> list[str]:
    issues: list[str] = []

    if not path.exists():
        return [f"[RAW] Missing file: {path}"]

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    values = payload.get("values", [])

    if not meta:
        issues.append(f"[RAW] Missing meta in {path}")

    if not values:
        issues.append(f"[RAW] No values in {path}")

    return issues


def check_normalized_file(path: Path) -> list[str]:
    issues: list[str] = []

    if not path.exists():
        return [f"[NORM] Missing file: {path}"]

    df = pd.read_parquet(path)

    required_cols = [
        "instrument_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "provider",
        "ingested_at",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"[NORM] Missing columns in {path}: {missing_cols}")
        return issues

    if df.empty:
        issues.append(f"[NORM] Empty dataframe in {path}")
        return issues

    if df["instrument_id"].isna().any():
        issues.append(f"[NORM] Null instrument_id in {path}")

    if df["date"].isna().any():
        issues.append(f"[NORM] Null date in {path}")

    if df["provider"].isna().any():
        issues.append(f"[NORM] Null provider in {path}")

    if df["ingested_at"].isna().any():
        issues.append(f"[NORM] Null ingested_at in {path}")
    
    if not pd.api.types.is_datetime64_any_dtype(df["ingested_at"]):
        issues.append(f"[NORM] `ingested_at` is not a pandas datetime dtype in {path}")

    if df.duplicated(subset=["instrument_id", "date"]).any():
        issues.append(f"[NORM] Duplicate instrument_id/date rows in {path}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        issues.append(f"[NORM] `date` is not a pandas datetime dtype in {path}")

    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().any():
        issues.append(f"[NORM] Unparseable date values in {path}")
    else:
        if not date_series.is_monotonic_increasing:
            issues.append(f"[NORM] Dates are not sorted ascending in {path}")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if df[col].isna().any():
            issues.append(f"[NORM] Nulls in numeric column `{col}` of {path}")

    if (df["open"] <= 0).any():
        issues.append(f"[NORM] Non-positive open values in {path}")
    if (df["high"] <= 0).any():
        issues.append(f"[NORM] Non-positive high values in {path}")
    if (df["low"] <= 0).any():
        issues.append(f"[NORM] Non-positive low values in {path}")
    if (df["close"] <= 0).any():
        issues.append(f"[NORM] Non-positive close values in {path}")
    if (df["volume"] < 0).any():
        issues.append(f"[NORM] Negative volume values in {path}")

    if (df["high"] < df["low"]).any():
        issues.append(f"[NORM] Found rows with high < low in {path}")

    return issues


def main() -> None:
    #RAW_FILES = discover_raw_files()                #-aca en principio deberian ir por orden, <castello>-
    #NORMALIZED_FILES = discover_normalized_files()

    issues: list[str] = []

    if not RAW_FILES:
        issues.append("[RAW] No active raw files discovered")
    if not NORMALIZED_FILES:
        issues.append("[NORM] No active normalized files discovered")

    for path in RAW_FILES:
        issues.extend(check_raw_file(path))

    for path in NORMALIZED_FILES:
        issues.extend(check_normalized_file(path))

    if issues:
        print("DATA QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    print("DATA QUALITY CHECKS: PASS")
    for path in RAW_FILES:
        print(f"[RAW] OK -> {path}")
    for path in NORMALIZED_FILES:
        print(f"[NORM] OK -> {path}")


if __name__ == "__main__":
    main()
