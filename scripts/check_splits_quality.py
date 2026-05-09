from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from quant_platform.evaluation import validate_split_output
from quant_platform.services.settings import load_settings


def discover_split_files(splits_root: Path, symbols: list[str]) -> list[Path]:
    """
    Descubre todos los archivos Parquet de splits para los símbolos dados.
    Busca dentro de splits_root/symbol_lower/ y devuelve una lista ordenada.
    """
    files: list[Path] = []

    for symbol in symbols:
        symbol_dir = splits_root / symbol.lower()
        if symbol_dir.exists():
            # Agregar todos los .parquet ordenados alfabéticamente
            files.extend(sorted(symbol_dir.glob("*.parquet")))

    return files


def check_single_split_file(path: Path, settings: dict) -> list[str]:
    """
    Valida un archivo de splits individual:
    - Existe y no está vacío.
    - Pasa la validación estructural con validate_split_output.
    - No tiene duplicados (instrument_id, split_id).
    - Los split_id son secuenciales (split_001, split_002, ...).
    - train_rows, validation_rows (si requiere), test_rows son numéricos > 0.
    - train_rows es no decreciente a través de los folds.
    Retorna una lista de problemas encontrados (vacía si todo está bien).
    """
    issues: list[str] = []

    if not path.exists():
        return [f"[SPLITS] Missing file: {path}"]

    df = pd.read_parquet(path)

    if df.empty:
        return [f"[SPLITS] Empty dataframe in {path}"]

    # Validación básica de columnas y relaciones temporales
    try:
        validate_split_output(df, settings)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"[SPLITS] Validation failed for {path}: {exc}")
        return issues

    # Verificar duplicados de (instrumento, split_id)
    if df.duplicated(subset=["instrument_id", "split_id"]).any():
        issues.append(f"[SPLITS] Duplicate instrument_id/split_id rows in {path}")

    # Validaciones específicas por instrumento
    for instrument_id, g in df.groupby("instrument_id", sort=True):
        g = g.sort_values(["train_start", "test_start"]).reset_index(drop=True)

        # No deben existir fechas nulas
        if g["train_start"].isna().any():
            issues.append(f"[SPLITS] Null train_start values in {path} for {instrument_id}")

        if g["test_start"].isna().any():
            issues.append(f"[SPLITS] Null test_start values in {path} for {instrument_id}")

        # Los split_id deben ser secuenciales: split_001, split_002, ...
        split_ids = g["split_id"].tolist()
        expected_split_ids = [f"split_{i:03d}" for i in range(1, len(split_ids) + 1)]
        if split_ids != expected_split_ids:
            issues.append(
                f"[SPLITS] Non-sequential split_id values in {path} for {instrument_id}: {split_ids}"
            )

        # Convertir a numérico y validar rangos
        train_rows = pd.to_numeric(g["train_rows"], errors="coerce")
        validation_rows = pd.to_numeric(g["validation_rows"], errors="coerce")
        test_rows = pd.to_numeric(g["test_rows"], errors="coerce")

        if train_rows.isna().any() or (train_rows <= 0).any():
            issues.append(f"[SPLITS] Invalid train_rows values in {path} for {instrument_id}")

        if settings["splits"]["require_validation"]:
            if validation_rows.isna().any() or (validation_rows <= 0).any():
                issues.append(
                    f"[SPLITS] Invalid validation_rows values in {path} for {instrument_id}"
                )

        if test_rows.isna().any() or (test_rows <= 0).any():
            issues.append(f"[SPLITS] Invalid test_rows values in {path} for {instrument_id}")

        # Verificar que train_rows nunca disminuya (expanding window)
        if len(g) >= 2:
            diffs = g["train_rows"].diff().dropna()
            if (diffs < 0).any():
                issues.append(
                    f"[SPLITS] train_rows is not non-decreasing across folds in {path} for {instrument_id}"
                )

    return issues


def main() -> None:
    """
    Orquestador de validación de calidad de splits:
    1. Carga configuración.
    2. Descubre todos los archivos de splits en artifacts/evaluations/splits.
    3. Valida cada archivo individualmente.
    4. Acumula problemas y reporta.
    5. Termina con código de salida 0 si todo está bien, 1 si hay errores.
    """
    settings = load_settings()

    splits_root = Path("artifacts/evaluations/splits")
    symbols = settings["data"]["universe"]

    split_files = discover_split_files(
        splits_root=splits_root,
        symbols=symbols,
    )

    issues: list[str] = []

    if not split_files:
        issues.append(f"[SPLITS] No split parquet files discovered under {splits_root}")

    for path in split_files:
        issues.extend(check_single_split_file(path, settings))

    if issues:
        print("SPLIT QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    print("SPLIT QUALITY CHECKS: PASS")
    for path in split_files:
        print(f"[SPLITS] OK -> {path}")


if __name__ == "__main__":
    main()