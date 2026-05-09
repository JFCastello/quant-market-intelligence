from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from quant_platform.targets import validate_regime_split_output
from quant_platform.services.settings import load_settings


def discover_regime_split_files(
    regime_root: Path,
    symbols: list[str],
) -> list[Path]:
    """
    Descubre todos los archivos Parquet de régimen por split para los símbolos dados.
    Busca dentro de regime_root/symbol_lower/ y devuelve una lista ordenada.
    """
    files: list[Path] = []

    for symbol in symbols:
        symbol_dir = regime_root / symbol.lower()
        if symbol_dir.exists():
            # Agregar todos los .parquet ordenados alfabéticamente
            files.extend(sorted(symbol_dir.glob("*.parquet")))

    return files


def check_single_regime_split_file(path: Path, settings: dict) -> list[str]:
    """
    Valida un archivo individual de régimen por split:
    - Existe y no está vacío.
    - Pasa la validación estructural con validate_regime_split_output.
    - No tiene duplicados (split_id, instrument_id, date).
    - Los split_id son secuenciales por instrumento.
    - Los roles (train/validation/test) son válidos y están presentes según configuración.
    - Los umbrales (threshold_low, threshold_high) no son nulos y están ordenados.
    - La fuente de umbrales coincide con la configuración.
    Retorna una lista de problemas encontrados (vacía si todo está bien).
    """
    issues: list[str] = []

    if not path.exists():
        return [f"[REGIME_SPLIT] Missing file: {path}"]

    df = pd.read_parquet(path)

    if df.empty:
        return [f"[REGIME_SPLIT] Empty dataframe in {path}"]

    # Validación básica de estructura y reglas de negocio
    try:
        validate_regime_split_output(
            regime_split_df=df,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"[REGIME_SPLIT] Validation failed for {path}: {exc}")
        return issues

    # Verificar duplicados por clave compuesta
    if df.duplicated(subset=["split_id", "instrument_id", "date"]).any():
        issues.append(
            f"[REGIME_SPLIT] Duplicate split_id/instrument_id/date rows in {path}"
        )

    # Validaciones específicas por instrumento
    for instrument_id, g in df.groupby("instrument_id", sort=True):
        g = g.sort_values(["split_id", "date"]).reset_index(drop=True)

        # Los split_id deben ser secuenciales (split_001, split_002, ...)
        split_ids = g["split_id"].drop_duplicates().tolist()
        expected_split_ids = [f"split_{i:03d}" for i in range(1, len(split_ids) + 1)]
        if split_ids != expected_split_ids:
            issues.append(
                f"[REGIME_SPLIT] Non-sequential split_id values in {path} for "
                f"{instrument_id}: {split_ids}"
            )

        # Validar roles del dataset
        observed_roles = set(g["dataset_role"].dropna().unique().tolist())
        allowed_roles = {"train", "validation", "test"}
        invalid_roles = observed_roles - allowed_roles
        if invalid_roles:
            issues.append(
                f"[REGIME_SPLIT] Invalid dataset_role values in {path} for "
                f"{instrument_id}: {sorted(invalid_roles)}"
            )

        # Si la configuración requiere validación, debe haber filas de validación
        if settings["splits"]["require_validation"] and "validation" not in observed_roles:
            issues.append(
                f"[REGIME_SPLIT] Missing validation rows in {path} for {instrument_id}"
            )

        # Siempre deben existir entrenamiento y prueba
        if "train" not in observed_roles:
            issues.append(
                f"[REGIME_SPLIT] Missing train rows in {path} for {instrument_id}"
            )

        if "test" not in observed_roles:
            issues.append(
                f"[REGIME_SPLIT] Missing test rows in {path} for {instrument_id}"
            )

        # Validar umbrales (no nulos y ordenados)
        if g["threshold_low"].isna().any():
            issues.append(
                f"[REGIME_SPLIT] Null threshold_low values in {path} for {instrument_id}"
            )

        if g["threshold_high"].isna().any():
            issues.append(
                f"[REGIME_SPLIT] Null threshold_high values in {path} for {instrument_id}"
            )

        bad_threshold_order = g["threshold_low"] > g["threshold_high"]
        if bad_threshold_order.any():
            issues.append(
                f"[REGIME_SPLIT] threshold_low > threshold_high in {path} for {instrument_id}"
            )

        # La fuente de umbrales debe coincidir con la configuración global
        if (g["regime_thresholds_source"] != settings["splits"]["regime_thresholds_source"]).any():
            issues.append(
                f"[REGIME_SPLIT] Invalid regime_thresholds_source values in {path} for "
                f"{instrument_id}"
            )

    return issues


def main() -> None:
    """
    Orquestador de validación de calidad de los targets de régimen por split:
    1. Carga configuración.
    2. Descubre todos los archivos de régimen en artifacts/evaluations/regime_targets.
    3. Valida cada archivo individualmente.
    4. Acumula problemas y reporta.
    5. Termina con código de salida 0 si todo está bien, 1 si hay errores.
    """
    settings = load_settings()

    regime_root = Path("artifacts/evaluations/regime_targets")
    symbols = settings["data"]["universe"]

    regime_files = discover_regime_split_files(
        regime_root=regime_root,
        symbols=symbols,
    )

    issues: list[str] = []

    if not regime_files:
        issues.append(
            f"[REGIME_SPLIT] No regime target parquet files discovered under {regime_root}"
        )

    for path in regime_files:
        issues.extend(check_single_regime_split_file(path, settings))

    if issues:
        print("REGIME TARGETS BY SPLIT QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    print("REGIME TARGETS BY SPLIT QUALITY CHECKS: PASS")
    for path in regime_files:
        print(f"[REGIME_SPLIT] OK -> {path}")


if __name__ == "__main__":
    main()