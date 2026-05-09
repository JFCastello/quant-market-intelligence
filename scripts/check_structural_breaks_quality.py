from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from quant_platform.services.settings import load_settings


# Conjunto mínimo de columnas que todo artefacto de señal debe contener,
# independientemente de la configuración específica del pipeline.
#
# "instrument_id":
#     identificador interno único del activo/instrumento.
#
# "date":
#     fecha de la observación temporal dentro de la señal.
SIGNAL_BASE_REQUIRED_COLS = {
    "instrument_id",
    "date",
}


# Conjunto de columnas obligatorias para el artefacto de eventos de
# structural breaks.
#
# Este contrato define la estructura mínima que debe tener el parquet
# de eventos para considerarse válido.
EVENTS_REQUIRED_COLS = {
    "event_id",
    "break_version",
    "instrument_id",
    "symbol",
    "breakpoint_index",
    "break_date",
    "previous_segment_start_date",
    "previous_segment_end_date",
    "next_segment_start_date",
    "next_segment_end_date",
    "method",
    "algorithm",
    "cost_model",
    "penalty",
    "min_size",
    "jump",
    "signal_row_count",
    "detected_at",
}


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: set[str],
    df_name: str,
    path: Path,
) -> list[str]:
    """
    Verifica que un DataFrame contenga todas las columnas obligatorias.

    Propósito:
    Esta función encapsula la validación estructural más básica del pipeline:
    confirmar que el DataFrame tenga los campos mínimos necesarios antes de
    aplicar controles más específicos.

    Parámetros:
    - df:
      DataFrame a revisar.
    - required_cols:
      conjunto de nombres de columnas que deben estar presentes.
    - df_name:
      nombre lógico del tipo de DataFrame, por ejemplo "SIGNAL" o "EVENTS".
      Se usa únicamente para construir mensajes de error más claros.
    - path:
      ruta del archivo que originó el DataFrame, útil para trazabilidad.

    Retorna:
    - Lista de issues encontrados.
      Si no falta ninguna columna, retorna una lista vacía.

    Diseño:
    En lugar de lanzar excepción, esta función acumula hallazgos.
    Eso permite que el chequeador general reporte múltiples problemas en
    una sola corrida.
    """
    issues: list[str] = []

    # Calcula la diferencia entre las columnas obligatorias y las columnas
    # reales del DataFrame.
    missing = sorted(required_cols - set(df.columns))

    if missing:
        issues.append(f"[{df_name}] Missing columns in {path}: {missing}")

    return issues


def discover_single_file_per_symbol(
    root_dir: Path,
    file_glob_pattern: str,
) -> dict[str, Path]:
    """
    Descubre exactamente un archivo por símbolo usando un patrón glob completo.

    Propósito:
    Este helper recorre una carpeta raíz y localiza archivos de artefactos
    organizados por símbolo, asegurándose de que exista como máximo uno por
    símbolo para el patrón solicitado.

    Parámetros:
    - root_dir:
      carpeta raíz bajo la cual se buscarán los archivos.
    - file_glob_pattern:
      patrón glob completo para buscar los archivos deseados.
      Ejemplos:
        "*/*_structural_break_signal_*.parquet"
        "*/*_structural_break_events_*.parquet"

    Retorna:
    - Un diccionario:
        {symbol: path}
      donde la clave es el símbolo inferido del nombre de la carpeta padre
      y el valor es la ruta del archivo encontrado.

    Diferencia importante respecto a la versión anterior:
    Antes la función recibía solo un sufijo y construía internamente el glob.
    En esta versión recibe directamente el patrón completo, lo cual la vuelve
    más flexible y reutilizable.

    Regla de integridad:
    Si para un mismo símbolo se encuentra más de un archivo que coincide
    con el patrón, se lanza RuntimeError. Esto evita ambigüedad sobre cuál
    artefacto debe validarse.
    """
    file_map: dict[str, Path] = {}

    # Busca todos los archivos que coincidan con el patrón dado dentro de
    # la raíz especificada.
    for path in sorted(root_dir.glob(file_glob_pattern)):
        symbol = path.parent.name

        # Si ya se había registrado un archivo para este símbolo, entonces
        # hay duplicidad y el pipeline pierde unicidad/reproducibilidad.
        if symbol in file_map:
            raise RuntimeError(
                f"Expected exactly one file per symbol for pattern={file_glob_pattern}, duplicate found for {symbol}"
            )

        file_map[symbol] = path

    return file_map


def check_signal_df(
    df: pd.DataFrame,
    path: Path,
    return_signal_column: str,
    volatility_signal_column: str,
    standardize_joint_signal: bool,
    min_required_rows: int,
) -> list[str]:
    """
    Ejecuta controles de calidad internos sobre un DataFrame de señal.

    Propósito:
    Confirmar que el artefacto de señal:
    - tenga las columnas esperadas,
    - no esté vacío,
    - tenga una longitud suficiente,
    - represente un solo instrumento,
    - tenga fechas parseables y ordenadas,
    - no conserve nulos indebidos,
    - no contenga duplicados por instrumento/fecha.

    Parámetros:
    - df:
      DataFrame de señal.
    - path:
      ruta del parquet leído.
    - return_signal_column:
      nombre de la columna de retornos esperada.
    - volatility_signal_column:
      nombre de la columna de volatilidad esperada.
    - standardize_joint_signal:
      indica si deben existir columnas estandarizadas tipo z-score.
    - min_required_rows:
      número mínimo de filas exigido por configuración.

    Retorna:
    - Lista de issues detectados.
    """
    issues: list[str] = []

    # Construye dinámicamente el conjunto de columnas requeridas según la
    # configuración actual del pipeline.
    required_cols = set(SIGNAL_BASE_REQUIRED_COLS) | {
        return_signal_column,
        volatility_signal_column,
    }

    # Si el pipeline trabaja con señal conjunta estandarizada, también deben
    # existir las columnas z_... derivadas.
    if standardize_joint_signal:
        required_cols |= {
            f"z_{return_signal_column}",
            f"z_{volatility_signal_column}",
        }

    # Valida estructura mínima.
    issues.extend(validate_required_columns(df, required_cols, "SIGNAL", path))
    if issues:
        return issues

    # Un artefacto vacío se considera inválido para señal.
    if df.empty:
        issues.append(f"[SIGNAL] Empty dataframe: {path}")
        return issues

    # Verifica que la cantidad de filas supere el umbral mínimo requerido
    # por la configuración del pipeline.
    if len(df) < min_required_rows:
        issues.append(
            f"[SIGNAL] Row count below configured minimum in {path}: {len(df)} < {min_required_rows}"
        )

    # instrument_id no debería contener nulos.
    if df["instrument_id"].isna().any():
        issues.append(f"[SIGNAL] Null instrument_id values found in {path}")

    # Cada archivo de señal debe representar exactamente un instrumento.
    instrument_ids = sorted(df["instrument_id"].dropna().unique().tolist())
    if len(instrument_ids) != 1:
        issues.append(f"[SIGNAL] Expected exactly one instrument_id in {path}, got {instrument_ids}")

    # Intenta convertir la columna date a datetime. Si falla, hay valores
    # mal formados. Si no falla, además se exige orden temporal ascendente.
    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().any():
        issues.append(f"[SIGNAL] Unparseable date values found in {path}")
    elif not date_series.is_monotonic_increasing:
        issues.append(f"[SIGNAL] Dates are not sorted ascending in {path}")

    # Las columnas de señal principales no deberían conservar nulos al final
    # del preprocesamiento.
    signal_cols = [return_signal_column, volatility_signal_column]
    if df[signal_cols].isna().any().any():
        issues.append(f"[SIGNAL] Null signal values remain after preprocessing in {path}")

    # Si el pipeline creó columnas estandarizadas, también se exige que no
    # tengan nulos.
    if standardize_joint_signal:
        z_cols = [f"z_{return_signal_column}", f"z_{volatility_signal_column}"]
        if df[z_cols].isna().any().any():
            issues.append(f"[SIGNAL] Null standardized signal values found in {path}")

    # No debería haber dos filas para la misma combinación instrumento-fecha.
    if df.duplicated(subset=["instrument_id", "date"]).any():
        issues.append(f"[SIGNAL] Duplicate instrument_id/date rows found in {path}")

    return issues


def check_events_df(
    df: pd.DataFrame,
    path: Path,
    algorithm: str,
    cost_model: str,
) -> list[str]:
    """
    Ejecuta controles de calidad internos sobre el DataFrame de eventos.

    Propósito:
    Confirmar que el artefacto de eventos:
    - tenga la estructura requerida,
    - mantenga unicidad de event_id,
    - represente un solo instrumento y símbolo,
    - refleje correctamente el método configurado,
    - tenga columnas temporales parseables,
    - respete relaciones cronológicas lógicas entre segmentos,
    - contenga valores válidos en índices y conteos.

    Parámetros:
    - df:
      DataFrame de eventos.
    - path:
      ruta del parquet leído.
    - algorithm:
      algoritmo esperado desde la configuración.
    - cost_model:
      modelo de costo esperado desde la configuración.

    Retorna:
    - Lista de issues detectados.

    Nota:
    Un DataFrame de eventos vacío puede ser válido. Significa simplemente
    que no se detectaron quiebres para ese símbolo.
    """
    issues: list[str] = []

    # Primero valida la presencia de columnas obligatorias.
    issues.extend(validate_required_columns(df, EVENTS_REQUIRED_COLS, "EVENTS", path))
    if issues:
        return issues

    # Si no hubo eventos detectados, no se considera error estructural.
    if df.empty:
        return issues

    # event_id debe existir para cada evento.
    if df["event_id"].isna().any():
        issues.append(f"[EVENTS] Null event_id values found in {path}")

    # event_id además debe ser único.
    if df["event_id"].duplicated().any():
        issues.append(f"[EVENTS] Duplicate event_id values found in {path}")

    # Cada archivo de eventos debería corresponder a un solo instrumento.
    instrument_ids = sorted(df["instrument_id"].dropna().unique().tolist())
    if len(instrument_ids) != 1:
        issues.append(f"[EVENTS] Expected exactly one instrument_id in {path}, got {instrument_ids}")

    # Cada archivo de eventos debería corresponder a un solo símbolo.
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    if len(symbols) != 1:
        issues.append(f"[EVENTS] Expected exactly one symbol in {path}, got {symbols}")

    # Se espera que el método sea la composición de algoritmo y cost_model.
    expected_method = f"{algorithm}_{cost_model}"

    if set(df["algorithm"].dropna().unique().tolist()) != {algorithm}:
        issues.append(f"[EVENTS] Unexpected algorithm values in {path}")

    if set(df["cost_model"].dropna().unique().tolist()) != {cost_model}:
        issues.append(f"[EVENTS] Unexpected cost_model values in {path}")

    if set(df["method"].dropna().unique().tolist()) != {expected_method}:
        issues.append(f"[EVENTS] Unexpected method values in {path}")

    # Lista de columnas que deben poder interpretarse como datetime.
    date_cols = [
        "break_date",
        "previous_segment_start_date",
        "previous_segment_end_date",
        "next_segment_start_date",
        "next_segment_end_date",
        "detected_at",
    ]

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().any():
            issues.append(f"[EVENTS] Unparseable datetime values in column `{col}` of {path}")

    # Los eventos deberían venir ordenados cronológicamente por break_date.
    break_dates = pd.to_datetime(df["break_date"], errors="coerce")
    if not break_dates.is_monotonic_increasing:
        issues.append(f"[EVENTS] break_date is not sorted ascending in {path}")

    # Convierte explícitamente las fechas de segmentos para verificar reglas
    # temporales entre ellas.
    prev_start = pd.to_datetime(df["previous_segment_start_date"], errors="coerce")
    prev_end = pd.to_datetime(df["previous_segment_end_date"], errors="coerce")
    next_start = pd.to_datetime(df["next_segment_start_date"], errors="coerce")
    next_end = pd.to_datetime(df["next_segment_end_date"], errors="coerce")

    # Ningún segmento puede comenzar después de terminar.
    if not (prev_start <= prev_end).all():
        issues.append(f"[EVENTS] Found previous segment start > end in {path}")

    if not (next_start <= next_end).all():
        issues.append(f"[EVENTS] Found next segment start > end in {path}")

    # El segmento previo debe cerrar antes de que el siguiente comience.
    if not (prev_end < next_start).all():
        issues.append(f"[EVENTS] Found previous segment end >= next segment start in {path}")

    # En este esquema, la fecha de quiebre coincide con el inicio del
    # siguiente segmento.
    if not (break_dates == next_start).all():
        issues.append(f"[EVENTS] break_date does not match next_segment_start_date in {path}")

    # El índice del breakpoint debe ser positivo.
    if (df["breakpoint_index"].astype(int) <= 0).any():
        issues.append(f"[EVENTS] Non-positive breakpoint_index values found in {path}")

    # El número de filas de la señal reportado como metadato también debe
    # ser positivo.
    if (df["signal_row_count"].astype(int) <= 0).any():
        issues.append(f"[EVENTS] Non-positive signal_row_count values found in {path}")

    return issues


def check_cross_consistency(
    signal_df: pd.DataFrame,
    events_df: pd.DataFrame,
    symbol: str,
) -> list[str]:
    """
    Verifica consistencia cruzada entre el DataFrame de señal y el DataFrame
    de eventos del mismo símbolo.

    Propósito:
    Asegurar que ambos artefactos no solo sean válidos por separado, sino
    compatibles entre sí.

    Se revisa específicamente que:
    - ambos hablen del mismo instrument_id,
    - las fechas usadas por los eventos existan realmente en la señal,
    - signal_row_count coincida con el tamaño real de signal_df,
    - breakpoint_index no quede fuera del rango permitido.

    Parámetros:
    - signal_df:
      DataFrame de señal.
    - events_df:
      DataFrame de eventos.
    - symbol:
      símbolo actual, usado para generar mensajes claros.

    Retorna:
    - Lista de issues detectados.
    """
    issues: list[str] = []

    # Primero asegura que signal_df representa exactamente un instrumento.
    signal_instrument_ids = sorted(signal_df["instrument_id"].dropna().unique().tolist())
    if len(signal_instrument_ids) != 1:
        issues.append(f"[CROSS] Signal dataframe has unexpected instrument_ids for symbol={symbol}")
        return issues

    # Si no hay eventos, no hay validaciones cruzadas adicionales que hacer.
    if events_df.empty:
        return issues

    # Verifica que el instrument_id en eventos coincida con el de la señal.
    event_instrument_ids = sorted(events_df["instrument_id"].dropna().unique().tolist())
    if signal_instrument_ids != event_instrument_ids:
        issues.append(f"[CROSS] Signal/events instrument_id mismatch for symbol={symbol}")

    # Conjunto de fechas realmente presentes en la señal.
    signal_dates = set(pd.to_datetime(signal_df["date"]).tolist())

    # Todas las fechas referenciadas por eventos deberían pertenecer a la
    # línea temporal de signal_df.
    for col in [
        "break_date",
        "previous_segment_start_date",
        "previous_segment_end_date",
        "next_segment_start_date",
        "next_segment_end_date",
    ]:
        event_dates = set(pd.to_datetime(events_df[col]).tolist())
        if not event_dates.issubset(signal_dates):
            issues.append(f"[CROSS] Event column `{col}` contains dates not present in signal_df for symbol={symbol}")

    # Cada evento guarda signal_row_count como metadato; ese valor debe
    # coincidir exactamente con el número de filas de signal_df.
    signal_row_count_values = set(events_df["signal_row_count"].astype(int).tolist())
    if signal_row_count_values != {len(signal_df)}:
        issues.append(f"[CROSS] signal_row_count mismatch for symbol={symbol}")

    # El breakpoint_index no debe apuntar más allá del final de la señal.
    if (events_df["breakpoint_index"].astype(int) >= len(signal_df)).any():
        issues.append(f"[CROSS] breakpoint_index out of bounds for symbol={symbol}")

    return issues


def main() -> None:
    """
    Orquesta la validación completa de los artefactos de structural breaks.

    Propósito global:
    Ejecutar un control de calidad de extremo a extremo sobre los parquets
    de señal y eventos generados por el pipeline.

    Flujo:
    1. Cargar settings.
    2. Obtener la configuración de structural_breaks.
    3. Resolver la carpeta raíz donde viven los artefactos.
    4. Descubrir archivos de señal y eventos mediante patrones glob.
    5. Verificar que existan artefactos y que el conjunto de símbolos coincida.
    6. Para cada símbolo:
       - leer señal,
       - leer eventos,
       - validar señal,
       - validar eventos,
       - validar consistencia cruzada.
    7. Si hay issues:
       - imprimir FAIL,
       - listar todos los problemas,
       - terminar con exit code 1.
    8. Si no hay issues:
       - imprimir PASS,
       - listar artefactos validados.

    Importancia:
    Este script funciona como compuerta de calidad del pipeline. Su objetivo
    no es producir artefactos, sino certificar que los ya producidos sean
    estructural y lógicamente coherentes.
    """
    settings = load_settings()
    cfg = settings["structural_breaks"]

    # Directorio raíz donde se espera encontrar los outputs del paso de
    # structural breaks.
    events_root = Path(cfg["outputs"]["events_dir"])

    # Descubre un archivo de señal por símbolo.
    signal_files = discover_single_file_per_symbol(
        events_root,
        "*/*_structural_break_signal_*.parquet",
    )

    # Descubre un archivo de eventos por símbolo.
    event_files = discover_single_file_per_symbol(
        events_root,
        "*/*_structural_break_events_*.parquet",
    )

    # Acumulador central de problemas encontrados durante la validación.
    issues: list[str] = []

    # Une el conjunto de símbolos descubiertos en uno u otro tipo de artefacto.
    all_symbols = sorted(set(signal_files) | set(event_files))

    # Si no se encontró ningún artefacto, el chequeo falla inmediatamente.
    if not all_symbols:
        print("STRUCTURAL BREAK QUALITY CHECKS: FAIL")
        print("[DISCOVERY] No structural break artifacts found.")
        sys.exit(1)

    # Señales y eventos deberían cubrir exactamente el mismo conjunto de símbolos.
    if set(signal_files) != set(event_files):
        issues.append("[DISCOVERY] Symbol sets do not match across signal/events artifacts.")

    for symbol in all_symbols:
        signal_path = signal_files.get(symbol)
        events_path = event_files.get(symbol)

        # Si falta el artefacto de señal para un símbolo, se reporta y se sigue.
        if signal_path is None:
            issues.append(f"[DISCOVERY] Missing signal artifact for symbol={symbol}")
            continue

        # Si falta el artefacto de eventos para un símbolo, se reporta y se sigue.
        if events_path is None:
            issues.append(f"[DISCOVERY] Missing events artifact for symbol={symbol}")
            continue

        # Carga ambos parquets para validarlos.
        signal_df = pd.read_parquet(signal_path)
        events_df = pd.read_parquet(events_path)

        # Ejecuta validaciones internas de la señal.
        issues.extend(
            check_signal_df(
                df=signal_df,
                path=signal_path,
                return_signal_column=cfg["signals"]["return_signal_column"],
                volatility_signal_column=cfg["signals"]["volatility_signal_column"],
                standardize_joint_signal=cfg["preprocessing"]["standardize_joint_signal"],
                min_required_rows=cfg["preprocessing"]["min_required_rows"],
            )
        )

        # Ejecuta validaciones internas de los eventos.
        issues.extend(
            check_events_df(
                df=events_df,
                path=events_path,
                algorithm=cfg["method"]["algorithm"],
                cost_model=cfg["method"]["cost_model"],
            )
        )

        # Ejecuta validaciones cruzadas entre ambos artefactos.
        issues.extend(
            check_cross_consistency(
                signal_df=signal_df,
                events_df=events_df,
                symbol=symbol,
            )
        )

    # Si hubo al menos un problema, imprime todos los hallazgos y retorna
    # código de salida 1 para marcar fallo del chequeo.
    if issues:
        print("STRUCTURAL BREAK QUALITY CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    # Si no hubo problemas, marca el chequeo como exitoso y enumera qué
    # archivos fueron validados.
    print("STRUCTURAL BREAK QUALITY CHECKS: PASS")
    for symbol in sorted(all_symbols):
        print(f"[SIGNAL] OK -> {signal_files[symbol]}")
        print(f"[EVENTS] OK -> {event_files[symbol]}")


if __name__ == "__main__":
    main()