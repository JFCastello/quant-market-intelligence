from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Esquema de datos de salida (definido en la plataforma cuantitativa)
from quant_platform.schemas import TargetRow

# Columnas mínimas requeridas en el DataFrame de entrada
REQUIRED_TARGET_SOURCE_COLUMNS = [
    "instrument_id",      # Identificador único del activo financiero
    "date",               # Fecha de la observación
    "feature_version",    # Versión de las características utilizadas (para trazabilidad)
    "log_ret_1d",         # Retorno logarítmico diario (base para el cálculo de volatilidad)
]


def _targets_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae la configuración específica de 'targets' del diccionario global de configuración.
    Lanza un error si la sección no existe.
    """
    if "targets" not in settings:
        raise KeyError("Missing `targets` section in settings.")
    return settings["targets"]


def get_enabled_target_columns(settings: dict[str, Any]) -> list[str]:
    """
    Devuelve la lista de columnas que debe tener el DataFrame final,
    en función de qué tipos de target están activos en la configuración.
    """
    cfg = _targets_cfg(settings)

    # Columnas siempre presentes
    columns: list[str] = [
        "instrument_id",
        "date",
        "target_version",   # Versión del target (para control de cambios)
    ]

    # Si el target continuo está habilitado, añadimos su nombre de columna
    if cfg["continuous_target"]["enabled"]:
        columns.append(cfg["continuous_target"]["name"])

    # Si el target de clasificación está habilitado, añadimos su nombre de columna
    if cfg["classification_target"]["enabled"]:
        columns.append(cfg["classification_target"]["name"])

    return columns


def validate_target_source_input(df: pd.DataFrame, settings: dict[str, Any]) -> None:
    """
    Valida que el DataFrame de entrada cumpla con los requisitos mínimos para
    poder calcular los targets.
    """
    cfg = _targets_cfg(settings)

    # Conjunto de columnas obligatorias (las constantes más las condicionales)
    required_cols = set(REQUIRED_TARGET_SOURCE_COLUMNS)

    # Si el target continuo está activo, necesitamos la columna de retorno base definida en configuración
    if cfg["continuous_target"]["enabled"]:
        required_cols.add(cfg["continuous_target"]["base_return_column"])

    # Comprobamos si faltan columnas en el DataFrame
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Target source input is missing required columns: {missing_cols}")

    # El DataFrame no debe estar vacío
    if df.empty:
        raise ValueError("Target source input dataframe is empty.")

    # No puede haber identificadores de instrumento nulos
    if df["instrument_id"].isna().any():
        raise ValueError("Target source input contains null `instrument_id` values.")

    # No puede haber fechas nulas
    if df["date"].isna().any():
        raise ValueError("Target source input contains null `date` values.")

    # No debe haber duplicados por instrumento y fecha (cada activo en una fecha es único)
    duplicated = df.duplicated(subset=["instrument_id", "date"])
    if duplicated.any():
        raise ValueError("Target source input contains duplicate `instrument_id`/`date` rows.")

    # Todas las fechas deben poder convertirse a datetime sin errores
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Target source input contains unparseable `date` values.")


def _sort_target_source_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame de entrada por instrumento y fecha, y convierte la columna 'date'
    a tipo datetime. Devuelve una copia ordenada con índice reseteado.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")  # Conversión estricta
    out = out.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return out


def compute_future_realized_volatility(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """
    Calcula la volatilidad realizada futura (desviación estándar de los retornos
    de los próximos 'horizon_days' días) para cada fila, agrupando por instrumento.
    """
    cfg = _targets_cfg(settings)
    continuous_cfg = cfg["continuous_target"]

    # Si el target continuo no está activo, devolvemos el DataFrame sin modificar
    if not continuous_cfg["enabled"]:
        return df.copy()

    # Extraemos los parámetros de la configuración
    target_name = continuous_cfg["name"]                  # Nombre de la columna a crear
    base_return_column = continuous_cfg["base_return_column"]  # Columna con retornos diarios
    horizon_days = continuous_cfg["horizon_days"]         # Número de días hacia adelante
    min_periods = continuous_cfg["min_periods"]           # Mínimo de observaciones requeridas
    annualization_factor = continuous_cfg["annualization_factor"]  # Factor de anualización (ej: 252)
    allow_partial_window = continuous_cfg["allow_partial_window"]  # Si se permiten ventanas incompletas
    window_type = continuous_cfg["window_type"]           # Tipo de ventana (solo 'fixed_forward' implementado)

    # Verificación de compatibilidad de parámetros
    if window_type != "fixed_forward":
        raise NotImplementedError(f"Unsupported target window_type: {window_type}")

    if not allow_partial_window and min_periods != horizon_days:
        raise ValueError(
            "When `allow_partial_window` is false, `min_periods` must equal `horizon_days`."
        )

    out = df.copy()

    # Función interna que calcula la volatilidad futura para una serie temporal de un único instrumento
    def _future_rv_from_returns(series: pd.Series) -> pd.Series:
        """
        Para cada punto de la serie, toma los siguientes 'horizon_days' retornos
        (excluyendo el valor actual), calcula su desviación estándar muestral (ddof=1)
        y la anualiza. Devuelve una serie del mismo tamaño con NaN donde no se pudo calcular.
        """
        values = series.to_numpy(dtype=float, copy=False)
        n = len(values)
        result = np.full(n, np.nan, dtype=float)

        # Iteramos sobre cada índice de la serie
        for i in range(n):
            # Seleccionamos los retornos desde i+1 hasta i+1+horizon_days (exclusivo)
            future_slice = values[i + 1 : i + 1 + horizon_days]

            # Si no hay suficientes observaciones, omitimos
            if len(future_slice) < min_periods:
                continue

            # Si algún valor es NaN, la ventana no es válida (evita volatilidades sesgadas)
            if np.isnan(future_slice).any():
                continue

            # Cálculo de volatilidad realizada anualizada
            result[i] = float(np.std(future_slice, ddof=1) * np.sqrt(annualization_factor))

        return pd.Series(result, index=series.index)

    # Aplicamos el cálculo agrupado por instrumento (cada activo por separado)
    out[target_name] = (
        out.groupby("instrument_id", group_keys=False)[base_return_column]
        .apply(_future_rv_from_returns)
    )

    return out


def build_continuous_targets(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """
    Función principal para construir el target continuo a partir de los datos fuente.
    Realiza validación, ordenamiento, cálculo y añade la versión del target.
    """
    # Validación inicial de la entrada
    validate_target_source_input(df, settings)

    # Ordenamos y convertimos fechas
    out = _sort_target_source_input(df)

    # Calculamos la volatilidad realizada futura
    out = compute_future_realized_volatility(out, settings)

    # Añadimos la columna 'target_version' con el valor de la configuración
    out["target_version"] = _targets_cfg(settings)["target_version"]
    return out


def adapt_targets_to_contract(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """
    Adapta el DataFrame de targets al contrato definido por el esquema TargetRow.
    Selecciona solo las columnas habilitadas y valida cada fila con Pydantic.
    Devuelve un DataFrame limpio y validado listo para ser usado en etapas posteriores.
    """
    target_columns = get_enabled_target_columns(settings)

    # Verificamos que el DataFrame tenga todas las columnas esperadas
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Target dataframe is missing expected contract columns: {missing_cols}"
        )

    # Seleccionamos solo las columnas relevantes
    contracted = df[target_columns].copy()

    # Convertimos a lista de diccionarios para validación con Pydantic
    records = contracted.to_dict(orient="records")
    validated_records = [
        TargetRow(**record).model_dump(include=set(target_columns))
        for record in records
    ]

    # Reconstruimos DataFrame a partir de los registros validados
    validated_df = pd.DataFrame(validated_records)
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="raise")

    return validated_df


def validate_target_output(df: pd.DataFrame, settings: dict[str, Any]) -> None:
    """
    Validación exhaustiva del DataFrame de salida ya construido.
    Asegura que cumple con todas las expectativas de estructura, tipos y contenido.
    """
    expected_columns = get_enabled_target_columns(settings)
    cfg = _targets_cfg(settings)

    # Columnas presentes
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Target output is missing expected columns: {missing_cols}")

    # No vacío
    if df.empty:
        raise ValueError("Target output dataframe is empty.")

    # Nulos en columnas clave
    if df["instrument_id"].isna().any():
        raise ValueError("Target output contains null `instrument_id` values.")
    if df["date"].isna().any():
        raise ValueError("Target output contains null `date` values.")
    if df["target_version"].isna().any():
        raise ValueError("Target output contains null `target_version` values.")

    # Duplicados
    duplicated = df.duplicated(subset=["instrument_id", "date"])
    if duplicated.any():
        raise ValueError("Target output contains duplicate `instrument_id`/`date` rows.")

    # Fechas parseables
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Target output contains unparseable `date` values.")

    # Ordenamiento correcto
    sorted_df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    current_df = df.reset_index(drop=True)
    if not current_df[["instrument_id", "date"]].equals(sorted_df[["instrument_id", "date"]]):
        raise ValueError("Target output is not sorted by `instrument_id`, `date`.")

    # Versión del target correcta
    expected_version = cfg["target_version"]
    if not (df["target_version"] == expected_version).all():
        raise ValueError(
            f"Target output contains rows with target_version different from `{expected_version}`."
        )

    # Validaciones específicas para target continuo (si está habilitado)
    if cfg["continuous_target"]["enabled"]:
        target_name = cfg["continuous_target"]["name"]

        if target_name not in df.columns:
            raise ValueError(f"Missing continuous target column `{target_name}` in output.")

        target_series = pd.to_numeric(df[target_name], errors="coerce")

        # Se detectan valores no numéricos comparando nulos antes y después de coerción
        if target_series.isna().sum() > df[target_name].isna().sum():
            raise ValueError(f"Non-numeric values found in `{target_name}`.")

        # No puede haber infinitos
        inf_mask = np.isinf(target_series.to_numpy(dtype=float, copy=True))
        if inf_mask.any():
            raise ValueError(f"Target output contains `inf` or `-inf` in `{target_name}`.")

        # La volatilidad no puede ser negativa
        if (target_series.dropna() < 0).any():
            raise ValueError(f"Target output contains negative values in `{target_name}`.")

    # Validaciones específicas para target de clasificación (si está habilitado)
    if cfg["classification_target"]["enabled"]:
        classification_name = cfg["classification_target"]["name"]
        allowed_labels = set(cfg["classification_target"]["labels"])

        if classification_name not in df.columns:
            raise ValueError(
                f"Missing classification target column `{classification_name}` in output."
            )

        observed = set(df[classification_name].dropna().unique().tolist())
        invalid_labels = observed - allowed_labels
        if invalid_labels:
            raise ValueError(
                f"Target output contains invalid labels in `{classification_name}`: {sorted(invalid_labels)}"
            )