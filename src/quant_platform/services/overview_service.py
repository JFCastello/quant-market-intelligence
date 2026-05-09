from __future__ import annotations

from typing import Any

import pandas as pd

from .artifact_loaders import (
    load_decision_summary,
    load_symbol_features,
    load_symbol_structural_break_events,
    load_symbol_targets,
)
from .settings import load_settings


def _get_latest_non_null_row(df: pd.DataFrame, value_columns: list[str]) -> pd.Series:
    """
    Devuelve la última fila cronológicamente válida para un conjunto de columnas
    requeridas.

    Propósito:
    Esta función busca la observación más reciente que tenga valores no nulos
    en ciertas columnas clave. Es útil cuando un DataFrame puede contener
    fechas recientes con NaN, pero se necesita el último dato realmente usable.

    Parámetros:
    - df:
      DataFrame que debe contener al menos una columna `date` y las columnas
      listadas en `value_columns`.
    - value_columns:
      columnas que deben ser no nulas en la fila seleccionada.

    Retorna:
    - Una fila de pandas (`pd.Series`) correspondiente a la observación más
      reciente que cumple la condición de no tener nulos en esas columnas.

    Lanza:
    - ValueError si el DataFrame está vacío.
    - ValueError si, después de filtrar por no nulos, no queda ninguna fila.

    Idea de diseño:
    Esta función encapsula un patrón muy común en paneles de resumen:
    “dame el último dato válido”, no simplemente “dame la última fila”.
    """
    # Si el DataFrame llega vacío, no hay nada que buscar.
    if df.empty:
        raise ValueError("Received empty dataframe while looking for latest non-null row.")

    # Se trabaja sobre una copia para no mutar el DataFrame original
    # accidentalmente.
    working_df = df.copy()

    # Convierte la columna date a datetime de forma estricta.
    # Si hay valores no parseables, se levanta error inmediatamente.
    working_df["date"] = pd.to_datetime(working_df["date"], errors="raise")

    # Elimina filas que tengan nulos en cualquiera de las columnas requeridas
    # y luego ordena cronológicamente.
    valid_df = working_df.dropna(subset=value_columns).sort_values("date")

    # Si no queda ninguna fila válida, se informa explícitamente.
    if valid_df.empty:
        raise ValueError(
            f"No non-null rows found for required columns: {value_columns}"
        )

    # Toma la última fila cronológica, es decir, la más reciente con datos válidos.
    return valid_df.iloc[-1]


def get_symbol_overview_timeseries(symbol: str) -> pd.DataFrame:
    """
    Construye una serie temporal unificada de overview para un símbolo.

    Propósito:
    Combinar en un solo DataFrame:
    - las features del símbolo
    - el target principal `future_rv_5d`

    Esto produce una tabla base más cómoda para análisis, snapshots,
    visualización o consumo desde una capa superior.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un DataFrame fusionado, ordenado por fecha, con features y target.

    Diseño:
    - Las features se toman como base.
    - El target se incorpora mediante un merge por `instrument_id` y `date`.
    - Se usa `validate="one_to_one"` para exigir que no existan duplicidades
      inesperadas en la clave de unión.
    """
    # Carga las features y targets del símbolo.
    # Se hace `.copy()` para trabajar sobre copias aisladas.
    features_df = load_symbol_features(symbol).copy()
    targets_df = load_symbol_targets(symbol).copy()

    # Normaliza ambas columnas date a datetime de manera estricta.
    features_df["date"] = pd.to_datetime(features_df["date"], errors="raise")
    targets_df["date"] = pd.to_datetime(targets_df["date"], errors="raise")

    # Une features con una selección mínima del target.
    #
    # Solo se trae:
    # - instrument_id
    # - date
    # - future_rv_5d
    #
    # porque eso es lo necesario para el overview actual.
    merged_df = features_df.merge(
        targets_df[["instrument_id", "date", "future_rv_5d"]],
        on=["instrument_id", "date"],
        how="left",
        validate="one_to_one",
    )

    # Ordena cronológicamente y reinicia el índice para dejar un DataFrame
    # limpio y predecible.
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    return merged_df


def get_symbol_decision_summary(symbol: str) -> pd.Series:
    """
    Recupera la fila única de decision summary para un símbolo.

    Propósito:
    Extraer del artefacto global de decision summary exactamente la fila que
    corresponde al símbolo solicitado.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Una fila (`pd.Series`) con el resumen de decisión del símbolo.

    Lanza:
    - ValueError si no existe ninguna fila para el símbolo.
    - ValueError si existen múltiples filas, ya que el contrato esperado es
      una sola fila por símbolo.

    Detalle de diseño:
    La comparación se hace en mayúsculas para volver la búsqueda más robusta
    frente a diferencias de capitalización.
    """
    # Carga el summary global de decisiones.
    decision_summary_df = load_decision_summary().copy()

    # Normaliza el símbolo de entrada a mayúsculas.
    symbol_upper = symbol.upper()

    # Filtra las filas correspondientes al símbolo solicitado.
    match_df = decision_summary_df.loc[
        decision_summary_df["symbol"].astype(str).str.upper() == symbol_upper
    ].copy()

    # Si no hay coincidencia, se reporta explícitamente.
    if match_df.empty:
        raise ValueError(f"No decision summary row found for symbol={symbol_upper}")

    # Si hay más de una coincidencia, el artefacto no cumple el contrato
    # esperado de unicidad por símbolo.
    if len(match_df) != 1:
        raise ValueError(
            f"Expected exactly one decision summary row for symbol={symbol_upper}, found {len(match_df)}"
        )

    # Retorna la única fila correspondiente.
    return match_df.iloc[0]


def get_symbol_recent_break_events(symbol: str, limit: int | None = None) -> pd.DataFrame:
    """
    Recupera los eventos recientes de structural breaks para un símbolo.

    Propósito:
    Cargar la tabla de eventos, ordenarla desde el quiebre más reciente al
    más antiguo, y opcionalmente limitar el número de filas retornadas.

    Parámetros:
    - symbol:
      símbolo a consultar.
    - limit:
      número máximo de eventos a devolver. Si es `None`, se devuelven todos.

    Retorna:
    - Un DataFrame ordenado por `break_date` descendente.
      Si el archivo de eventos está vacío, se devuelve tal cual.

    Uso típico:
    Esta función está pensada para alimentar snapshots o paneles de resumen
    donde solo interesan los últimos quiebres detectados.
    """
    # Carga el DataFrame de eventos del símbolo.
    events_df = load_symbol_structural_break_events(symbol).copy()

    # Si no hay eventos, devuelve el DataFrame vacío directamente.
    if events_df.empty:
        return events_df

    # Normaliza break_date y ordena de más reciente a más antiguo.
    events_df["break_date"] = pd.to_datetime(events_df["break_date"], errors="raise")
    events_df = events_df.sort_values("break_date", ascending=False).reset_index(drop=True)

    # Si se pidió límite, recorta el DataFrame.
    if limit is not None:
        events_df = events_df.head(limit).reset_index(drop=True)

    return events_df


def build_symbol_overview_snapshot(symbol: str) -> dict[str, Any]:
    """
    Construye un snapshot compacto con los datos más importantes de un símbolo.

    Propósito:
    Consolidar, en un único diccionario, información reciente y resumida de:
    - la serie temporal fusionada de features/targets,
    - la fila de decision summary,
    - los structural breaks recientes.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un diccionario con valores de overview listos para consumo por una capa
      superior, por ejemplo un servicio web, dashboard o componente UI.

    Qué incluye el snapshot:
    - identidad básica del símbolo e instrumento
    - fechas más recientes con datos válidos
    - últimas métricas numéricas relevantes
    - decisión de modelo
    - métricas comparativas
    - información resumida de quiebres estructurales recientes

    Idea de diseño:
    Esta función representa una capa de agregación/orquestación. No descubre
    archivos ni ejecuta lógica analítica compleja; simplemente compone datos
    ya procesados en una estructura lista para exposición.
    """
    # Carga settings globales y extrae la configuración de services_layer.
    settings = load_settings()
    services_cfg = settings["services_layer"]

    # Obtiene desde configuración cuántos break events recientes deben incluirse.
    break_limit = int(services_cfg["defaults"]["latest_break_events_limit"])

    # Construye la serie temporal fusionada del símbolo.
    timeseries_df = get_symbol_overview_timeseries(symbol)

    # Obtiene la fila única de decision summary del símbolo.
    decision_row = get_symbol_decision_summary(symbol)

    # Obtiene los eventos de quiebre más recientes, limitados por configuración.
    recent_breaks_df = get_symbol_recent_break_events(symbol, limit=break_limit)

    # Busca la última fila con log_ret_1d no nulo.
    # Esto permite identificar el dato de features más reciente realmente usable.
    latest_feature_row = _get_latest_non_null_row(
        timeseries_df,
        value_columns=["log_ret_1d"],
    )

    # Busca la última fila con future_rv_5d no nulo.
    # Puede no coincidir exactamente con latest_feature_row si el target tiene
    # una disponibilidad temporal distinta.
    latest_target_row = _get_latest_non_null_row(
        timeseries_df,
        value_columns=["future_rv_5d"],
    )

    # Construye el snapshot final.
    snapshot = {
        # Símbolo según la fila de decision summary.
        "symbol": str(decision_row["symbol"]),

        # Instrumento según la última fila válida de features.
        "instrument_id": str(latest_feature_row["instrument_id"]),

        # Fechas más recientes con datos válidos para features y target.
        "latest_feature_date": pd.to_datetime(latest_feature_row["date"]),
        "latest_target_date": pd.to_datetime(latest_target_row["date"]),

        # Último retorno diario disponible.
        "latest_log_ret_1d": float(latest_feature_row["log_ret_1d"]),

        # Volatilidad rolling 20d, si existe y no es nula.
        # Si no está disponible, retorna None.
        "latest_vol_20d": float(latest_feature_row["vol_20d"]) if pd.notna(latest_feature_row.get("vol_20d")) else None,

        # Drawdown 60, si existe y no es nulo.
        "latest_drawdown_60": float(latest_feature_row["drawdown_60"]) if pd.notna(latest_feature_row.get("drawdown_60")) else None,

        # Última volatilidad futura disponible.
        "latest_future_rv_5d": float(latest_target_row["future_rv_5d"]),

        # Decisión final tomada para el símbolo.
        "decision": str(decision_row["decision"]),

        # Métricas de comparación entre modelos.
        "relative_qlike_improvement_mean": float(decision_row["relative_qlike_improvement_mean"]),
        "macro_f1_delta": float(decision_row["macro_f1_delta"]),
        "balanced_accuracy_delta": float(decision_row["balanced_accuracy_delta"]),

        # Estado de calibración reportado en el summary.
        "calibration_status": str(decision_row["calibration_status"]),

        # Número de break events recientes incluidos en el snapshot.
        "recent_break_count": int(len(recent_breaks_df)),

        # Fecha del quiebre más reciente, si existe al menos un evento.
        "most_recent_break_date": (
            pd.to_datetime(recent_breaks_df.iloc[0]["break_date"])
            if not recent_breaks_df.empty
            else None
        ),
    }

    return snapshot


def get_symbol_overview_bundle(symbol: str) -> dict[str, Any]:
    """
    Construye un bundle completo de overview para un símbolo.

    Propósito:
    Retornar, en una sola estructura, tanto el snapshot resumido como los
    artefactos tabulares subyacentes que lo respaldan.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un diccionario con cuatro entradas:
      * "snapshot":
          resumen agregado y compacto del símbolo
      * "timeseries_df":
          DataFrame fusionado de features y target
      * "decision_summary_row":
          fila de decision summary para el símbolo
      * "recent_break_events_df":
          DataFrame con los break events recientes

    Diferencia respecto a `build_symbol_overview_snapshot`:
    - `build_symbol_overview_snapshot` devuelve solo un resumen compacto.
    - `get_symbol_overview_bundle` devuelve además los datos detallados.

    Uso típico:
    Esta función es útil cuando una capa superior necesita tanto la vista
    resumida como las tablas de respaldo para renderizar detalles,
    inspecciones o visualizaciones.
    """
    return {
        # Resumen compacto listo para mostrar.
        "snapshot": build_symbol_overview_snapshot(symbol),

        # Serie temporal completa combinada.
        "timeseries_df": get_symbol_overview_timeseries(symbol),

        # Fila única del decision summary para el símbolo.
        "decision_summary_row": get_symbol_decision_summary(symbol),

        # Eventos recientes de structural breaks, limitados por configuración.
        "recent_break_events_df": get_symbol_recent_break_events(
            symbol,
            limit=int(load_settings()["services_layer"]["defaults"]["latest_break_events_limit"]),
        ),
    }