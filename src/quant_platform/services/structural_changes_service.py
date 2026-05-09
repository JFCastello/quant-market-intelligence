from __future__ import annotations

from typing import Any

import pandas as pd

from .artifact_loaders import (
    load_symbol_structural_break_events,
    load_symbol_structural_break_signal,
)
from .settings import load_settings


def get_symbol_structural_break_signal(symbol: str) -> pd.DataFrame:
    """
    Carga y normaliza el artefacto de señal de structural breaks para un símbolo.

    Propósito:
    Esta función expone una versión "lista para consumo" del parquet de señal.
    Su trabajo no es recalcular la señal, sino:
    1. cargarla desde la capa de artifact loaders,
    2. asegurar que la columna `date` tenga tipo datetime válido,
    3. devolver el DataFrame ordenado cronológicamente.

    Parámetros:
    - symbol:
      símbolo cuyo artefacto de señal se quiere consultar.

    Retorna:
    - Un DataFrame ordenado ascendentemente por `date`.

    Importancia:
    Esta función actúa como una pequeña capa de saneamiento/normalización
    antes de que otras funciones del módulo usen la señal para construir
    resúmenes o bundles.
    """
    # Carga el parquet de señal y trabaja sobre una copia para evitar
    # mutaciones accidentales del objeto original retornado por el loader.
    df = load_symbol_structural_break_signal(symbol).copy()

    # Convierte la columna de fechas a datetime de forma estricta.
    # Si existe una fecha malformada, se lanza una excepción inmediatamente.
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # Ordena cronológicamente la señal y reinicia el índice para dejar
    # un DataFrame limpio, consistente y fácil de consumir.
    return df.sort_values("date").reset_index(drop=True)


def get_symbol_structural_break_events(symbol: str) -> pd.DataFrame:
    """
    Carga y normaliza el artefacto de eventos de structural breaks.

    Propósito:
    Esta función prepara la tabla de eventos para consumo posterior:
    - carga el parquet,
    - convierte a datetime las columnas temporales relevantes,
    - ordena los eventos por `break_date`.

    Parámetros:
    - symbol:
      símbolo cuyos eventos se quieren consultar.

    Retorna:
    - Un DataFrame de eventos ordenado ascendentemente por `break_date`.

    Nota:
    Si el artefacto está vacío, se devuelve tal cual, porque un DataFrame vacío
    puede significar simplemente que no se detectaron quiebres para ese símbolo.
    """
    # Carga el parquet de eventos y crea una copia aislada para trabajar
    # sin modificar el DataFrame retornado por la capa inferior.
    df = load_symbol_structural_break_events(symbol).copy()

    # Si no hay eventos, retorna inmediatamente.
    # Esto evita trabajo innecesario y conserva la semántica de
    # "no hubo quiebres detectados".
    if df.empty:
        return df

    # Lista de columnas temporales que se espera existan en el artefacto
    # de eventos. Cada una se convierte a datetime para asegurar tipado
    # correcto y comparaciones temporales seguras.
    date_cols = [
        "break_date",
        "previous_segment_start_date",
        "previous_segment_end_date",
        "next_segment_start_date",
        "next_segment_end_date",
        "detected_at",
    ]

    # Convierte solo las columnas que efectivamente estén presentes.
    # Esto vuelve la función un poco más robusta frente a variaciones
    # menores en el esquema del DataFrame.
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="raise")

    # Ordena los eventos desde el más antiguo al más reciente.
    return df.sort_values("break_date").reset_index(drop=True)


def get_symbol_recent_structural_break_events(
    symbol: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Devuelve los eventos de quiebre más recientes para un símbolo.

    Propósito:
    Esta función construye una vista orientada a "recencia" a partir del
    DataFrame de eventos ya normalizado. Es útil para dashboards, snapshots
    o paneles donde interesan principalmente los últimos quiebres.

    Parámetros:
    - symbol:
      símbolo a consultar.
    - limit:
      número máximo de eventos a devolver.
      Si es `None`, se devuelven todos los eventos, pero ordenados del más
      reciente al más antiguo.

    Retorna:
    - Un DataFrame ordenado por `break_date` descendente.
    """
    # Obtiene el DataFrame de eventos ya parseado y ordenado por fecha ascendente.
    events_df = get_symbol_structural_break_events(symbol)

    # Si no hay eventos, devuelve el DataFrame vacío sin más transformación.
    if events_df.empty:
        return events_df

    # Reordena los eventos desde el más reciente hacia atrás.
    recent_df = events_df.sort_values("break_date", ascending=False).reset_index(drop=True)

    # Si el usuario o la configuración pide solo los N más recientes,
    # recorta el DataFrame.
    if limit is not None:
        recent_df = recent_df.head(limit).reset_index(drop=True)

    return recent_df


def build_symbol_structural_break_summary(symbol: str) -> dict[str, Any]:
    """
    Construye un resumen compacto del estado de structural breaks para un símbolo.

    Propósito:
    Reunir en un único diccionario la información más importante de la señal
    y de los eventos detectados, de forma conveniente para una capa de servicios,
    una API o un dashboard.

    Parámetros:
    - symbol:
      símbolo a resumir.

    Retorna:
    - Un diccionario con información resumida como:
      * símbolo
      * instrument_id
      * número de filas de la señal
      * número total de eventos
      * rango temporal cubierto por la señal
      * último valor observado de variables clave
      * fecha del quiebre más reciente
      * fecha del quiebre más antiguo
      * límite configurado de eventos recientes

    Lanza:
    - ValueError si la señal está vacía.
      A diferencia de los eventos, una señal vacía sí implica que no hay base
      mínima para construir el resumen.
    """
    # Carga settings globales del proyecto.
    settings = load_settings()

    # Extrae la configuración de la capa de servicios.
    services_cfg = settings["services_layer"]

    # Obtiene cuántos eventos recientes deben considerarse como máximo
    # en el resumen.
    latest_limit = int(services_cfg["defaults"]["latest_break_events_limit"])

    # Carga la señal completa, los eventos completos y una vista de eventos
    # recientes limitada por configuración.
    signal_df = get_symbol_structural_break_signal(symbol)
    events_df = get_symbol_structural_break_events(symbol)
    recent_events_df = get_symbol_recent_structural_break_events(symbol, limit=latest_limit)

    # Una señal vacía invalida el resumen completo, porque no habría forma
    # de inferir fechas, últimas observaciones ni identidad del instrumento.
    if signal_df.empty:
        raise ValueError(f"Signal dataframe is empty for symbol={symbol}")

    # Toma la última fila cronológica de la señal.
    # Como signal_df ya viene ordenado por fecha ascendente, iloc[-1]
    # corresponde al dato más reciente.
    latest_signal_row = signal_df.iloc[-1]

    # Construye el resumen final en forma de diccionario.
    summary = {
        # Símbolo en formato canónico de salida.
        "symbol": str(symbol).upper(),

        # Instrumento asociado a la última fila de señal.
        "instrument_id": str(latest_signal_row["instrument_id"]),

        # Número total de observaciones en la señal.
        "signal_rows": int(len(signal_df)),

        # Número total de eventos detectados.
        "event_count": int(len(events_df)),

        # Fecha mínima y máxima cubiertas por la señal.
        "signal_start_date": pd.to_datetime(signal_df["date"].min()),
        "signal_end_date": pd.to_datetime(signal_df["date"].max()),

        # Fecha del último dato disponible en la señal.
        "latest_signal_date": pd.to_datetime(latest_signal_row["date"]),

        # Últimos valores observados de las variables principales.
        "latest_log_ret_1d": float(latest_signal_row["log_ret_1d"]),
        "latest_future_rv_5d": float(latest_signal_row["future_rv_5d"]),

        # Fecha del quiebre más reciente, si existe al menos uno.
        # Como recent_events_df viene en orden descendente, la fila 0
        # representa el quiebre más reciente.
        "most_recent_break_date": (
            pd.to_datetime(recent_events_df.iloc[0]["break_date"])
            if not recent_events_df.empty
            else None
        ),

        # Fecha del quiebre más antiguo, si existe al menos uno.
        # Como events_df viene ordenado ascendentemente, la fila 0
        # representa el quiebre más antiguo.
        "oldest_break_date": (
            pd.to_datetime(events_df.iloc[0]["break_date"])
            if not events_df.empty
            else None
        ),

        # Límite configurado usado para construir la vista de eventos recientes.
        "latest_break_events_limit": latest_limit,
    }

    return summary


def get_symbol_structural_changes_bundle(symbol: str) -> dict[str, Any]:
    """
    Devuelve un bundle completo con las principales vistas de structural breaks
    para un símbolo.

    Propósito:
    Reunir en una sola estructura:
    - el resumen agregado,
    - la señal completa,
    - los eventos completos,
    - los eventos recientes.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un diccionario con cuatro entradas:
      * "summary"
      * "signal_df"
      * "events_df"
      * "recent_events_df"

    Utilidad:
    Esta función está pensada como punto de acceso conveniente para una capa
    superior que necesite tanto una vista resumida como los DataFrames de
    respaldo para inspección o visualización.
    """
    # Lee la configuración para recuperar el límite estándar de eventos recientes.
    settings = load_settings()
    latest_limit = int(settings["services_layer"]["defaults"]["latest_break_events_limit"])

    return {
        # Resumen agregado y compacto del símbolo.
        "summary": build_symbol_structural_break_summary(symbol),

        # Señal cronológica completa de structural breaks.
        "signal_df": get_symbol_structural_break_signal(symbol),

        # Tabla completa de eventos detectados.
        "events_df": get_symbol_structural_break_events(symbol),

        # Subconjunto de eventos más recientes, limitado por configuración.
        "recent_events_df": get_symbol_recent_structural_break_events(
            symbol,
            limit=latest_limit,
        ),
    }