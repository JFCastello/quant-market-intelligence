# Archivo: context_features.py
# Propósito: Construir features de contexto (comparativas entre múltiples instrumentos)
#            a partir de un universo de activos, y enriquecer las features objetivo
#            con información de mercado relativa (spreads, volatilidad relativa,
#            correlaciones móviles, etc.)

from __future__ import annotations

from typing import Any

import pandas as pd


# ----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE CONFIGURACIÓN
# ----------------------------------------------------------------------------
def _context_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae la subconfiguración 'context_features' del diccionario settings.
    Lanza KeyError si no existe.
    """
    if "context_features" not in settings:
        raise KeyError("Missing `context_features` section in settings.")
    return settings["context_features"]


def get_enabled_context_feature_columns(settings: dict[str, Any]) -> list[str]:
    """
    Devuelve la lista de nombres de columnas que se generarán cuando las
    features de contexto están habilitadas, según la configuración.
    """
    cfg = _context_cfg(settings)
    prefix = cfg["naming"]["prefix"]           # Prefijo para todas las columnas de contexto

    columns: list[str] = []

    if not cfg["enabled"]:
        return columns                         # Si contexto deshabilitado, lista vacía

    # 1. Rendimientos directos (direct_returns) para cada rol (ej. "benchmark", "peer")
    if cfg["direct_returns"]["enabled"]:
        method = cfg["direct_returns"]["method"]       # "log" o "simple"
        for role in cfg["role_map"].keys():            # roles definidos (ej. "market", "sector")
            for window in cfg["direct_returns"]["windows"]:
                columns.append(f"{prefix}_{role}_{method}_ret_{window}d")

    # 2. Volatilidad directa (direct_volatility) para cada rol
    if cfg["direct_volatility"]["enabled"]:
        for role in cfg["role_map"].keys():
            for window in cfg["direct_volatility"]["windows"]:
                columns.append(f"{prefix}_{role}_vol_{window}d")

    # 3. Spreads (diferencias) entre pares de roles
    if cfg["spreads"]["enabled"]:
        for definition in cfg["spreads"]["definitions"]:
            columns.append(f"{prefix}_{definition['name']}")

    # 4. Volatilidad relativa (cociente entre dos volatilidades)
    if cfg["relative_volatility"]["enabled"]:
        columns.append(f"{prefix}_{cfg['relative_volatility']['output_name']}")

    # 5. Correlación móvil (rolling correlation) entre activo objetivo y rol de referencia
    if cfg["rolling_correlation"]["enabled"]:
        columns.append(f"{prefix}_{cfg['rolling_correlation']['output_name']}")

    return columns


# ----------------------------------------------------------------------------
# INFERENCIA DE IDs REALES A PARTIR DE SÍMBOLOS EN LA CONFIGURACIÓN
# ----------------------------------------------------------------------------
def infer_role_to_instrument_id(
    universe_features_df: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, str]:
    """
    Mapea cada rol definido en role_map (ej. "market" -> "spy_us") al
    instrument_id real que existe en el DataFrame de features del universo.
    Permite que el símbolo sea exacto o un prefijo (ej. "spy_" para "spy_us_2020").
    Lanza error si no encuentra exactamente una coincidencia.
    """
    cfg = _context_cfg(settings)
    # Lista ordenada de todos los instrument_id presentes (sin nulos)
    available_ids = sorted(universe_features_df["instrument_id"].dropna().unique().tolist())

    role_to_instrument_id: dict[str, str] = {}

    for role, symbol in cfg["role_map"].items():
        symbol_lower = symbol.lower()

        # Busca coincidencia exacta o que empiece por "symbol_"
        matches = [
            instrument_id
            for instrument_id in available_ids
            if instrument_id == symbol_lower or instrument_id.startswith(f"{symbol_lower}_")
        ]

        if not matches:
            raise ValueError(
                f"Could not infer instrument_id for role `{role}` from symbol `{symbol}`. "
                f"Available instrument_ids: {available_ids}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous instrument_id inference for role `{role}` from symbol `{symbol}`. "
                f"Matches found: {matches}"
            )

        role_to_instrument_id[role] = matches[0]

    return role_to_instrument_id


# ----------------------------------------------------------------------------
# VALIDACIÓN DE DATOS DE ENTRADA (UNIVERSO DE FEATURES)
# ----------------------------------------------------------------------------
def validate_universe_feature_input(df: pd.DataFrame, settings: dict[str, Any]) -> None:
    """
    Verifica que el DataFrame del universo de features contenga todas las columnas
    necesarias según las características habilitadas (rendimientos, volatilidad,
    fuente para volatilidad relativa, etc.)
    """
    required_cols = {"instrument_id", "date", "feature_version"}

    cfg = _context_cfg(settings)

    if cfg["direct_returns"]["enabled"]:
        method = cfg["direct_returns"]["method"]
        for window in cfg["direct_returns"]["windows"]:
            required_cols.add(f"{method}_ret_{window}d")

    if cfg["direct_volatility"]["enabled"]:
        for window in cfg["direct_volatility"]["windows"]:
            required_cols.add(f"vol_{window}d")

    if cfg["relative_volatility"]["enabled"]:
        required_cols.add(cfg["relative_volatility"]["source_feature"])

    if cfg["rolling_correlation"]["enabled"]:
        required_cols.add(cfg["rolling_correlation"]["source_feature"])

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Universe feature input is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Universe feature input dataframe is empty.")

    if df["instrument_id"].isna().any():
        raise ValueError("Universe feature input has null `instrument_id` values.")

    if df["date"].isna().any():
        raise ValueError("Universe feature input has null `date` values.")


# ----------------------------------------------------------------------------
# ORDENAMIENTO INTERNO DE DATOS (POR INSTRUMENTO Y FECHA)
# ----------------------------------------------------------------------------

def _sort_universe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena el DataFrame por instrument_id y fecha, y resetea el índice."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# CONSTRUCCIÓN DEL PANEL DE CONTEXTO A NIVEL DE FECHA (SIN TARGET)
# ----------------------------------------------------------------------------

def build_date_level_context_panel(
    universe_features_df: pd.DataFrame,
    settings: dict[str, Any],
    role_to_instrument_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Crea un DataFrame con una columna 'date' y columnas adicionales para cada
    rol y feature de contexto (rendimientos directos, volatilidad directa, spreads).
    Este panel contiene una sola fila por fecha, combinando información de
    todos los roles.
    """
    validate_universe_feature_input(universe_features_df, settings)

    cfg = _context_cfg(settings)
    prefix = cfg["naming"]["prefix"]

    if not cfg["enabled"]:
        return pd.DataFrame(columns=["date"])   # Vacío si contexto deshabilitado

    out_df = _sort_universe_features(universe_features_df)

    # Inferir mapeo si no se proporcionó
    if role_to_instrument_id is None:
        role_to_instrument_id = infer_role_to_instrument_id(out_df, settings)

    panel: pd.DataFrame | None = None

    # Procesa cada rol por separado, extrae las columnas de interés y las renombra
    for role, instrument_id in role_to_instrument_id.items():
        role_df = out_df.loc[out_df["instrument_id"] == instrument_id].copy()

        if role_df.empty:
            raise ValueError(
                f"No rows found for role `{role}` with instrument_id `{instrument_id}`."
            )

        keep_cols = ["date"]

        # Añade columnas de rendimientos directos si están habilitadas
        if cfg["direct_returns"]["enabled"]:
            method = cfg["direct_returns"]["method"]
            for window in cfg["direct_returns"]["windows"]:
                keep_cols.append(f"{method}_ret_{window}d")

        # Añade columnas de volatilidad directa si están habilitadas
        if cfg["direct_volatility"]["enabled"]:
            for window in cfg["direct_volatility"]["windows"]:
                keep_cols.append(f"vol_{window}d")

        role_df = role_df[keep_cols].copy()

        # Renombra columnas añadiendo el prefijo y el rol: ej. "ctx_market_log_ret_1d"
        rename_map: dict[str, str] = {}
        for col in keep_cols:
            if col == "date":
                continue
            rename_map[col] = f"{prefix}_{role}_{col}"
        role_df = role_df.rename(columns=rename_map)

        # Combina con el panel principal mediante merge por fecha (outer join)
        if panel is None:
            panel = role_df
        else:
            panel = panel.merge(role_df, on="date", how="outer")

    if panel is None:
        panel = pd.DataFrame(columns=["date"])

    # Añade spreads (diferencias) entre columnas de dos roles distintos
    if cfg["spreads"]["enabled"]:
        for definition in cfg["spreads"]["definitions"]:
            left_role = definition["left_role"]
            right_role = definition["right_role"]
            source_feature = definition["source_feature"]   # ej. "log_ret_1d"
            output_name = definition["name"]

            left_col = f"{prefix}_{left_role}_{source_feature}"
            right_col = f"{prefix}_{right_role}_{source_feature}"
            out_col = f"{prefix}_{output_name}"

            if left_col not in panel.columns or right_col not in panel.columns:
                raise ValueError(
                    f"Cannot build spread `{out_col}` because `{left_col}` or `{right_col}` is missing."
                )

            panel[out_col] = panel[left_col] - panel[right_col]

    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


# ----------------------------------------------------------------------------
# MEZCLA DEL PANEL DE CONTEXTO CON LAS FEATURES TARGET (POR FECHA)
# ----------------------------------------------------------------------------

def merge_date_level_context(
    target_features_df: pd.DataFrame,
    date_level_context_panel: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Combina el DataFrame de features objetivo (un instrumento por fila) con el
    panel de contexto (una fila por fecha) mediante una clave de merge
    (normalmente 'date').
    """
    cfg = _context_cfg(settings)
    merge_keys = cfg["merge_keys"]   # normalmente ["date"]

    out = target_features_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")

    panel = date_level_context_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")

    out = out.merge(panel, on=merge_keys, how="left")
    out = out.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# CÁLCULO DE VOLATILIDAD RELATIVA (TARGET / REFERENCIA)
# ----------------------------------------------------------------------------

def add_relative_volatility_context(
    df: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Añade una columna de volatilidad relativa = volatilidad del target /
    volatilidad del rol de referencia (ambas ya deben existir en el DataFrame).
    """
    cfg = _context_cfg(settings)
    prefix = cfg["naming"]["prefix"]

    if not cfg["relative_volatility"]["enabled"]:
        return df.copy()

    out = df.copy()

    source_feature = cfg["relative_volatility"]["source_feature"]   # ej. "vol_20d"
    reference_role = cfg["relative_volatility"]["reference_role"]   # ej. "market"
    output_name = cfg["relative_volatility"]["output_name"]         # ej. "rel_vol"

    ref_col = f"{prefix}_{reference_role}_{source_feature}"
    out_col = f"{prefix}_{output_name}"

    if source_feature not in out.columns:
        raise ValueError(f"Missing target source feature `{source_feature}` for relative volatility.")

    if ref_col not in out.columns:
        raise ValueError(f"Missing reference context feature `{ref_col}` for relative volatility.")

    out[out_col] = out[source_feature] / out[ref_col]
    return out


# ----------------------------------------------------------------------------
# CÁLCULO DE CORRELACIÓN MÓVIL ENTRE TARGET Y REFERENCIA
# ----------------------------------------------------------------------------

def add_rolling_correlation_context(
    df: pd.DataFrame,
    settings: dict[str, Any],
    role_to_instrument_id: dict[str, str],
) -> pd.DataFrame:
    """
    Añade una columna con la correlación móvil (rolling) entre una feature
    del instrumento target y la misma feature del rol de referencia.
    Permite tratar la autocorrelación (cuando target == referencia) según
    política: 'nan' (poner NA) u otra.
    """
    cfg = _context_cfg(settings)
    prefix = cfg["naming"]["prefix"]

    if not cfg["rolling_correlation"]["enabled"]:
        return df.copy()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    source_feature = cfg["rolling_correlation"]["source_feature"]   # ej. "log_ret_1d"
    reference_role = cfg["rolling_correlation"]["reference_role"]   # ej. "market"
    window = cfg["rolling_correlation"]["window"]                   # ventana móvil (días)
    output_name = cfg["rolling_correlation"]["output_name"]         # ej. "corr"
    self_reference_policy = cfg["rolling_correlation"]["self_reference_policy"]  # "nan" u otro

    ref_ctx_col = f"{prefix}_{reference_role}_{source_feature}"
    out_col = f"{prefix}_{output_name}"

    if source_feature not in out.columns:
        raise ValueError(
            f"Missing target source feature `{source_feature}` for rolling correlation."
        )

    if ref_ctx_col not in out.columns:
        raise ValueError(
            f"Missing reference context feature `{ref_ctx_col}` for rolling correlation."
        )

    # Calcula correlación rodante por cada instrumento (target) con la serie de referencia
    # Nota: La referencia es la misma para todos los targets (por eso no se agrupa por ella)
    corr_series = (
        out.groupby("instrument_id", group_keys=False)[[source_feature, ref_ctx_col]]
        .apply(
            lambda g: g[source_feature]
            .rolling(window=window, min_periods=window)
            .corr(g[ref_ctx_col])
        )
        .reset_index(level=0, drop=True)
    )

    out[out_col] = corr_series

    # Política para cuando el instrumento target es exactamente el mismo que el de referencia
    reference_instrument_id = role_to_instrument_id[reference_role]
    if self_reference_policy == "nan":
        mask = out["instrument_id"] == reference_instrument_id
        out.loc[mask, out_col] = pd.NA   # o pd.NA, o np.nan según versión

    return out


# ----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: CONSTRUYE FEATURES ENRIQUECIDAS CON CONTEXTO
# ----------------------------------------------------------------------------

def build_context_enriched_features(
    universe_features_df: pd.DataFrame,
    settings: dict[str, Any],
    role_to_instrument_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Orquesta todo el pipeline de contexto:
      1. Ordena los datos de entrada.
      2. Infiere el mapeo rol -> instrument_id (si no se da).
      3. Construye el panel de contexto a nivel de fecha.
      4. Mezcla el panel con las features originales (target).
      5. Añade volatilidad relativa y correlación móvil.
      6. Devuelve el DataFrame final ordenado.
    """
    cfg = _context_cfg(settings)

    if not cfg["enabled"]:
        return _sort_universe_features(universe_features_df)

    base_df = _sort_universe_features(universe_features_df)

    if role_to_instrument_id is None:
        role_to_instrument_id = infer_role_to_instrument_id(base_df, settings)

    context_panel = build_date_level_context_panel(
        universe_features_df=base_df,
        settings=settings,
        role_to_instrument_id=role_to_instrument_id,
    )

    enriched = merge_date_level_context(
        target_features_df=base_df,
        date_level_context_panel=context_panel,
        settings=settings,
    )

    enriched = add_relative_volatility_context(
        df=enriched,
        settings=settings,
    )

    enriched = add_rolling_correlation_context(
        df=enriched,
        settings=settings,
        role_to_instrument_id=role_to_instrument_id,
    )

    enriched = enriched.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return enriched