from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_platform.schemas import FeatureRow


REQUIRED_NORMALIZED_COLUMNS = [ # Define las columnas mínimas que el input debe traer - <Castello>
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


def _feature_cfg(settings: dict[str, Any]) -> dict[str, Any]: # Extrae de settings solo la parte de features - <Castello>
    if "features" not in settings:
        raise KeyError("Missing `features` section in settings.")
    return settings["features"]


def get_enabled_feature_columns(settings: dict[str, Any]) -> list[str]: # Construye dinamicamente la lista de columnas finales esperadas 
    cfg = _feature_cfg(settings)                                        # según el YAML.

    columns: list[str] = [
        "instrument_id",
        "date",
        "feature_version",
    ]

    if cfg["returns"]["enabled"]:
        method = cfg["returns"]["method"]
        for window in cfg["returns"]["windows"]:
            columns.append(f"{method}_ret_{window}d")

    if cfg["volatility"]["enabled"]:
        for window in cfg["volatility"]["windows"]:
            columns.append(f"vol_{window}d")

    if cfg["intraday_range"]["enabled"]:
        if cfg["intraday_range"]["include_hl_range"]:
            columns.append("hl_range")
        if cfg["intraday_range"]["include_co_range"]:
            columns.append("co_range")

    if cfg["atr"]["enabled"]:
        columns.append(f"atr_{cfg['atr']['window']}")

    if cfg["momentum"]["enabled"]:
        for window in cfg["momentum"]["windows"]:
            columns.append(f"mom_{window}d")

    if cfg["moving_averages"]["enabled"]:
        for window in cfg["moving_averages"]["windows"]:
            columns.append(f"ma_{window}")

        for short_window, long_window in cfg["moving_averages"]["ratios"]:
            columns.append(f"ma_ratio_{short_window}_{long_window}")

    if cfg["drawdown"]["enabled"]:
        for window in cfg["drawdown"]["windows"]:
            columns.append(f"drawdown_{window}")

    return columns

def validate_normalized_input(df: pd.DataFrame) -> None:
    # Valida que el DataFrame de entrada:
        # 1.tenga las columnas mínimas
        # 2. no estE vacío
        # 3. no tenga instrument_id nulo
        # 4. no tenga date nula
        # 5. no tenga duplicados por instrument_id,date
        # 6. tenga fechas parseables
        # - <Castello>
        
    # Validacion de columnas requeridas
    missing_cols = [col for col in REQUIRED_NORMALIZED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Normalized input is missing required columns: {missing_cols}")

    # Validacion de contenido
    if df.empty:
        raise ValueError("Normalized input dataframe is empty.")

    # Validacion de nulls en instrument_id y date. (no pueden existir instrumentos sin id o filas sin fecha)
    if df["instrument_id"].isna().any():
        raise ValueError("Normalized input contains null `instrument_id` values.")
    if df["date"].isna().any():
        raise ValueError("Normalized input contains null `date` values.")

    # Validacion de duplicados por instrument_id/date (no pueden existir dos filas para el mismo instrumento y fecha, es decir
    # clave unica)
    duplicated = df.duplicated(subset=["instrument_id", "date"])
    if duplicated.any():
        raise ValueError("Normalized input contains duplicate `instrument_id`/`date` rows.")

    # Validacion de fechas parseables (todas las fechas deben ser parseables a datetime)
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Normalized input contains unparseable `date` values.")

# Convierte date a datetime, ordena por: instrument_id y date, y resetea el index. 
# Esto asegura que las filas estén en el orden correcto para los cálculos posteriores. - <Castello>
def _sort_input(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["date"] = pd.to_datetime(sorted_df["date"], errors="raise")
    sorted_df = sorted_df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return sorted_df

# Calculos de Featurees ------------------------------------------------------------
# Calcula los retornos logarítmicos definidos en el YAML
def compute_log_returns(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    returns_cfg = cfg["returns"]
    price_col = cfg["price_column"]
    method = returns_cfg["method"]

    if method != "log":
        raise NotImplementedError(f"Unsupported returns method: {method}")

    out = df.copy()
    grouped_price = out.groupby("instrument_id")[price_col]

    for window in returns_cfg["windows"]:
        col_name = f"{method}_ret_{window}d"
        out[col_name] = grouped_price.transform(lambda s: np.log(s / s.shift(window)))

    return out

# calcular la volatilidad histórica (rodante) de los rendimientos para cada instrumento, usando diferentes ventanas de tiempo
def compute_rolling_volatility(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    vol_cfg = cfg["volatility"]
    annualization_factor = cfg["annualization_factor"]
    base_return_column = vol_cfg["base_return_column"]

    if base_return_column not in df.columns:
        raise ValueError(
            f"Base return column `{base_return_column}` not found before volatility calculation."
        )

    out = df.copy()
    grouped_returns = out.groupby("instrument_id")[base_return_column]

    for window in vol_cfg["windows"]:
        col_name = f"vol_{window}d"
        out[col_name] = grouped_returns.transform(
            lambda s: s.rolling(window=window, min_periods=window).std()
        )

        if vol_cfg["annualize"]:
            out[col_name] = out[col_name] * np.sqrt(annualization_factor)

    return out

# Calcula rangos intradía (medidas de volatilidad o movimiento dentro del día) a partir de precios de alta, baja, apertura y cierre.
def compute_intraday_ranges(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    range_cfg = cfg["intraday_range"]

    out = df.copy()

    if range_cfg["include_hl_range"]:
        out["hl_range"] = (out["high"] - out["low"]) / out["close"]

    if range_cfg["include_co_range"]:
        out["co_range"] = (out["close"] - out["open"]) / out["open"]

    return out

# Calcula el Average True Range (ATR) normalizado.
def compute_atr(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    atr_cfg = cfg["atr"]
    price_col = atr_cfg["normalize_by"]
    window = atr_cfg["window"]

    out = df.copy()
    out["prev_close"] = out.groupby("instrument_id")["close"].shift(1)

    high_low = out["high"] - out["low"]
    high_prev_close = (out["high"] - out["prev_close"]).abs()
    low_prev_close = (out["low"] - out["prev_close"]).abs()

    out["true_range"] = pd.concat(
        [high_low, high_prev_close, low_prev_close], axis=1
    ).max(axis=1)

    atr_raw = out.groupby("instrument_id")["true_range"].transform(
        lambda s: s.rolling(window=window, min_periods=window).mean()
    )

    out[f"atr_{window}"] = atr_raw / out[price_col]
    return out

# Igual que compute_log_returns, pero se llama "momentum". De hecho, es idéntica: calcula log(P_t / P_{t-window}) para cada ventana.
def compute_momentum(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    momentum_cfg = cfg["momentum"]
    price_col = cfg["price_column"]
    method = momentum_cfg["method"]

    if method != "log":
        raise NotImplementedError(f"Unsupported momentum method: {method}")

    out = df.copy()
    grouped_price = out.groupby("instrument_id")[price_col]

    for window in momentum_cfg["windows"]:
        col_name = f"mom_{window}d"
        out[col_name] = grouped_price.transform(lambda s: np.log(s / s.shift(window)))

    return out

# Calcula medias móviles simples (SMA) y sus ratios.
#   Para cada ventana en windows → columna ma_{window} con la SMA del precio.
#   Para cada par (short_window, long_window) en ratios → columna ma_ratio_{short}_{long} = ma_short / ma_long.
def compute_moving_averages(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    ma_cfg = cfg["moving_averages"]
    price_col = cfg["price_column"]

    out = df.copy()
    grouped_price = out.groupby("instrument_id")[price_col]

    for window in ma_cfg["windows"]:
        out[f"ma_{window}"] = grouped_price.transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )

    for short_window, long_window in ma_cfg["ratios"]:
        short_col = f"ma_{short_window}"
        long_col = f"ma_{long_window}"
        ratio_col = f"ma_ratio_{short_window}_{long_window}"
        out[ratio_col] = out[short_col] / out[long_col]

    return out

#  Calcula el drawdown (caída desde máximo) para diferentes ventanas.
def compute_drawdowns(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    cfg = _feature_cfg(settings)
    drawdown_cfg = cfg["drawdown"]
    price_col = cfg["price_column"]

    out = df.copy()
    grouped_price = out.groupby("instrument_id")[price_col]

    for window in drawdown_cfg["windows"]:
        rolling_max = grouped_price.transform(
            lambda s: s.rolling(window=window, min_periods=window).max()
        )
        out[f"drawdown_{window}"] = out[price_col] / rolling_max - 1.0

    return out

# Funcion orquesatdora:
#   1. Valida entrada con validate_normalized_input.
#   2. Ordena con _sort_input.
#   3. Aplica secuencialmente: log returns → volatilidad → rangos intradía → ATR → momentum → medias móviles → drawdowns.
#   4. Añade columna feature_version (desde settings).
#   5. Devuelve DataFrame con todas las features calculadas.

def build_base_features(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    validate_normalized_input(df)

    out = _sort_input(df)
    out = compute_log_returns(out, settings)
    out = compute_rolling_volatility(out, settings)
    out = compute_intraday_ranges(out, settings)
    out = compute_atr(out, settings)
    out = compute_momentum(out, settings)
    out = compute_moving_averages(out, settings)
    out = compute_drawdowns(out, settings)

    out["feature_version"] = _feature_cfg(settings)["feature_version"]
    return out

# Filtra y valida las features para que cumplan el "contrato" (esquema esperado). 
def adapt_features_to_contract(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    feature_columns = get_enabled_feature_columns(settings)

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature dataframe is missing expected contract columns: {missing_cols}"
        )

    contracted = df[feature_columns].copy()

    records = contracted.to_dict(orient="records")
    validated_records = [
        FeatureRow(**record).model_dump(include=set(feature_columns))
        for record in records
    ]
    validated_df = pd.DataFrame(validated_records)
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="raise")

    return validated_df

# Validación exhaustiva del DataFrame final de features. Se comprueba:
#   -Columnas esperadas
#   -No vacio
#   -Sin nulos en instrument_id, date, feature_version.
#   -Sin duplicados por instrumento/fecha.
#   -Fechas parseables.
#   -Orden correcto (por instrumento y fecha).
#   -Versión de features consistente.
#   -Columnas numéricas sean numéricas.
#   -Sin valores infinitos.
#   -Volatilidad y ATR no negativos.
#   -Drawdowns no positivos (deben ser ≤ 0).

def validate_feature_output(df: pd.DataFrame, settings: dict[str, Any]) -> None:
    expected_columns = get_enabled_feature_columns(settings)

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature output is missing expected columns: {missing_cols}")

    if df.empty:
        raise ValueError("Feature output dataframe is empty.")

    if df["instrument_id"].isna().any():
        raise ValueError("Feature output contains null `instrument_id` values.")

    if df["date"].isna().any():
        raise ValueError("Feature output contains null `date` values.")

    if df["feature_version"].isna().any():
        raise ValueError("Feature output contains null `feature_version` values.")

    duplicated = df.duplicated(subset=["instrument_id", "date"])
    if duplicated.any():
        raise ValueError("Feature output contains duplicate `instrument_id`/`date` rows.")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Feature output contains unparseable `date` values.")

    sorted_dates = df.sort_values(["instrument_id", "date"])["date"].reset_index(drop=True)
    current_dates = df["date"].reset_index(drop=True)
    if not current_dates.equals(sorted_dates):
        raise ValueError("Feature output is not sorted by `instrument_id`, `date`.")

    expected_version = _feature_cfg(settings)["feature_version"]
    if not (df["feature_version"] == expected_version).all():
        raise ValueError(
            f"Feature output contains rows with feature_version different from `{expected_version}`."
        )

    numeric_cols = [col for col in expected_columns if col not in {"instrument_id", "date", "feature_version"}]
    if numeric_cols:
        numeric_frame = df[numeric_cols].select_dtypes(include=[np.number])
        if numeric_frame.empty and len(numeric_cols) > 0:
            raise ValueError("Feature output numeric columns are not numeric as expected.")

        inf_mask = np.isinf(df[numeric_cols].to_numpy(dtype=float, copy=True))
        if inf_mask.any():
            raise ValueError("Feature output contains `inf` or `-inf` values.")

        vol_cols = [col for col in numeric_cols if col.startswith("vol_")]
        for col in vol_cols:
            if (df[col].dropna() < 0).any():
                raise ValueError(f"Feature output contains negative values in `{col}`.")

        atr_cols = [col for col in numeric_cols if col.startswith("atr_")]
        for col in atr_cols:
            if (df[col].dropna() < 0).any():
                raise ValueError(f"Feature output contains negative values in `{col}`.")

        drawdown_cols = [col for col in numeric_cols if col.startswith("drawdown_")]
        for col in drawdown_cols:
            if (df[col].dropna() > 1e-12).any():
                raise ValueError(f"Feature output contains positive values in `{col}`.")