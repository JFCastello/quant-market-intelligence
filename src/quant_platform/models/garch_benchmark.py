from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from arch import arch_model


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Verifica que el DataFrame contenga todas las columnas requeridas. Lanza error si falta alguna."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def _validate_benchmark_settings(settings: Mapping[str, Any]) -> None:
    """
    Valida la configuración del benchmark GARCH:
    - Claves requeridas presentes.
    - Volatilidad solo 'garch' (versión inicial).
    - Tipo de retorno: 'log' o 'simple'.
    - Parámetros numéricos positivos.
    """
    required_keys = {
        "mean_model",
        "vol_model",
        "p",
        "o",
        "q",
        "distribution",
        "input_price_column",
        "return_type",
        "return_column_name",
        "fit_scale",
        "annualization_factor",
        "forecast_horizon_days",
        "output_target_name",
        "min_train_points",
        "score_roles",
        "enforce_positive_forecasts",
        "persist_fit_params",
    }
    missing = required_keys - set(settings.keys())
    if missing:
        raise ValueError(f"Benchmark settings missing keys: {sorted(missing)}")

    if settings["vol_model"] != "garch":
        raise ValueError("Only vol_model='garch' is supported in v1.")

    if settings["return_type"] not in {"log", "simple"}:
        raise ValueError("return_type must be 'log' or 'simple'.")

    if int(settings["forecast_horizon_days"]) <= 0:
        raise ValueError("forecast_horizon_days must be > 0.")

    if float(settings["fit_scale"]) <= 0:
        raise ValueError("fit_scale must be > 0.")

    if int(settings["annualization_factor"]) <= 0:
        raise ValueError("annualization_factor must be > 0.")

    if int(settings["min_train_points"]) <= 0:
        raise ValueError("min_train_points must be > 0.")

def _resolve_first_existing_column(
    columns: set[str],
    candidates: list[str],
    logical_name: str,
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not resolve split column for '{logical_name}'. "
        f"Candidates tried: {candidates}. "
        f"Available columns: {sorted(columns)}"
    )


def _expand_interval_split_records_to_daily_roles(
    split_df: pd.DataFrame,
    available_dates: pd.Series,
) -> pd.DataFrame:
    """
    Convierte un split_df de nivel-fold (una fila por split_id con límites temporales)
    en un split_df de nivel-diario con columnas:
    - date
    - split_id
    - dataset_role

    Soporta nombres de columnas comunes para train/validation/test start/end.
    """
    _require_columns(split_df, {"split_id"}, "split_df")

    columns = set(split_df.columns)

    train_start_col = _resolve_first_existing_column(
        columns,
        ["train_start_date", "train_start", "train_date_start"],
        "train_start",
    )
    train_end_col = _resolve_first_existing_column(
        columns,
        ["train_end_date", "train_end", "train_date_end"],
        "train_end",
    )
    validation_start_col = _resolve_first_existing_column(
        columns,
        ["validation_start_date", "validation_start", "val_start_date", "val_start"],
        "validation_start",
    )
    validation_end_col = _resolve_first_existing_column(
        columns,
        ["validation_end_date", "validation_end", "val_end_date", "val_end"],
        "validation_end",
    )
    test_start_col = _resolve_first_existing_column(
        columns,
        ["test_start_date", "test_start"],
        "test_start",
    )
    test_end_col = _resolve_first_existing_column(
        columns,
        ["test_end_date", "test_end"],
        "test_end",
    )

    calendar = pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(available_dates, errors="raise"))
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        }
    )

    expanded_parts: list[pd.DataFrame] = []

    for row in split_df.itertuples(index=False):
        split_id = getattr(row, "split_id")

        train_start = pd.to_datetime(getattr(row, train_start_col), errors="raise")
        train_end = pd.to_datetime(getattr(row, train_end_col), errors="raise")
        validation_start = pd.to_datetime(getattr(row, validation_start_col), errors="raise")
        validation_end = pd.to_datetime(getattr(row, validation_end_col), errors="raise")
        test_start = pd.to_datetime(getattr(row, test_start_col), errors="raise")
        test_end = pd.to_datetime(getattr(row, test_end_col), errors="raise")

        split_calendar = calendar.copy()
        split_calendar["split_id"] = split_id
        split_calendar["dataset_role"] = pd.NA

        train_mask = split_calendar["date"].between(train_start, train_end, inclusive="both")
        validation_mask = split_calendar["date"].between(validation_start, validation_end, inclusive="both")
        test_mask = split_calendar["date"].between(test_start, test_end, inclusive="both")

        split_calendar.loc[train_mask, "dataset_role"] = "train"
        split_calendar.loc[validation_mask, "dataset_role"] = "validation"
        split_calendar.loc[test_mask, "dataset_role"] = "test"

        split_calendar = split_calendar.loc[
            split_calendar["dataset_role"].notna(),
            ["date", "split_id", "dataset_role"],
        ].copy()

        expanded_parts.append(split_calendar)

    if not expanded_parts:
        return pd.DataFrame(columns=["date", "split_id", "dataset_role"])

    expanded_df = pd.concat(expanded_parts, ignore_index=True)
    expanded_df["date"] = pd.to_datetime(expanded_df["date"], errors="raise")

    return expanded_df.sort_values(["split_id", "date"]).reset_index(drop=True)


def _normalize_split_df_for_benchmark(
    split_df: pd.DataFrame,
    available_dates: pd.Series,
) -> pd.DataFrame:
    """
    Acepta dos formatos:
    1) daily-level: date, split_id, dataset_role
    2) interval-level: una fila por split con ventanas train/validation/test

    Siempre devuelve el formato daily-level.
    """
    daily_required = {"date", "split_id", "dataset_role"}

    if daily_required.issubset(split_df.columns):
        out = split_df.loc[:, ["date", "split_id", "dataset_role"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        return out.sort_values(["split_id", "date"]).reset_index(drop=True)

    return _expand_interval_split_records_to_daily_roles(
        split_df=split_df,
        available_dates=available_dates,
    )


def _map_mean_model(mean_model: str) -> str:
    """Convierte el nombre del modelo de media al formato esperado por arch."""
    mapping = {
        "zero": "Zero",
        "constant": "Constant",
    }
    if mean_model not in mapping:
        raise ValueError(f"Unsupported mean_model: {mean_model}")
    return mapping[mean_model]


def _map_distribution(distribution: str) -> str:
    """Convierte el nombre de la distribución al formato esperado por arch."""
    mapping = {
        "normal": "normal",
        "gaussian": "normal",
        "studentst": "t",
        "t": "t",
        "skewt": "skewt",
        "ged": "ged",
    }
    if distribution not in mapping:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return mapping[distribution]


def build_benchmark_input_df(
    normalized_df: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Construye la serie base para el benchmark GARCH a partir de precios normalizados.
    - Valida configuración y columnas.
    - Calcula retornos diarios (log o simple) y los escala según fit_scale.
    - Retorna un DataFrame con columnas: date, precio, retorno escalado, y opcionalmente symbol.
    """
    _validate_benchmark_settings(settings)

    price_col = settings["input_price_column"]
    return_col = settings["return_column_name"]
    fit_scale = float(settings["fit_scale"])
    return_type = settings["return_type"]

    _require_columns(normalized_df, {"date", price_col}, "normalized_df")

    df = normalized_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    price_series = pd.to_numeric(df[price_col], errors="raise").astype(float)

    if (price_series <= 0).any():
        raise ValueError(f"All values in '{price_col}' must be positive to build returns.")

    # Calcular retornos según tipo
    if return_type == "log":
        returns = np.log(price_series / price_series.shift(1))
    else:
        returns = price_series.pct_change()

    # Escalar retornos para estabilidad numérica en el ajuste GARCH
    df[return_col] = returns * fit_scale

    keep_cols = ["date", price_col, return_col]
    if "symbol" in df.columns:
        keep_cols.append("symbol")

    return df[keep_cols].copy()


def fit_garch_on_train_returns(
    train_returns: pd.Series,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Ajusta un modelo GARCH(p,o,q) con la distribución especificada sobre los retornos de entrenamiento.
    Retorna un diccionario con:
    - result: objeto de resultado de arch
    - params: parámetros limpios (omega, alpha, beta, nu)
    - mu: media estimada (si mean_model = 'constant')
    - sigma2_last: última varianza condicional in-sample
    - epsilon_last: último residuo observado
    - n_train: número de puntos usados
    - fit_status: 'converged' o 'not_converged'
    """
    _validate_benchmark_settings(settings)

    clean_train = pd.Series(train_returns).dropna().astype(float)

    min_train_points = int(settings["min_train_points"])
    if len(clean_train) < min_train_points:
        raise ValueError(
            f"Not enough train points for GARCH fit: {len(clean_train)} < {min_train_points}"
        )

    mean_model = _map_mean_model(settings["mean_model"])
    distribution = _map_distribution(settings["distribution"])

    model = arch_model(
        clean_train,
        mean=mean_model,
        vol="GARCH",
        p=int(settings["p"]),
        o=int(settings["o"]),
        q=int(settings["q"]),
        dist=distribution,
        rescale=False,
    )

    result = model.fit(disp="off", show_warning=False)

    params = result.params.to_dict()

    # Extraer media (mu) si el modelo es constante
    mu = 0.0
    if settings["mean_model"] == "constant":
        mu = float(params.get("mu", 0.0))

    conditional_vol = pd.Series(result.conditional_volatility, index=clean_train.index).astype(float)
    sigma2_last = float(conditional_vol.iloc[-1] ** 2)

    epsilon_last = float(clean_train.iloc[-1] - mu)

    # Limpiar parámetros GARCH
    clean_params = {
        "omega": float(params.get("omega", np.nan)),
        "alpha_1": float(params.get("alpha[1]", np.nan)),
        "beta_1": float(params.get("beta[1]", np.nan)),
        "nu": float(params.get("nu", np.nan)) if "nu" in params else np.nan,
    }

    return {
        "result": result,
        "params": clean_params,
        "mu": mu,
        "sigma2_last": sigma2_last,
        "epsilon_last": epsilon_last,
        "n_train": int(len(clean_train)),
        "fit_status": "converged" if getattr(result, "convergence_flag", 0) == 0 else "not_converged",
    }


def _forecast_variance_path_from_observed_state(
    epsilon_t: float,
    sigma2_t: float,
    omega: float,
    alpha_1: float,
    beta_1: float,
    horizon: int,
) -> np.ndarray:
    """
    Dado el estado observado al cierre del día t:
    - epsilon_t = residuo observado
    - sigma2_t  = varianza condicional de ese día
    produce la trayectoria de varianzas pronosticadas para t+1 ... t+horizon.
    Para h=1 usa la ecuación GARCH(1,1) con el residuo observado.
    Para h>=2 usa la esperanza condicional estándar (persistencia).
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    path = np.zeros(horizon, dtype=float)

    # Primer paso: usa el residuo observado
    next_sigma2 = omega + alpha_1 * (epsilon_t ** 2) + beta_1 * sigma2_t
    path[0] = max(next_sigma2, 0.0)

    persistence = alpha_1 + beta_1
    for i in range(1, horizon):
        next_sigma2 = omega + persistence * path[i - 1]
        path[i] = max(next_sigma2, 0.0)

    return path


def _annualize_forecast_from_variance_path(
    daily_variance_path_scaled: np.ndarray,
    annualization_factor: int,
    horizon_days: int,
    fit_scale: float,
    enforce_positive_forecasts: bool,
) -> float:
    """
    Convierte la suma de varianzas diarias pronosticadas (escaladas) en una volatilidad
    acumulada a horizonte fijo (ej. 5 días) y luego anualizada.
    - Desescala las varianzas dividiendo por fit_scale^2.
    - Calcula varianza anualizada: (annual_factor / horizon) * suma_varianzas_desescaladas.
    - Retorna la raíz cuadrada (volatilidad).
    """
    variance_sum_scaled = float(np.sum(daily_variance_path_scaled))
    variance_sum_unscaled = variance_sum_scaled / (fit_scale ** 2)

    annualized_variance = (annualization_factor / horizon_days) * variance_sum_unscaled
    forecast_vol = float(np.sqrt(max(annualized_variance, 0.0)))

    if enforce_positive_forecasts and forecast_vol <= 0:
        raise ValueError("Non-positive volatility forecast encountered.")

    return forecast_vol


def build_garch_benchmark_forecasts_by_split(
    normalized_df: pd.DataFrame,
    split_df: pd.DataFrame,
    settings: Mapping[str, Any],
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Construye forecasts benchmark GARCH por split.

    Acepta split_df en dos formatos:
    1) daily-level: date, split_id, dataset_role
    2) interval-level: una fila por split con ventanas train/validation/test

    Diseño metodológico de v1:
    - ajusta parámetros SOLO con train
    - congela parámetros del split
    - recorre validation/test secuencialmente usando los retornos ya observados
      para actualizar el estado condicional
    - emite un forecast continuo a horizonte fijo (5d) por cada fecha scored
    """
    _validate_benchmark_settings(settings)

    benchmark_input_df = build_benchmark_input_df(
        normalized_df=normalized_df,
        settings=settings,
    )

    split_df_prepared = _normalize_split_df_for_benchmark(
        split_df=split_df,
        available_dates=benchmark_input_df["date"],
    )

    merged = benchmark_input_df.merge(
        split_df_prepared,
        on="date",
        how="inner",
        validate="one_to_many",
    )

    merged["date"] = pd.to_datetime(merged["date"], errors="raise")
    merged = merged.sort_values(["split_id", "date"]).reset_index(drop=True)

    return_col = settings["return_column_name"]
    score_roles = set(settings["score_roles"])
    horizon_days = int(settings["forecast_horizon_days"])
    annualization_factor = int(settings["annualization_factor"])
    fit_scale = float(settings["fit_scale"])
    enforce_positive = bool(settings["enforce_positive_forecasts"])

    symbol_value = symbol
    if symbol_value is None and "symbol" in merged.columns and merged["symbol"].notna().any():
        symbol_value = str(merged["symbol"].dropna().iloc[0])

    records: list[dict[str, Any]] = []

    for split_id, split_slice in merged.groupby("split_id", sort=True):
        split_slice = split_slice.sort_values("date").reset_index(drop=True)

        train_df = split_slice.loc[split_slice["dataset_role"] == "train"].copy()
        score_df = split_slice.loc[split_slice["dataset_role"].isin(score_roles)].copy()

        train_returns = train_df[return_col].dropna()
        if train_returns.empty:
            raise ValueError(f"Split {split_id} has no non-null train returns.")

        fit_info = fit_garch_on_train_returns(
            train_returns=train_returns,
            settings=settings,
        )

        omega = fit_info["params"]["omega"]
        alpha_1 = fit_info["params"]["alpha_1"]
        beta_1 = fit_info["params"]["beta_1"]
        nu = fit_info["params"]["nu"]
        mu = float(fit_info["mu"])

        sigma2_current = (
            omega
            + alpha_1 * (fit_info["epsilon_last"] ** 2)
            + beta_1 * fit_info["sigma2_last"]
        )
        sigma2_current = max(float(sigma2_current), 0.0)

        score_df = score_df.loc[score_df[return_col].notna()].copy()
        if score_df.empty:
            continue

        train_start_date = pd.to_datetime(train_df["date"].min())
        train_end_date = pd.to_datetime(train_df["date"].max())

        for row in score_df.itertuples(index=False):
            row_date = pd.to_datetime(row.date)
            dataset_role = str(row.dataset_role)
            observed_return = float(getattr(row, return_col))
            epsilon_t = observed_return - mu

            variance_path_scaled = _forecast_variance_path_from_observed_state(
                epsilon_t=epsilon_t,
                sigma2_t=sigma2_current,
                omega=omega,
                alpha_1=alpha_1,
                beta_1=beta_1,
                horizon=horizon_days,
            )

            yhat_future_rv_5d = _annualize_forecast_from_variance_path(
                daily_variance_path_scaled=variance_path_scaled,
                annualization_factor=annualization_factor,
                horizon_days=horizon_days,
                fit_scale=fit_scale,
                enforce_positive_forecasts=enforce_positive,
            )

            records.append(
                {
                    "symbol": symbol_value,
                    "date": row_date,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": settings.get("benchmark_name", "garch_11_student_t"),
                    "benchmark_version": settings.get("benchmark_version", "v1"),
                    "forecast_horizon_days": horizon_days,
                    "output_target_name": settings["output_target_name"],
                    "yhat_future_rv_5d": yhat_future_rv_5d,
                    "train_start_date": train_start_date,
                    "train_end_date": train_end_date,
                    "n_train": fit_info["n_train"],
                    "fit_status": fit_info["fit_status"],
                    "omega": omega,
                    "alpha_1": alpha_1,
                    "beta_1": beta_1,
                    "nu": nu,
                }
            )

            sigma2_current = float(variance_path_scaled[0])

    forecast_df = pd.DataFrame.from_records(records)

    if forecast_df.empty:
        return forecast_df

    forecast_df = forecast_df.sort_values(["split_id", "date"]).reset_index(drop=True)
    return forecast_df