from __future__ import annotations

from typing import Any

import pandas as pd

from .artifact_loaders import (
    load_symbol_benchmark_forecasts,
    load_symbol_normalized_bars,
    load_symbol_ml_forecasts,
)
from .overview_service import (
    build_symbol_overview_snapshot,
    get_symbol_decision_summary,
    get_symbol_recent_break_events,
)
from .settings import load_settings


def get_symbol_market_bars(symbol: str) -> pd.DataFrame:
    """
    Carga y ordena las barras de mercado OHLCV de un símbolo.
    """
    bars_df = load_symbol_normalized_bars(symbol).copy()
    bars_df["date"] = pd.to_datetime(bars_df["date"], errors="raise")
    return bars_df.sort_values("date").reset_index(drop=True)


def get_symbol_forecast_comparison_timeseries(symbol: str) -> pd.DataFrame:
    """
    Construye una serie temporal conjunta con observado, benchmark y ML.

    Propósito:
    Dejar lista la materia prima para la vista de forecast continuo de la UI,
    evitando que Streamlit tenga que hacer merges entre artefactos.
    """
    benchmark_df = load_symbol_benchmark_forecasts(symbol).copy()
    ml_df = load_symbol_ml_forecasts(symbol).copy()

    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], errors="raise")
    ml_df["date"] = pd.to_datetime(ml_df["date"], errors="raise")

    join_keys = ["symbol", "date", "split_id", "dataset_role"]

    benchmark_cols = join_keys + [
        "model_name",
        "yhat_future_rv_5d",
        "train_start_date",
        "train_end_date",
    ]
    ml_cols = join_keys + [
        "model_name",
        "future_rv_5d",
        "yhat_future_rv_5d",
        "best_iteration",
        "best_score",
        "feature_count",
    ]

    merged_df = benchmark_df[benchmark_cols].merge(
        ml_df[ml_cols],
        on=join_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_benchmark", "_ml"),
    )

    merged_df = merged_df.rename(
        columns={
            "model_name_benchmark": "benchmark_model_name",
            "model_name_ml": "ml_model_name",
            "yhat_future_rv_5d_benchmark": "benchmark_yhat_future_rv_5d",
            "yhat_future_rv_5d_ml": "ml_yhat_future_rv_5d",
        }
    )

    merged_df["train_start_date"] = pd.to_datetime(
        merged_df["train_start_date"],
        errors="coerce",
    )
    merged_df["train_end_date"] = pd.to_datetime(
        merged_df["train_end_date"],
        errors="coerce",
    )

    return merged_df.sort_values(["date", "split_id", "dataset_role"]).reset_index(
        drop=True
    )


def _get_latest_forecast_row(forecast_df: pd.DataFrame) -> pd.Series:
    """
    Devuelve la fila de forecast más reciente, privilegiando `test`.
    """
    if forecast_df.empty:
        raise ValueError("Received empty forecast dataframe.")

    sorted_df = forecast_df.sort_values(
        ["date", "split_id", "dataset_role"]
    ).reset_index(drop=True)

    preferred_df = sorted_df.loc[sorted_df["dataset_role"] == "test"].copy()
    if not preferred_df.empty:
        return preferred_df.iloc[-1]

    return sorted_df.iloc[-1]


def build_symbol_market_forecast_snapshot(symbol: str) -> dict[str, Any]:
    """
    Construye un snapshot ejecutivo para la página Market & Forecast.
    """
    bars_df = get_symbol_market_bars(symbol)
    forecast_df = get_symbol_forecast_comparison_timeseries(symbol)
    overview_snapshot = build_symbol_overview_snapshot(symbol)
    decision_row = get_symbol_decision_summary(symbol)
    latest_forecast_row = _get_latest_forecast_row(forecast_df)
    latest_bar_row = bars_df.iloc[-1]

    return {
        "symbol": str(symbol).upper(),
        "decision": str(decision_row["decision"]),
        "latest_close": float(latest_bar_row["close"]),
        "latest_volume": float(latest_bar_row["volume"]),
        "latest_bar_date": pd.to_datetime(latest_bar_row["date"]),
        "latest_forecast_date": pd.to_datetime(latest_forecast_row["date"]),
        "latest_dataset_role": str(latest_forecast_row["dataset_role"]),
        "latest_split_id": str(latest_forecast_row["split_id"]),
        "latest_future_rv_5d": float(latest_forecast_row["future_rv_5d"]),
        "latest_benchmark_yhat_future_rv_5d": float(
            latest_forecast_row["benchmark_yhat_future_rv_5d"]
        ),
        "latest_ml_yhat_future_rv_5d": float(
            latest_forecast_row["ml_yhat_future_rv_5d"]
        ),
        "relative_qlike_improvement_mean": float(
            decision_row["relative_qlike_improvement_mean"]
        ),
        "macro_f1_delta": float(decision_row["macro_f1_delta"]),
        "balanced_accuracy_delta": float(decision_row["balanced_accuracy_delta"]),
        "most_recent_break_date": overview_snapshot["most_recent_break_date"],
    }


def get_symbol_market_forecast_bundle(symbol: str) -> dict[str, Any]:
    """
    Devuelve un bundle listo para la página Market & Forecast.

    Propósito:
    Reunir contexto de mercado, serie de forecasts y señales narrativas
    mínimas sin reimplementar lógica en Streamlit.
    """
    forecast_df = get_symbol_forecast_comparison_timeseries(symbol)

    if forecast_df.empty:
        raise ValueError(f"No forecast rows available for symbol={symbol}")

    latest_split_id = str(
        forecast_df["split_id"].dropna().astype(str).sort_values().iloc[-1]
    )
    focus_forecast_df = forecast_df.loc[
        forecast_df["split_id"] == latest_split_id
    ].copy()

    settings = load_settings()
    break_limit = int(
        settings["services_layer"]["defaults"]["latest_break_events_limit"]
    )

    return {
        "snapshot": build_symbol_market_forecast_snapshot(symbol),
        "decision_summary_row": get_symbol_decision_summary(symbol),
        "bars_df": get_symbol_market_bars(symbol),
        "forecast_df": forecast_df,
        "focus_forecast_df": focus_forecast_df,
        "focus_split_id": latest_split_id,
        "recent_break_events_df": get_symbol_recent_break_events(
            symbol, limit=break_limit
        ),
    }
