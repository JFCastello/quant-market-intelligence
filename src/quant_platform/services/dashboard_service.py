from __future__ import annotations

from typing import Any

import pandas as pd

from .artifact_loaders import list_available_symbols, load_decision_summary
from .overview_service import build_symbol_overview_snapshot


def _build_calibration_counts(decision_summary_df: pd.DataFrame) -> dict[str, int]:
    """
    Construye conteos de estados de calibración con semántica correcta.

    Propósito:
    Evitar la confusión existente entre el nombre `calibration_pending_count`
    y un conteo que en realidad no correspondía a pendientes.
    """
    calibration_status = decision_summary_df["calibration_status"].astype(str)

    return {
        "calibration_pending_count": int(
            (calibration_status == "pending_evaluation").sum()
        ),
        "calibration_not_evaluable_count": int(
            (calibration_status == "not_evaluable_yet").sum()
        ),
        "calibration_other_count": int(
            (
                ~calibration_status.isin(["pending_evaluation", "not_evaluable_yet"])
            ).sum()
        ),
    }


def get_executive_summary_bundle() -> dict[str, Any]:
    """
    Construye el bundle principal para la Home ejecutiva del dashboard.

    Propósito:
    Reunir en una sola estructura:
    - el summary global de decisión,
    - un resumen corto por activo con fechas y breaks recientes,
    - KPIs agregados del universo.
    """
    decision_summary_df = load_decision_summary().copy()
    decision_summary_df = decision_summary_df.sort_values("symbol").reset_index(
        drop=True
    )

    asset_rows: list[dict[str, Any]] = []
    for symbol in list_available_symbols():
        snapshot = build_symbol_overview_snapshot(symbol)
        asset_rows.append(
            {
                "symbol": snapshot["symbol"],
                "decision": snapshot["decision"],
                "latest_target_date": snapshot["latest_target_date"],
                "latest_future_rv_5d": snapshot["latest_future_rv_5d"],
                "most_recent_break_date": snapshot["most_recent_break_date"],
                "recent_break_count": snapshot["recent_break_count"],
            }
        )

    asset_snapshot_df = pd.DataFrame(asset_rows)

    if not asset_snapshot_df.empty:
        asset_snapshot_df["latest_target_date"] = pd.to_datetime(
            asset_snapshot_df["latest_target_date"],
            errors="raise",
        )
        asset_snapshot_df["most_recent_break_date"] = pd.to_datetime(
            asset_snapshot_df["most_recent_break_date"],
            errors="coerce",
        )

    asset_summary_df = decision_summary_df.merge(
        asset_snapshot_df,
        on=["symbol", "decision"],
        how="left",
        validate="one_to_one",
    )

    calibration_counts = _build_calibration_counts(decision_summary_df)

    kpis = {
        "total_assets": int(len(decision_summary_df)),
        "promote_count": int((decision_summary_df["decision"] == "promote_ml").sum()),
        "benchmark_preferred_count": int(
            (decision_summary_df["decision"] == "do_not_promote_ml").sum()
        ),
        "qlike_win_count": int(
            (decision_summary_df["relative_qlike_improvement_mean"] > 0).sum()
        ),
        "discrete_guardrail_pass_count": int(
            (
                (decision_summary_df["macro_f1_delta"] >= 0)
                & (decision_summary_df["balanced_accuracy_delta"] >= 0)
            ).sum()
        ),
        **calibration_counts,
    }

    return {
        "kpis": kpis,
        "decision_summary_df": decision_summary_df,
        "asset_summary_df": asset_summary_df,
    }
