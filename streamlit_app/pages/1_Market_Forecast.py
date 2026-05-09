from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from quant_platform.services import (
    get_symbol_market_forecast_bundle,
    list_available_symbols,
)

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from charts import (
    build_candlestick_volume_figure,
    build_forecast_figure,
)
from ui import (
    configure_page,
    decision_badge_html,
    format_date,
    format_number,
    format_signed_number,
    format_signed_percent,
    format_volume,
    render_hero,
    render_symbol_sidebar,
)


@st.cache_data(show_spinner=False)
def _load_market_forecast_bundle(symbol: str) -> dict:
    return get_symbol_market_forecast_bundle(symbol)


def main() -> None:
    dashboard_cfg = configure_page("Market & Forecast")
    symbols = list_available_symbols()
    selected_symbol = render_symbol_sidebar(dashboard_cfg, symbols)

    bundle = _load_market_forecast_bundle(selected_symbol)
    snapshot = bundle["snapshot"]
    bars_df = bundle["bars_df"].copy()
    focus_forecast_df = bundle["focus_forecast_df"].copy()
    recent_break_events_df = bundle["recent_break_events_df"].copy()

    window_start = focus_forecast_df["date"].min()
    window_end = focus_forecast_df["date"].max()
    market_window_df = bars_df.loc[
        (bars_df["date"] >= window_start) & (bars_df["date"] <= window_end)
    ].copy()
    chart_breaks_df = recent_break_events_df.loc[
        (recent_break_events_df["break_date"] >= window_start)
        & (recent_break_events_df["break_date"] <= window_end)
    ].copy()

    role_counts = focus_forecast_df["dataset_role"].value_counts().to_dict()
    latest_actual = snapshot["latest_future_rv_5d"]
    latest_benchmark = snapshot["latest_benchmark_yhat_future_rv_5d"]
    latest_ml = snapshot["latest_ml_yhat_future_rv_5d"]

    render_hero(
        "Market & Forecast",
        f"{selected_symbol}: price action, trading activity, and the 5-day volatility signal",
        (
            "The upper panel shows the asset’s market path, while the lower panel compares both forecasts over "
            "exactly the same historical window. This is an out-of-sample window: the models did not use it for "
            "estimation, so it is the relevant segment for judging how well each specification reads an unseen market."
        ),
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Asset", selected_symbol)
    metric_cols[1].metric(
        "Current lead",
        "Benchmark" if snapshot["decision"] == "do_not_promote_ml" else "ML",
    )
    metric_cols[2].metric("Last close", format_number(snapshot["latest_close"], 2))
    metric_cols[3].metric("Observed RV 5d", format_number(latest_actual, 3))
    metric_cols[4].metric(
        "Benchmark fcst",
        format_number(latest_benchmark, 3),
        delta=format_signed_number(latest_benchmark - latest_actual, 3),
    )
    metric_cols[5].metric(
        "ML fcst",
        format_number(latest_ml, 3),
        delta=format_signed_number(latest_ml - latest_actual, 3),
    )

    st.markdown(
        f"""
        <div class="section-caption">
          {decision_badge_html(snapshot["decision"])}
          <span style="margin-left: 0.8rem;">
            Evaluation window: <strong>{format_date(window_start)}</strong> to
            <strong>{format_date(window_end)}</strong> |
            validation: <strong>{role_counts.get("validation", 0)}</strong> rows |
            terminal test: <strong>{role_counts.get("test", 0)}</strong> rows
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Training = the historical estimation sample used to fit each model. "
        "Validation = the intermediate holdout used for review and tuning. "
        "Test = the final unseen segment used as the terminal verdict."
    )

    st.plotly_chart(
        build_candlestick_volume_figure(
            market_window_df,
            break_dates=chart_breaks_df["break_date"],
        ),
        use_container_width=True,
    )
    st.caption(
        "The gold dotted lines mark structural breaks detected from returns and forward volatility."
    )

    st.plotly_chart(
        build_forecast_figure(focus_forecast_df),
        use_container_width=True,
    )
    st.caption(
        "The training segment is not plotted here because the economically relevant comparison for an investor "
        "is the one that takes place on periods not seen by the models during estimation."
    )

    insight_col, detail_col = st.columns([0.9, 1.1], gap="large")
    with insight_col:
        st.markdown("### How to read this window")
        st.markdown(
            f"""
            <div class="insight-list">
              <p>The latest observation in this window belongs to the <strong>{snapshot["latest_dataset_role"]}</strong> segment and falls on
              <strong>{format_date(snapshot["latest_forecast_date"])}</strong>.</p>
              <p>Benchmark deviation from observed volatility: <strong>{format_signed_number(latest_benchmark - latest_actual, 3)}</strong><br>
              ML deviation from observed volatility: <strong>{format_signed_number(latest_ml - latest_actual, 3)}</strong></p>
              <p>Average QLIKE edge for this asset: <strong>{format_signed_percent(snapshot["relative_qlike_improvement_mean"])}</strong>.</p>
              <p>Latest structural break within the broader sample: <strong>{format_date(snapshot["most_recent_break_date"])}</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_col:
        with st.expander("Latest forecast rows", expanded=False):
            detail_df = (
                focus_forecast_df[
                    [
                        "date",
                        "dataset_role",
                        "future_rv_5d",
                        "benchmark_yhat_future_rv_5d",
                        "ml_yhat_future_rv_5d",
                    ]
                ]
                .tail(12)
                .copy()
            )
            detail_df["date"] = detail_df["date"].map(format_date)
            detail_df["future_rv_5d"] = detail_df["future_rv_5d"].map(
                lambda value: format_number(value, 3)
            )
            detail_df["benchmark_yhat_future_rv_5d"] = detail_df[
                "benchmark_yhat_future_rv_5d"
            ].map(lambda value: format_number(value, 3))
            detail_df["ml_yhat_future_rv_5d"] = detail_df["ml_yhat_future_rv_5d"].map(
                lambda value: format_number(value, 3)
            )
            detail_df = detail_df.rename(
                columns={
                    "date": "Date",
                    "dataset_role": "Segment",
                    "future_rv_5d": "Observed RV",
                    "benchmark_yhat_future_rv_5d": "Benchmark fcst",
                    "ml_yhat_future_rv_5d": "ML fcst",
                }
            )
            st.dataframe(detail_df, width="stretch", hide_index=True)

        with st.expander("Recent structural breaks", expanded=False):
            if recent_break_events_df.empty:
                st.info("No recent structural breaks were detected for this asset.")
            else:
                break_df = recent_break_events_df[
                    [
                        "event_id",
                        "break_date",
                        "previous_segment_end_date",
                        "next_segment_start_date",
                        "next_segment_end_date",
                    ]
                ].copy()
                for column in [
                    "break_date",
                    "previous_segment_end_date",
                    "next_segment_start_date",
                    "next_segment_end_date",
                ]:
                    break_df[column] = break_df[column].map(format_date)
                break_df = break_df.rename(
                    columns={
                        "event_id": "Event",
                        "break_date": "Break date",
                        "previous_segment_end_date": "End of prior regime",
                        "next_segment_start_date": "Start of new regime",
                        "next_segment_end_date": "End of new regime",
                    }
                )
                st.dataframe(break_df, width="stretch", hide_index=True)

    st.caption(
        f"Latest available bar: {format_date(snapshot['latest_bar_date'])} | "
        f"Traded volume: {format_volume(snapshot['latest_volume'])}"
    )


if __name__ == "__main__":
    main()
