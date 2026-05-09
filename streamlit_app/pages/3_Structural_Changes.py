from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from quant_platform.services import (
    get_symbol_market_forecast_bundle,
    get_symbol_structural_changes_bundle,
    list_available_symbols,
)

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from charts import build_structural_break_figure
from ui import (
    configure_page,
    format_date,
    format_number,
    render_hero,
    render_symbol_sidebar,
)


@st.cache_data(show_spinner=False)
def _load_structural_bundle(symbol: str) -> dict:
    return get_symbol_structural_changes_bundle(symbol)


@st.cache_data(show_spinner=False)
def _load_market_bundle(symbol: str) -> dict:
    return get_symbol_market_forecast_bundle(symbol)


def main() -> None:
    dashboard_cfg = configure_page("Structural Changes")
    symbols = list_available_symbols()
    selected_symbol = render_symbol_sidebar(dashboard_cfg, symbols)

    bundle = _load_structural_bundle(selected_symbol)
    market_bundle = _load_market_bundle(selected_symbol)
    summary = bundle["summary"]
    signal_df = bundle["signal_df"].copy()
    events_df = bundle["events_df"].copy()
    recent_events_df = bundle["recent_events_df"].copy()
    focus_forecast_df = market_bundle["focus_forecast_df"].copy()

    window_start = focus_forecast_df["date"].min()
    window_end = focus_forecast_df["date"].max()
    chart_signal_df = signal_df.loc[
        (signal_df["date"] >= window_start) & (signal_df["date"] <= window_end)
    ].copy()
    chart_events_df = events_df.loc[
        (events_df["break_date"] >= window_start)
        & (events_df["break_date"] <= window_end)
    ].copy()

    render_hero(
        "Structural Changes",
        f"{selected_symbol}: when did market behavior shift materially?",
        (
            "Here, a structural change is defined as a breakpoint at which the joint process formed by daily returns "
            "and 5-day forward volatility ceases to behave like the preceding segment under the PELT detector. "
            "This does not determine the winning model by itself, but it helps explain why forecasting difficulty "
            "changes over time."
        ),
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Breaks", summary["event_count"])
    metric_cols[1].metric("Latest break", format_date(summary["most_recent_break_date"]))
    metric_cols[2].metric("In window", int(len(chart_events_df)))
    metric_cols[3].metric("RV 5d", format_number(summary["latest_future_rv_5d"], 3))
    metric_cols[4].metric("Return 1d", format_number(summary["latest_log_ret_1d"], 3))

    st.markdown(
        f"""
        <div class="section-caption">
          Window aligned with Market & Forecast:
          <strong>{format_date(window_start)}</strong> a <strong>{format_date(window_end)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        build_structural_break_figure(chart_signal_df, chart_events_df),
        use_container_width=True,
    )

    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.markdown("### Recent breaks")
        if recent_events_df.empty:
            st.info("No structural breaks were detected for this asset.")
        else:
            display_df = recent_events_df[
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
                display_df[column] = display_df[column].map(format_date)
            display_df = display_df.rename(
                columns={
                    "event_id": "Event",
                    "break_date": "Break date",
                    "previous_segment_end_date": "End of prior regime",
                    "next_segment_start_date": "Start of new regime",
                    "next_segment_end_date": "End of new regime",
                }
            )
            st.dataframe(display_df, width="stretch", hide_index=True)

    with right_col:
        st.markdown("### How to read this signal")
        st.markdown(
            f"""
            <div class="insight-list">
              <p>The plotted lines are precisely the signals used by the detector: daily return and forward volatility, both standardized.</p>
              <p>Latest detected break: <strong>{format_date(summary["most_recent_break_date"])}</strong>.</p>
              <p>If the market shifts regime, it is economically plausible that the relative accuracy of each model shifts as well.</p>
              <p>That is why these breaks serve as interpretive context rather than a substitute for formal forecast comparison.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Full event table", expanded=False):
        st.markdown(
            "Main columns: `event_id` identifies the event; `break_date` is the breakpoint date; "
            "`previous_segment_start_date` and `previous_segment_end_date` delimit the prior regime; "
            "`next_segment_start_date` and `next_segment_end_date` delimit the new regime; "
            "`method`, `algorithm`, and `cost_model` describe the detector; `detected_at` records artifact generation time."
        )
        full_df = events_df.copy()
        for column in [
            "break_date",
            "previous_segment_start_date",
            "previous_segment_end_date",
            "next_segment_start_date",
            "next_segment_end_date",
            "detected_at",
        ]:
            if column in full_df.columns:
                full_df[column] = full_df[column].map(format_date)
        st.dataframe(full_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
