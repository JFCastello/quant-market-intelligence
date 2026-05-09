from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from quant_platform.services import get_executive_summary_bundle

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from charts import build_universe_tradeoff_figure
from ui import (
    configure_page,
    decision_badge_html,
    format_date,
    format_signed_percent,
    render_hero,
    render_story_card,
)


@st.cache_data(show_spinner=False)
def _load_home_bundle() -> dict:
    return get_executive_summary_bundle()


def main() -> None:
    dashboard_cfg = configure_page("Executive Summary")
    bundle = _load_home_bundle()

    asset_summary_df = bundle["asset_summary_df"].copy()
    decision_summary_df = bundle["decision_summary_df"].copy()
    kpis = bundle["kpis"]

    symbols = asset_summary_df["symbol"].sort_values().tolist()
    default_symbol = str(dashboard_cfg["defaults"]["symbol"]).upper()
    default_index = symbols.index(default_symbol) if default_symbol in symbols else 0
    focus_symbol = st.sidebar.selectbox(
        "Reference asset",
        options=symbols,
        index=default_index,
    )
    focus_row = asset_summary_df.loc[asset_summary_df["symbol"] == focus_symbol].iloc[0]

    render_hero(
        "Executive Summary",
        "Which volatility engine offers the more reliable read on near-term market risk",
        (
            "This dashboard compares two volatility-forecasting engines for the next five trading sessions "
            "across SPY, TLT, GLD, and HYG. The current evidence still favors the classical benchmark as the "
            "primary reference model, while the ML specification remains under evaluation to determine whether "
            "it delivers a stable and economically meaningful out-of-sample gain."
        ),
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Assets", kpis["total_assets"])
    kpi_cols[1].metric("ML leading", kpis["promote_count"])
    kpi_cols[2].metric("Benchmark leading", kpis["benchmark_preferred_count"])
    kpi_cols[3].metric("Regime probabilities", "N/A")

    context_cols = st.columns(4)
    with context_cols[0]:
        render_story_card(
            "Universe",
            "SPY · TLT · GLD · HYG",
            "Liquid ETFs spanning equities, duration, gold, and credit.",
        )
    with context_cols[1]:
        render_story_card(
            "Horizon",
            "5 trading days",
            "The core question is how much realized volatility is likely over the next trading week.",
        )
    with context_cols[2]:
        render_story_card(
            "Classical engine",
            "GARCH(1,1) Student-t",
            "Acts as a conservative reference model for judging whether ML truly adds value.",
        )
    with context_cols[3]:
        render_story_card(
            "ML engine",
            "XGBoost Regressor",
            "Targets nonlinear structure without giving up strict out-of-sample discipline.",
        )

    insight_col, chart_col = st.columns([0.84, 1.16], gap="large")
    with insight_col:
        st.markdown("### Asset in focus")
        st.markdown(
            decision_badge_html(str(focus_row["decision"])), unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="insight-list">
              <p><strong>{focus_symbol}</strong> still shows a market reading in which the benchmark remains the more robust reference specification.</p>
              <p>QLIKE edge: <strong>{format_signed_percent(focus_row["relative_qlike_improvement_mean"])}</strong><br>
              Macro-F1 delta: <strong>{format_signed_percent(focus_row["macro_f1_delta"])}</strong><br>
              Balanced Accuracy delta: <strong>{format_signed_percent(focus_row["balanced_accuracy_delta"])}</strong></p>
              <p>Latest valid target date: <strong>{format_date(focus_row["latest_target_date"])}</strong><br>
              Latest structural break: <strong>{format_date(focus_row["most_recent_break_date"])}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The upper-right quadrant would indicate that ML improves both the continuous volatility forecast "
            "and the regime-classification layer. At present, no asset reaches that pattern with enough clarity."
        )

    with chart_col:
        st.plotly_chart(
            build_universe_tradeoff_figure(asset_summary_df),
            use_container_width=True,
        )

    st.markdown("### Cross-asset signal")
    st.markdown(
        '<div class="section-caption">Compact readout for the evaluation universe.</div>',
        unsafe_allow_html=True,
    )
    display_df = decision_summary_df[
        [
            "symbol",
            "decision",
            "relative_qlike_improvement_mean",
            "macro_f1_delta",
            "balanced_accuracy_delta",
            "calibration_status",
        ]
    ].copy()
    display_df["relative_qlike_improvement_mean"] = display_df[
        "relative_qlike_improvement_mean"
    ].map(format_signed_percent)
    display_df["macro_f1_delta"] = display_df["macro_f1_delta"].map(
        format_signed_percent
    )
    display_df["balanced_accuracy_delta"] = display_df["balanced_accuracy_delta"].map(
        format_signed_percent
    )
    display_df["calibration_status"] = display_df["calibration_status"].map(
        lambda value: "Probabilities not published"
        if value == "not_evaluable_yet"
        else str(value)
    )
    display_df = display_df.rename(
        columns={
            "symbol": "Asset",
            "decision": "Leading model",
            "relative_qlike_improvement_mean": "QLIKE edge",
            "macro_f1_delta": "Delta Macro-F1",
            "balanced_accuracy_delta": "Delta Bal. Acc.",
            "calibration_status": "Regime probabilities",
        }
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

    with st.expander("How to read the metrics", expanded=False):
        st.markdown(
            "For an investor, these diagnostics answer two questions: "
            "which model makes smaller errors when forecasting volatility magnitude, "
            "and which model classifies the market state more reliably."
        )
        st.markdown("**Volatility-forecast loss metrics: lower is better.**")
        st.latex(r"e_t = \widehat{RV}_t - RV_t")
        st.markdown(
            r"Here, $e_t$ is the forecast error at date $t$, $\widehat{RV}_t$ is the model forecast "
            r"for forward realized volatility, and $RV_t$ is the realized forward volatility that was "
            r"actually observed."
        )
        st.latex(r"MAE = \frac{1}{n}\sum_{t=1}^{n}|e_t|")
        st.markdown(
            r"Here, $n$ is the number of forecast observations and $|e_t|$ is the absolute forecast error."
        )
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}e_t^2}")
        st.markdown(
            r"Here, $e_t^2$ is the squared forecast error, so RMSE gives more weight to larger misses."
        )
        st.latex(
            r"QLIKE = \frac{1}{n}\sum_{t=1}^{n}\left(\frac{RV_t}{\widehat{RV}_t} - \log\frac{RV_t}{\widehat{RV}_t} - 1\right)"
        )
        st.markdown(
            "QLIKE penalizes volatility-scale misspecification more sharply, "
            "which is why it is the project’s primary loss function."
        )
        st.markdown(
            r"In the QLIKE expression, $RV_t / \widehat{RV}_t$ is the ratio between realized and forecast volatility, "
            r"$\log(\cdot)$ is the natural logarithm, and $n$ is again the number of forecast observations."
        )
        st.markdown("**Regime-classification quality metrics: higher is better.**")
        st.latex(
            r"Macro\text{-}F1 = \frac{1}{K}\sum_{k=1}^{K}F1_k,\qquad F1_k=\frac{2P_kR_k}{P_k+R_k}"
        )
        st.markdown(
            r"Here, $K$ is the number of regimes, $F1_k$ is the F1 score for regime $k$, "
            r"$P_k$ is precision for regime $k$, and $R_k$ is recall for regime $k$."
        )
        st.latex(r"Balanced\ Accuracy = \frac{1}{K}\sum_{k=1}^{K}Recall_k")
        st.markdown(
            "Macro-F1 and Balanced Accuracy help determine whether the model recognizes "
            "Calm, Normal, and Stress regimes more effectively without allowing one class to dominate the score."
        )
        st.markdown(
            r"Here, $Recall_k$ is the share of observations in regime $k$ that the model classifies correctly, "
            r"and $K$ is the number of regimes being averaged."
        )

    st.markdown("### Dashboard flow")
    nav_specs = [
        (
            "pages/1_Market_Forecast.py",
            "Market & Forecast",
            "Inspect price, volume, and the direct comparison between observed volatility and both forecasts.",
        ),
        (
            "pages/2_Model_Comparison.py",
            "Model Comparison",
            "Examine in greater detail which model commits smaller errors and which one classifies regimes more accurately.",
        ),
        (
            "pages/3_Structural_Changes.py",
            "Structural Changes",
            "Add context around the points in time when market behavior shifted materially.",
        ),
    ]
    nav_cols = st.columns(3)
    for col, (path, title, body) in zip(nav_cols, nav_specs):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
                st.page_link(path, label=f"Open {title}")


if __name__ == "__main__":
    main()
