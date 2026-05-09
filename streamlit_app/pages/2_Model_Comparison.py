from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from quant_platform.services import (
    get_symbol_decision_summary,
    get_symbol_model_comparison_dashboard_bundle,
    list_available_symbols,
)

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from charts import (
    build_confusion_matrix_figure,
    build_metric_relative_improvement_figure,
    build_role_metric_heatmap,
)
from ui import (
    configure_page,
    decision_badge_html,
    format_signed_percent,
    render_hero,
    render_symbol_sidebar,
)


@st.cache_data(show_spinner=False)
def _load_comparison_bundle(symbol: str) -> dict:
    return get_symbol_model_comparison_dashboard_bundle(symbol)


@st.cache_data(show_spinner=False)
def _load_decision_row(symbol: str) -> dict:
    return get_symbol_decision_summary(symbol).to_dict()


def main() -> None:
    dashboard_cfg = configure_page("Model Comparison")
    symbols = list_available_symbols()
    selected_symbol = render_symbol_sidebar(dashboard_cfg, symbols)

    bundle = _load_comparison_bundle(selected_symbol)
    decision_row = _load_decision_row(selected_symbol)

    summary_df = bundle["summary_df"].copy()
    role_summary_df = bundle["role_summary_df"].copy()
    confusion_summary_df = bundle["confusion_summary_df"].copy()
    pivot_df = bundle["pivot_df"].copy()
    panel_df = bundle["panel_df"].copy()

    dataset_roles = (
        confusion_summary_df["dataset_role"].dropna().drop_duplicates().tolist()
    )
    default_role = "test" if "test" in dataset_roles else dataset_roles[0]

    render_hero(
        "Model Comparison",
        f"{selected_symbol}: which specification read volatility and market regimes more effectively",
        (
            "This page compresses the competition between the classical engine and the ML engine into two dimensions: "
            "which model forecast the magnitude of volatility more accurately, and which one classified the market state "
            "more reliably as Calm, Normal, or Stress."
        ),
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Leading model",
        "Benchmark" if decision_row["decision"] == "do_not_promote_ml" else "ML",
    )
    metric_cols[1].metric(
        "QLIKE edge",
        format_signed_percent(decision_row["relative_qlike_improvement_mean"]),
    )
    metric_cols[2].metric(
        "Macro-F1 delta",
        format_signed_percent(decision_row["macro_f1_delta"]),
    )
    metric_cols[3].metric(
        "Bal. Acc. delta",
        format_signed_percent(decision_row["balanced_accuracy_delta"]),
    )

    st.markdown(
        f"""
        <div class="section-caption">
          {decision_badge_html(str(decision_row["decision"]))}
          <span style="margin-left: 0.8rem;">
            Training = estimation sample used to fit the model. Validation = intermediate holdout used for review and tuning.
            Test = final unseen segment used as the terminal verdict.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    visual_cols = st.columns(2, gap="large")
    with visual_cols[0]:
        st.plotly_chart(
            build_metric_relative_improvement_figure(summary_df),
            use_container_width=True,
        )
    with visual_cols[1]:
        st.plotly_chart(
            build_role_metric_heatmap(role_summary_df),
            use_container_width=True,
        )

    st.markdown("### Regime classification")
    selected_role = st.radio(
        "Evaluation segment",
        options=dataset_roles,
        horizontal=True,
        index=dataset_roles.index(default_role),
        format_func=lambda role: "Validation" if role == "validation" else "Terminal test",
    )

    benchmark_model_name = str(summary_df["benchmark_model_name"].iloc[0])
    ml_model_name = str(summary_df["ml_model_name"].iloc[0])

    confusion_cols = st.columns(2, gap="large")
    with confusion_cols[0]:
        st.plotly_chart(
            build_confusion_matrix_figure(
                confusion_summary_df,
                model_name=benchmark_model_name,
                dataset_role=selected_role,
            ),
            use_container_width=True,
        )
    with confusion_cols[1]:
        st.plotly_chart(
            build_confusion_matrix_figure(
                confusion_summary_df,
                model_name=ml_model_name,
                dataset_role=selected_role,
            ),
            use_container_width=True,
        )

    st.caption(
        "Color = row share within each observed regime. Text = count and row share."
    )

    st.markdown("### Regime definition")
    regime_cols = st.columns(3)
    with regime_cols[0]:
        st.markdown("**Calm**")
        st.caption(
            "Lower tercile of observed forward volatility in the training sample. "
            "Represents a relatively stable market environment."
        )
    with regime_cols[1]:
        st.markdown("**Normal**")
        st.caption(
            "Middle tercile. Represents a volatility state closer to the asset’s historical baseline."
        )
    with regime_cols[2]:
        st.markdown("**Stress**")
        st.caption(
            "Upper tercile of observed forward volatility in the training sample. "
            "Signals a more turbulent or fragile market environment."
        )

    st.markdown("### How to read this comparison")
    st.markdown(
        f"""
        <div class="insight-list">
          <p>The continuous readout is ordered first by <strong>QLIKE</strong>, because it is the project’s primary volatility-loss metric.</p>
          <p>For <strong>{selected_symbol}</strong>, the current QLIKE edge is
          <strong>{format_signed_percent(decision_row["relative_qlike_improvement_mean"])}</strong>, which is not strong enough to displace the benchmark.</p>
          <p>The confusion matrices and the deltas in <strong>Macro-F1</strong> and <strong>Balanced Accuracy</strong>
          indicate whether the model is not only forecasting magnitude, but also classifying the market state correctly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical detail", expanded=False):
        st.markdown(
            "The first table summarizes, for each historical segment and each metric, "
            "the benchmark and ML values side by side."
        )
        st.markdown("**Pivot by split / role / metric**")
        st.dataframe(
            pivot_df[
                [
                    "split_id",
                    "dataset_role",
                    "metric_name",
                    "benchmark_metric_value",
                    "ml_metric_value",
                    "relative_improvement_vs_benchmark",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            "The second table moves to the date-by-date panel and shows the observed value, "
            "each model forecast, and the observed/predicted regime."
        )
        st.markdown("**Recent panel rows**")
        st.dataframe(
            panel_df[
                [
                    "date",
                    "split_id",
                    "dataset_role",
                    "model_name",
                    "future_rv_5d",
                    "yhat_future_rv_5d",
                    "future_regime_5d",
                    "yhat_future_regime_5d",
                ]
            ].tail(16),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
