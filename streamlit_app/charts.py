from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK = "#102A43"
GREEN = "#1B6E5A"
COPPER = "#B35C2E"
GOLD = "#C89F3D"
SLATE = "#7B8794"
SOFT_RED = "#B54738"


def _build_rangebreaks(dates: pd.Series) -> list[dict]:
    """Removes weekends and missing business days from time axes."""
    normalized_dates = (
        pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates()
    )
    if normalized_dates.empty:
        return []

    business_days = pd.bdate_range(normalized_dates.min(), normalized_dates.max())
    missing_business_days = business_days.difference(normalized_dates.sort_values())

    rangebreaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if len(missing_business_days) > 0:
        rangebreaks.append(dict(values=list(missing_business_days)))

    return rangebreaks


def _apply_base_layout(fig: go.Figure, *, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=dict(family="Avenir Next, Segoe UI, sans-serif", color=INK),
        margin=dict(l=18, r=18, t=48, b=84),
        hoverlabel=dict(bgcolor="#fffaf2", font_color=INK),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(16,42,67,0.08)", zeroline=False)
    return fig


def build_universe_tradeoff_figure(asset_summary_df: pd.DataFrame) -> go.Figure:
    """Builds a narrative scatter for continuous vs discrete trade-offs."""
    df = asset_summary_df.copy().sort_values("symbol")
    max_abs_color = max(
        0.01,
        float(df["balanced_accuracy_delta"].abs().max()) if not df.empty else 0.01,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["relative_qlike_improvement_mean"],
            y=df["macro_f1_delta"],
            text=df["symbol"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=18,
                color=df["balanced_accuracy_delta"],
                cmin=-max_abs_color,
                cmax=max_abs_color,
                colorscale=[[0.0, SOFT_RED], [0.5, "#f3ede0"], [1.0, GREEN]],
                line=dict(width=1.4, color="#fdfbf5"),
                colorbar=dict(title="Delta Bal. Acc.", tickformat=".1%"),
            ),
            customdata=df[["balanced_accuracy_delta"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "QLIKE edge: %{x:.2%}<br>"
                "Macro-F1 delta: %{y:.2%}<br>"
                "Balanced Accuracy delta: %{customdata[0]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_dash="dot", line_color=SLATE)
    fig.add_hline(y=0, line_width=1.2, line_dash="dot", line_color=SLATE)
    fig.update_layout(
        title="Which engine read volatility and regimes more effectively",
        xaxis_title="Relative QLIKE improvement vs benchmark",
        yaxis_title="Macro-F1 delta vs benchmark",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    return _apply_base_layout(fig, height=430)


def build_candlestick_volume_figure(
    bars_df: pd.DataFrame,
    break_dates: pd.Series | None = None,
) -> go.Figure:
    """Builds the OHLC + volume view."""
    df = bars_df.copy().sort_values("date")
    colors = [
        GREEN if close >= open_ else COPPER
        for open_, close in zip(df["open"], df["close"])
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.74, 0.26],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=GREEN,
            decreasing_line_color=COPPER,
            showlegend=False,
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            marker_color=colors,
            opacity=0.55,
            showlegend=False,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    if break_dates is not None:
        valid_break_dates = pd.to_datetime(break_dates, errors="coerce").dropna()
        min_date = df["date"].min()
        max_date = df["date"].max()
        for break_date in valid_break_dates:
            if min_date <= break_date <= max_date:
                fig.add_vline(
                    x=break_date,
                    line_color=GOLD,
                    line_dash="dot",
                    line_width=1.3,
                )

    fig.update_layout(
        title="Price and volume over the same window used for out-of-sample forecast evaluation",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangebreaks=_build_rangebreaks(df["date"]))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _apply_base_layout(fig, height=560)


def build_forecast_figure(forecast_df: pd.DataFrame) -> go.Figure:
    """Builds the continuous comparison between observed, benchmark, and ML."""
    df = forecast_df.copy().sort_values("date")
    fig = go.Figure()

    role_df = df[["date", "dataset_role"]].drop_duplicates().sort_values("date")
    role_df["segment_id"] = (
        role_df["dataset_role"].ne(role_df["dataset_role"].shift()).cumsum()
    )
    role_colors = {
        "validation": "rgba(200, 159, 61, 0.10)",
        "test": "rgba(27, 110, 90, 0.08)",
    }
    role_labels = {"validation": "Validation", "test": "Test"}
    for _, segment_df in role_df.groupby("segment_id"):
        role = str(segment_df["dataset_role"].iloc[0])
        fig.add_vrect(
            x0=segment_df["date"].min(),
            x1=segment_df["date"].max(),
            line_width=0,
            fillcolor=role_colors.get(role, "rgba(123, 135, 148, 0.08)"),
            annotation_text=role_labels.get(role, role.title()),
            annotation_position="top left",
            annotation_font_color=SLATE,
        )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["future_rv_5d"],
            mode="lines",
            line=dict(color=INK, width=3),
            name="Observed volatility",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["benchmark_yhat_future_rv_5d"],
            mode="lines",
            line=dict(color=COPPER, width=2.4, dash="dash"),
            name="Benchmark forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ml_yhat_future_rv_5d"],
            mode="lines",
            line=dict(color=GREEN, width=2.4),
            name="ML forecast",
        )
    )
    fig.update_layout(
        title="Observed 5-day realized volatility versus both forecasts on the latest unseen window",
        yaxis_title="Forward 5-day realized volatility",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="left", x=0.0),
    )
    fig.update_xaxes(rangebreaks=_build_rangebreaks(df["date"]))
    return _apply_base_layout(fig, height=430)


def build_metric_relative_improvement_figure(summary_df: pd.DataFrame) -> go.Figure:
    """Builds the aggregated comparison by metric."""
    metric_order = ["qlike", "rmse", "mae", "macro_f1", "balanced_accuracy"]
    metric_labels = {
        "qlike": "QLIKE",
        "rmse": "RMSE",
        "mae": "MAE",
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Balanced accuracy",
    }

    df = summary_df.copy()
    df["metric_name"] = pd.Categorical(
        df["metric_name"], categories=metric_order, ordered=True
    )
    df = df.sort_values("metric_name")
    colors = [
        GREEN if value >= 0 else SOFT_RED for value in df["mean_relative_improvement"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["mean_relative_improvement"],
            y=[metric_labels.get(metric, metric) for metric in df["metric_name"]],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.1%}" for value in df["mean_relative_improvement"]],
            textposition="outside",
            hovertemplate="%{y}<br>Relative improvement: %{x:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_dash="dot", line_color=SLATE)
    fig.update_layout(
        title="Average advantage by metric",
        xaxis_title="Relative improvement",
        yaxis_title="",
    )
    fig.update_xaxes(tickformat=".0%")
    return _apply_base_layout(fig, height=430)


def build_role_metric_heatmap(role_summary_df: pd.DataFrame) -> go.Figure:
    """Builds a compact heatmap by dataset role and metric."""
    metric_order = ["qlike", "rmse", "mae", "macro_f1", "balanced_accuracy"]
    metric_labels = {
        "qlike": "QLIKE",
        "rmse": "RMSE",
        "mae": "MAE",
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Balanced accuracy",
    }
    role_order = ["validation", "test"]
    role_labels = {"validation": "Validation", "test": "Test"}

    pivot_df = role_summary_df.pivot(
        index="metric_name",
        columns="dataset_role",
        values="mean_relative_improvement",
    ).reindex(index=metric_order, columns=role_order)

    text = [
        ["" if pd.isna(value) else f"{value:.1%}" for value in row]
        for row in pivot_df.values
    ]
    z_values = pivot_df.fillna(0.0).values

    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=[role_labels.get(role, role.title()) for role in pivot_df.columns],
            y=[metric_labels.get(metric, metric) for metric in pivot_df.index],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Improvement", tickformat=".0%"),
            hovertemplate="%{y}<br>%{x}: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title="Where the gap appears: validation versus test")
    return _apply_base_layout(fig, height=430)


def build_confusion_matrix_figure(
    confusion_summary_df: pd.DataFrame,
    *,
    model_name: str,
    dataset_role: str,
) -> go.Figure:
    """Builds an aggregated confusion-matrix heatmap."""
    labels = ["calm", "normal", "stress"]
    filtered_df = confusion_summary_df.loc[
        (confusion_summary_df["model_name"] == model_name)
        & (confusion_summary_df["dataset_role"] == dataset_role)
    ].copy()

    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No confusion matrix available",
            showarrow=False,
            font=dict(color=SLATE),
        )
        return _apply_base_layout(fig, height=360)

    count_df = filtered_df.pivot(
        index="y_true", columns="y_pred", values="count"
    ).reindex(index=labels, columns=labels).fillna(0)
    share_df = filtered_df.pivot(
        index="y_true", columns="y_pred", values="row_share"
    ).reindex(index=labels, columns=labels).fillna(0.0)

    text = [
        [
            f"{int(count_df.iloc[row_idx, col_idx])}<br>{share_df.iloc[row_idx, col_idx]:.0%}"
            for col_idx in range(len(labels))
        ]
        for row_idx in range(len(labels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=share_df.values,
            x=[label.title() for label in labels],
            y=[label.title() for label in labels],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#fff6df"], [0.55, "#d8b35d"], [1.0, GREEN]],
            zmin=0,
            zmax=1,
            colorbar=dict(title="Share", tickformat=".0%"),
            hovertemplate="Actual %{y}<br>Predicted %{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=model_name.replace("_", " ").title(),
        xaxis_title="Predicted regime",
        yaxis_title="Observed regime",
    )
    return _apply_base_layout(fig, height=360)


def build_structural_break_figure(
    signal_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> go.Figure:
    """Builds the structural-break signal with break markers."""
    df = signal_df.copy().sort_values("date")
    fig = go.Figure()

    if "z_future_rv_5d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["z_future_rv_5d"],
                mode="lines",
                line=dict(color=GREEN, width=2.6),
                name="z_future_rv_5d",
            )
        )

    if "z_log_ret_1d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["z_log_ret_1d"],
                mode="lines",
                line=dict(color=COPPER, width=2.1),
                name="z_log_ret_1d",
            )
        )

    for break_date in pd.to_datetime(events_df["break_date"], errors="coerce").dropna():
        fig.add_vline(
            x=break_date,
            line_color=GOLD,
            line_dash="dot",
            line_width=1.35,
        )

    fig.update_layout(
        title="Signal used to detect structural change over the same window as the main forecast analysis",
        yaxis_title="Standardized signal",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="left", x=0.0),
    )
    fig.update_xaxes(rangebreaks=_build_rangebreaks(df["date"]))
    return _apply_base_layout(fig, height=470)
