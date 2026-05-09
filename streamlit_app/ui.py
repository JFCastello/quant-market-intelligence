from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_platform.services import load_settings

APP_STYLES = """
<style>
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(200, 159, 61, 0.14), transparent 28%),
      radial-gradient(circle at top right, rgba(27, 110, 90, 0.10), transparent 24%),
      linear-gradient(180deg, #f5f1e8 0%, #fbfaf7 45%, #f3ede0 100%);
    color: #102a43;
    font-family: "Avenir Next", "Segoe UI", sans-serif;
  }

  .block-container {
    max-width: 1220px;
    padding-top: 1.8rem;
    padding-bottom: 2.2rem;
  }

  h1, h2, h3 {
    color: #102a43;
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    letter-spacing: 0.01em;
  }

  div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #e4dac9;
    border-radius: 18px;
    padding: 0.95rem 1rem;
    box-shadow: 0 12px 32px rgba(16, 42, 67, 0.06);
  }

  div[data-testid="stMetricLabel"] {
    color: #7b8794;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .hero-panel {
    padding: 1.35rem 1.45rem;
    border-radius: 24px;
    border: 1px solid #e4dac9;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(240, 231, 212, 0.86) 100%);
    box-shadow: 0 18px 42px rgba(16, 42, 67, 0.08);
    margin-bottom: 1rem;
  }

  .hero-eyebrow {
    color: #b35c2e;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }

  .hero-title {
    color: #102a43;
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: 2.35rem;
    line-height: 1.04;
    margin: 0.4rem 0 0.65rem;
  }

  .hero-body {
    color: #52606d;
    font-size: 1rem;
    line-height: 1.6;
    max-width: 54rem;
  }

  .story-card {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #e4dac9;
    border-radius: 20px;
    min-height: 148px;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 10px 26px rgba(16, 42, 67, 0.05);
  }

  .story-card-label {
    color: #9b6b2d;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  .story-card-value {
    color: #102a43;
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: 1.35rem;
    line-height: 1.15;
    margin-bottom: 0.5rem;
  }

  .story-card-caption {
    color: #61707f;
    font-size: 0.94rem;
    line-height: 1.45;
  }

  .decision-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .decision-benchmark {
    background: rgba(179, 92, 46, 0.12);
    color: #9a3f1b;
  }

  .decision-ml {
    background: rgba(27, 110, 90, 0.12);
    color: #145a49;
  }

  .section-caption {
    color: #61707f;
    font-size: 0.95rem;
    margin-top: -0.35rem;
    margin-bottom: 0.75rem;
  }

  .insight-list {
    color: #425466;
    line-height: 1.65;
  }
</style>
"""


def load_dashboard_config() -> dict:
    """Returns the local dashboard configuration."""
    settings = load_settings()
    dashboard_cfg = settings.get("dashboard_local")

    if not isinstance(dashboard_cfg, dict):
        raise ValueError(
            "dashboard_local settings are missing or invalid in configs/base.yaml"
        )

    return dashboard_cfg


def configure_page(page_title: str) -> dict:
    """Configures shared page metadata and styling."""
    dashboard_cfg = load_dashboard_config()
    app_cfg = dashboard_cfg["app"]

    st.set_page_config(
        page_title=f"{page_title} | {app_cfg['title']}",
        layout=app_cfg["layout"],
        initial_sidebar_state=app_cfg["sidebar_state"],
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    return dashboard_cfg


def render_hero(eyebrow: str, title: str, body: str) -> None:
    """Renders the main narrative header for each page."""
    st.markdown(
        f"""
        <section class="hero-panel">
          <div class="hero-eyebrow">{eyebrow}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-body">{body}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_story_card(label: str, value: str, caption: str) -> None:
    """Renders a short contextual card."""
    st.markdown(
        f"""
        <section class="story-card">
          <div class="story-card-label">{label}</div>
          <div class="story-card-value">{value}</div>
          <div class="story-card-caption">{caption}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_symbol_sidebar(
    dashboard_cfg: dict,
    symbols: list[str],
    *,
    label: str = "Asset",
) -> str:
    """Renders the sidebar asset selector with the configured default."""
    default_symbol = str(dashboard_cfg["defaults"]["symbol"]).upper()
    default_index = symbols.index(default_symbol) if default_symbol in symbols else 0

    return st.sidebar.selectbox(
        label,
        options=symbols,
        index=default_index,
    )


def decision_badge_html(decision: str) -> str:
    """Returns a styled HTML badge for the final decision."""
    if decision == "promote_ml":
        label = "ML"
        klass = "decision-chip decision-ml"
    else:
        label = "Benchmark"
        klass = "decision-chip decision-benchmark"

    return f'<span class="{klass}">{label}</span>'


def format_date(value) -> str:
    """Formats timestamps for the UI."""
    if value is None or pd.isna(value):
        return "N/A"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def format_number(value, digits: int = 3) -> str:
    """Formats decimal values for the UI."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def format_signed_number(value, digits: int = 3) -> str:
    """Formats signed numeric values."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):+,.{digits}f}"


def format_percent(value, digits: int = 1) -> str:
    """Formats ratios as percentages."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def format_signed_percent(value, digits: int = 1) -> str:
    """Formats signed percentages."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:+.{digits}f}%"


def format_volume(value) -> str:
    """Formats large traded volumes compactly."""
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"
