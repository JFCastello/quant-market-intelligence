# Archivo: test_context_features.py
# Propósito: Tests unitarios para el pipeline de features de contexto.
#            Verifica generación de columnas, inferencia de roles,
#            construcción de paneles, enriquecimiento y políticas de auto‑referencia.

from __future__ import annotations

from datetime import datetime
import math

import pandas as pd
import pytest

from quant_platform.features import (
    adapt_features_to_contract,
    build_base_features,
    build_context_enriched_features,
    build_date_level_context_panel,
    get_enabled_context_feature_columns,
    infer_role_to_instrument_id,
)


# ============================================================================
# CONFIGURACIÓN DE PRUEBA (TEST_SETTINGS)
# ============================================================================
# Este diccionario replica la estructura de un archivo de settings real,
# pero con valores fijos para que los tests sean deterministas y aislados.
TEST_SETTINGS = {
    "features": {
        "feature_version": "v1",
        "price_column": "close",
        "annualization_factor": 252,
        "drop_warmup_rows": False,
        "include_intermediate_columns": False,
        # Configuración de rendimientos (para features base)
        "returns": {
            "enabled": True,
            "method": "log",
            "windows": [1, 5],
        },
        # Configuración de volatilidad (para features base)
        "volatility": {
            "enabled": True,
            "base_return_column": "log_ret_1d",
            "windows": [5, 10, 20, 60],
            "annualize": True,
        },
        # Rangos intradía
        "intraday_range": {
            "enabled": True,
            "include_hl_range": True,
            "include_co_range": True,
        },
        # ATR
        "atr": {
            "enabled": True,
            "window": 14,
            "normalize_by": "close",
        },
        # Momentum
        "momentum": {
            "enabled": True,
            "method": "log",
            "windows": [10],
        },
        # Medias móviles
        "moving_averages": {
            "enabled": True,
            "windows": [5, 20, 60],
            "ratios": [[5, 20], [20, 60]],
        },
        # Drawdowns
        "drawdown": {
            "enabled": True,
            "windows": [20, 60],
        },
    },
    # Configuración específica de features de contexto
    "context_features": {
        "enabled": True,
        # Mapeo de roles a símbolos (se usarán para inferir instrument_ids reales)
        "role_map": {
            "equity_proxy": "SPY",
            "duration_proxy": "TLT",
            "credit_proxy": "HYG",
            "real_asset_proxy": "GLD",
        },
        "source_layer": "features_v1",
        # Rendimientos directos para cada rol (log returns con ventanas 1 y 5 días)
        "direct_returns": {
            "enabled": True,
            "method": "log",
            "windows": [1, 5],
        },
        # Volatilidad directa para cada rol (ventana fija de 20 días)
        "direct_volatility": {
            "enabled": True,
            "windows": [20],
        },
        # Spreads (diferencias) entre pares de roles
        "spreads": {
            "enabled": True,
            "definitions": [
                {
                    "name": "equity_duration_ret_5d_spread",
                    "left_role": "equity_proxy",
                    "right_role": "duration_proxy",
                    "source_feature": "log_ret_5d",
                },
                {
                    "name": "credit_duration_ret_5d_spread",
                    "left_role": "credit_proxy",
                    "right_role": "duration_proxy",
                    "source_feature": "log_ret_5d",
                },
            ],
        },
        # Volatilidad relativa (target / referencia)
        "relative_volatility": {
            "enabled": True,
            "reference_role": "equity_proxy",
            "source_feature": "vol_20d",
            "output_name": "rel_vol_20d_vs_equity_proxy",
        },
        # Correlación móvil con política para cuando target == referencia
        "rolling_correlation": {
            "enabled": True,
            "reference_role": "equity_proxy",
            "source_feature": "log_ret_1d",
            "window": 20,
            "output_name": "corr_20d_vs_equity_proxy",
            "self_reference_policy": "nan",   # Para equity_proxy, la correlación será NaN
        },
        "merge_keys": ["date"],
        "naming": {
            "prefix": "ctx",   # Prefijo para todas las columnas de contexto
        },
    },
}


# ============================================================================
# FUNCIÓN AUXILIAR: GENERAR DATOS NORMALIZADOS PARA UN INSTRUMENTO
# ============================================================================
def make_sample_normalized_df(
    instrument_id: str,
    start_price: float,
    phase: float,
    n_rows: int = 90,
) -> pd.DataFrame:
    """
    Crea un DataFrame sintético con estructura OHLC diaria para un instrumento.
    El precio de cierre sigue una tendencia lineal + dos componentes sinusoidales
    (para dar variabilidad realista). Los demás precios se derivan.
    """
    # Fechas en días hábiles (business days)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")

    # Precio de cierre: tendencia + ondas
    close = pd.Series(
        [
            start_price
            + 0.25 * i
            + 1.2 * math.sin(i / 5 + phase)
            + 0.4 * math.cos(i / 11 + phase)
            for i in range(n_rows)
        ],
        dtype=float,
    )

    # OHLC típico: apertura ligeramente inferior, high y low simétricos
    open_ = close - 0.15
    high = close + 0.55
    low = close - 0.65
    volume = pd.Series([1_000_000 + i * 1000 for i in range(n_rows)], dtype=float)

    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "provider": "twelve_data",
            "ingested_at": pd.Timestamp(datetime(2026, 4, 8, 12, 0, 0), tz="UTC"),
        }
    )


# ============================================================================
# FUNCIÓN AUXILIAR: CONSTRUIR EL UNIVERSO COMPLETO DE FEATURES BASE
# ============================================================================
def make_base_feature_universe_df() -> pd.DataFrame:
    """
    Genera un DataFrame con las features base (rendimientos, volatilidad, etc.)
    para los cuatro instrumentos que actúan como roles: SPY, TLT, HYG, GLD.
    """
    # Datos normalizados para cada instrumento (con diferentes precios iniciales y fases)
    normalized_dfs = [
        make_sample_normalized_df("spy_us", start_price=100.0, phase=0.1),
        make_sample_normalized_df("tlt_us", start_price=130.0, phase=0.5),
        make_sample_normalized_df("hyg_us", start_price=85.0, phase=1.0),
        make_sample_normalized_df("gld_us", start_price=120.0, phase=1.5),
    ]

    # Para cada uno, construir features base y adaptarlas al contrato
    feature_dfs = []
    for normalized_df in normalized_dfs:
        feature_df = build_base_features(normalized_df, TEST_SETTINGS)
        feature_df = adapt_features_to_contract(feature_df, TEST_SETTINGS)
        feature_dfs.append(feature_df)

    # Concatenar todos los instrumentos en un solo DataFrame (universo)
    return pd.concat(feature_dfs, ignore_index=True)


# ============================================================================
# TEST 1: Verificar que las columnas de contexto generadas coinciden con el contrato
# ============================================================================
def test_get_enabled_context_feature_columns_matches_expected_contract() -> None:
    """
    Comprueba que get_enabled_context_feature_columns(settings) devuelve
    exactamente la lista de columnas esperada según la configuración de prueba.
    """
    columns = get_enabled_context_feature_columns(TEST_SETTINGS)

    expected = [
    "ctx_equity_proxy_log_ret_1d",
    "ctx_equity_proxy_log_ret_5d",
    "ctx_duration_proxy_log_ret_1d",
    "ctx_duration_proxy_log_ret_5d",
    "ctx_credit_proxy_log_ret_1d",
    "ctx_credit_proxy_log_ret_5d",
    "ctx_real_asset_proxy_log_ret_1d",
    "ctx_real_asset_proxy_log_ret_5d",
    "ctx_equity_proxy_vol_20d",
    "ctx_duration_proxy_vol_20d",
    "ctx_credit_proxy_vol_20d",
    "ctx_real_asset_proxy_vol_20d",
    "ctx_equity_duration_ret_5d_spread",
    "ctx_credit_duration_ret_5d_spread",
    "ctx_rel_vol_20d_vs_equity_proxy",
    "ctx_corr_20d_vs_equity_proxy",
]

    assert columns == expected


# ============================================================================
# TEST 2: Inferencia de rol a instrument_id real
# ============================================================================
def test_infer_role_to_instrument_id_returns_expected_mapping() -> None:
    """
    Verifica que infer_role_to_instrument_id mapea correctamente los símbolos
    de la configuración (SPY, TLT, HYG, GLD) a los instrument_ids reales
    que existen en el DataFrame ("spy_us", "tlt_us", "hyg_us", "gld_us").
    """
    universe_df = make_base_feature_universe_df()
    role_map = infer_role_to_instrument_id(universe_df, TEST_SETTINGS)

    assert role_map == {
        "equity_proxy": "spy_us",
        "duration_proxy": "tlt_us",
        "credit_proxy": "hyg_us",
        "real_asset_proxy": "gld_us",
    }


# ============================================================================
# TEST 3: Construcción del panel de contexto (solo fechas, sin target)
# ============================================================================
def test_build_date_level_context_panel_creates_expected_columns_and_spread() -> None:
    """
    Construye el panel de contexto a nivel de fecha y verifica:
    - Las columnas son las esperadas (incluye solo las de contexto, no las del target).
    - Los spreads se calculan correctamente como diferencia entre dos columnas.
    """
    universe_df = make_base_feature_universe_df()
    panel = build_date_level_context_panel(
        universe_features_df=universe_df,
        settings=TEST_SETTINGS,
    )

    # Conjunto esperado de columnas (date + todas las de contexto excepto rel_vol y corr,
    # porque esas se añaden después en el enriquecimiento completo)
    expected_panel_cols = {
        "date",
        "ctx_equity_proxy_log_ret_1d",
        "ctx_equity_proxy_log_ret_5d",
        "ctx_equity_proxy_vol_20d",
        "ctx_duration_proxy_log_ret_1d",
        "ctx_duration_proxy_log_ret_5d",
        "ctx_duration_proxy_vol_20d",
        "ctx_credit_proxy_log_ret_1d",
        "ctx_credit_proxy_log_ret_5d",
        "ctx_credit_proxy_vol_20d",
        "ctx_real_asset_proxy_log_ret_1d",
        "ctx_real_asset_proxy_log_ret_5d",
        "ctx_real_asset_proxy_vol_20d",
        "ctx_equity_duration_ret_5d_spread",
        "ctx_credit_duration_ret_5d_spread",
    }

    assert set(panel.columns) == expected_panel_cols

    # Toma filas donde las columnas origen y el spread no son nulos
    valid_rows = panel[
        panel["ctx_equity_proxy_log_ret_5d"].notna()
        & panel["ctx_duration_proxy_log_ret_5d"].notna()
        & panel["ctx_equity_duration_ret_5d_spread"].notna()
    ]
    assert not valid_rows.empty

    # Comprueba la fórmula spread = left - right
    row = valid_rows.iloc[0]
    manual_spread = (
        row["ctx_equity_proxy_log_ret_5d"] - row["ctx_duration_proxy_log_ret_5d"]
    )
    assert row["ctx_equity_duration_ret_5d_spread"] == pytest.approx(manual_spread)


# ============================================================================
# TEST 4: Enriquecimiento completo añade columnas de contexto y preserva filas
# ============================================================================
def test_build_context_enriched_features_adds_context_columns_and_preserves_rows() -> None:
    """
    Ejecuta build_context_enriched_features sobre el universo de features base
    y comprueba que:
    - Todas las columnas de contexto están presentes (incluyendo rel_vol y corr).
    - El número de filas no cambia (solo se añaden columnas).
    """
    universe_df = make_base_feature_universe_df()
    enriched_df = build_context_enriched_features(
        universe_features_df=universe_df,
        settings=TEST_SETTINGS,
    )

    ctx_cols = get_enabled_context_feature_columns(TEST_SETTINGS)
    for col in ctx_cols:
        assert col in enriched_df.columns

    assert len(enriched_df) == len(universe_df)


# ============================================================================
# TEST 5: Política de auto-referencia para la correlación
# ============================================================================
def test_self_reference_policy_sets_spy_correlation_to_nan() -> None:
    """
    Verifica que cuando el instrumento target es el mismo que el rol de referencia
    (equity_proxy = SPY), la columna de correlación sea completamente NaN,
    según la política self_reference_policy = "nan".
    """
    universe_df = make_base_feature_universe_df()
    enriched_df = build_context_enriched_features(
        universe_features_df=universe_df,
        settings=TEST_SETTINGS,
    )

    spy_df = enriched_df.loc[enriched_df["instrument_id"] == "spy_us"].copy()
    # Todas las filas de spy_us deben tener NaN en la columna de correlación
    assert spy_df["ctx_corr_20d_vs_equity_proxy"].isna().all()