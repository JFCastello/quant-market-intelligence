from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class FeatureRow(BaseModel): # Se define el modelo de datos, si no se pasa se asigna un None 
    instrument_id: str       # <Castello>
    date: date
    feature_version: str

    log_ret_1d: float | None = None
    log_ret_5d: float | None = None

    vol_5d: float | None = None
    vol_10d: float | None = None
    vol_20d: float | None = None
    vol_60d: float | None = None

    hl_range: float | None = None
    co_range: float | None = None

    atr_14: float | None = None

    mom_10d: float | None = None

    ma_5: float | None = None
    ma_20: float | None = None
    ma_60: float | None = None

    ma_ratio_5_20: float | None = None
    ma_ratio_20_60: float | None = None

    drawdown_20: float | None = None
    drawdown_60: float | None = None
    
    # Context Features -------------
    ctx_equity_proxy_log_ret_1d: float | None = None
    ctx_duration_proxy_log_ret_1d: float | None = None
    ctx_credit_proxy_log_ret_1d: float | None = None
    ctx_real_asset_proxy_log_ret_1d: float | None = None

    ctx_equity_proxy_log_ret_5d: float | None = None
    ctx_duration_proxy_log_ret_5d: float | None = None
    ctx_credit_proxy_log_ret_5d: float | None = None
    ctx_real_asset_proxy_log_ret_5d: float | None = None

    ctx_equity_proxy_vol_20d: float | None = None
    ctx_duration_proxy_vol_20d: float | None = None
    ctx_credit_proxy_vol_20d: float | None = None
    ctx_real_asset_proxy_vol_20d: float | None = None

    ctx_equity_duration_ret_5d_spread: float | None = None
    ctx_credit_duration_ret_5d_spread: float | None = None

    ctx_rel_vol_20d_vs_equity_proxy: float | None = None
    ctx_corr_20d_vs_equity_proxy: float | None = None