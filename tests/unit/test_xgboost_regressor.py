from __future__ import annotations

import pandas as pd

from quant_platform.models.xgboost_regressor import (
    build_ml_regressor_input_panel,
    build_xgboost_regressor_forecasts_by_split,
)


def make_ml_regressor_settings() -> dict:
    """Crea una configuración de prueba estándar para el regresor XGBoost."""
    return {
        "enabled": True,
        "model_name": "xgboost_regressor",
        "model_version": "v1",
        "feature_source": "features_context",
        "target_column": "future_rv_5d",
        "score_roles": ["validation", "test"],
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": 1,
        "early_stopping_rounds": 10,
        "persist_models": True,
        "persist_feature_columns": True,
        "allow_native_missing_values": True,
    }


def make_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye DataFrames de prueba: features, targets y splits para un instrumento."""
    dates = pd.bdate_range("2021-01-01", periods=80)

    # DataFrame de features con columnas de ejemplo
    features_df = pd.DataFrame(
        {
            "instrument_id": ["spy_us"] * len(dates),
            "date": dates,
            "feature_version": ["v1"] * len(dates),
            "log_ret_1d": [0.001 * ((i % 5) - 2) for i in range(len(dates))],
            "log_ret_5d": [0.002 * ((i % 7) - 3) for i in range(len(dates))],
            "vol_20d": [0.10 + 0.001 * i for i in range(len(dates))],
            "mom_10d": [100 + i for i in range(len(dates))],
            "ctx_equity_proxy_log_ret_1d": [0.0005 * ((i % 3) - 1) for i in range(len(dates))],
            "ctx_rel_vol_20d_vs_equity_proxy": [1.0 + 0.01 * (i % 4) for i in range(len(dates))],
        }
    )

    # DataFrame de targets continuos
    targets_df = pd.DataFrame(
        {
            "instrument_id": ["spy_us"] * len(dates),
            "date": dates,
            "target_version": ["v1"] * len(dates),
            "future_rv_5d": [0.08 + 0.0015 * i for i in range(len(dates))],
        }
    )

    # DataFrame de splits con un único pliegue
    split_df = pd.DataFrame(
        {
            "split_id": ["split_001"],
            "split_version": ["v1"],
            "instrument_id": ["spy_us"],
            "train_start": [dates[0]],
            "train_end": [dates[39]],
            "validation_start": [dates[40]],
            "validation_end": [dates[59]],
            "test_start": [dates[60]],
            "test_end": [dates[79]],
            "train_rows": [40],
            "validation_rows": [20],
            "test_rows": [20],
            "regime_thresholds_source": ["train_only"],
        }
    )

    return features_df, targets_df, split_df


def test_build_ml_regressor_input_panel_returns_expected_feature_columns() -> None:
    """
    Prueba que build_ml_regressor_input_panel:
    - Combine correctamente features y targets.
    - Identifique automáticamente las columnas de features (excluyendo metadatos).
    """
    settings = make_ml_regressor_settings()
    features_df, targets_df, _ = make_base_frames()

    panel_df, feature_columns = build_ml_regressor_input_panel(
        features_df=features_df,
        targets_df=targets_df,
        settings=settings,
    )

    assert not panel_df.empty
    assert "future_rv_5d" in panel_df.columns
    assert "instrument_id" in panel_df.columns
    assert "date" in panel_df.columns
    assert "feature_version" in panel_df.columns

    # Verificar que las columnas de features sean las esperadas
    assert set(feature_columns) == {
        "log_ret_1d",
        "log_ret_5d",
        "vol_20d",
        "mom_10d",
        "ctx_equity_proxy_log_ret_1d",
        "ctx_rel_vol_20d_vs_equity_proxy",
    }


def test_build_xgboost_regressor_forecasts_by_split_returns_scored_rows_only() -> None:
    """
    Prueba end-to-end de build_xgboost_regressor_forecasts_by_split:
    - Genera forecasts solo para validation y test (no train).
    - Los valores de forecast son positivos y no nulos.
    - Los metadatos (model_name, version, target_column) son constantes.
    """
    settings = make_ml_regressor_settings()
    features_df, targets_df, split_df = make_base_frames()

    forecast_df, split_models = build_xgboost_regressor_forecasts_by_split(
        features_df=features_df,
        targets_df=targets_df,
        split_df=split_df,
        settings=settings,
        symbol="SPY",
    )

    assert not forecast_df.empty
    assert len(split_models) == 1

    assert set(forecast_df["dataset_role"].unique()) == {"validation", "test"}
    assert set(forecast_df["split_id"].unique()) == {"split_001"}
    assert set(forecast_df["model_name"].unique()) == {"xgboost_regressor"}
    assert set(forecast_df["model_version"].unique()) == {"v1"}
    assert set(forecast_df["target_column"].unique()) == {"future_rv_5d"}

    assert forecast_df["yhat_future_rv_5d"].notna().all()
    assert forecast_df["future_rv_5d"].notna().all()
    assert (forecast_df["yhat_future_rv_5d"] > 0).all()

    # Verificar que los metadatos de entrenamiento sean constantes dentro del split
    assert forecast_df["feature_count"].nunique() == 1
    assert forecast_df["n_train"].nunique() == 1
    assert forecast_df["n_validation"].nunique() == 1
    assert forecast_df["n_score"].nunique() == 1


def test_build_xgboost_regressor_forecasts_by_split_returns_model_metadata() -> None:
    """
    Comprueba que la función devuelva los metadatos del modelo por split:
    - split_id correcto.
    - Número de filas de entrenamiento y validación > 0.
    - Lista de features no vacía.
    - Objeto del modelo presente.
    """
    settings = make_ml_regressor_settings()
    features_df, targets_df, split_df = make_base_frames()

    _, split_models = build_xgboost_regressor_forecasts_by_split(
        features_df=features_df,
        targets_df=targets_df,
        split_df=split_df,
        settings=settings,
        symbol="SPY",
    )

    split_model = split_models[0]

    assert split_model["split_id"] == "split_001"
    assert split_model["n_train"] > 0
    assert split_model["n_validation"] > 0
    assert isinstance(split_model["feature_columns"], list)
    assert len(split_model["feature_columns"]) > 0
    assert split_model["model"] is not None


def test_build_xgboost_regressor_forecasts_by_split_handles_missing_feature_values() -> None:
    """
    Verifica que el modelo pueda manejar valores ausentes en las features
    (permitido por allow_native_missing_values=True en la configuración).
    """
    settings = make_ml_regressor_settings()
    features_df, targets_df, split_df = make_base_frames()

    # Introducir valores nulos en algunas features
    features_df = features_df.copy()
    features_df.loc[5, "vol_20d"] = None
    features_df.loc[10, "ctx_equity_proxy_log_ret_1d"] = None

    forecast_df, split_models = build_xgboost_regressor_forecasts_by_split(
        features_df=features_df,
        targets_df=targets_df,
        split_df=split_df,
        settings=settings,
        symbol="SPY",
    )

    assert not forecast_df.empty
    assert len(split_models) == 1
    assert forecast_df["yhat_future_rv_5d"].notna().all()
    
def test_build_xgboost_regressor_forecasts_by_split_requires_non_empty_validation() -> None:
    """
    Prueba que la función lance una excepción cuando el conjunto de validación
    está vacío o no contiene datos útiles (targets nulos).
    """
    settings = make_ml_regressor_settings()
    features_df, targets_df, split_df = make_base_frames()

    split_row = split_df.iloc[0]
    validation_start = pd.to_datetime(split_row["validation_start"])
    validation_end = pd.to_datetime(split_row["validation_end"])

    targets_df = targets_df.copy()
    validation_mask = targets_df["date"].between(validation_start, validation_end, inclusive="both")
    targets_df.loc[validation_mask, "future_rv_5d"] = None

    try:
        build_xgboost_regressor_forecasts_by_split(
            features_df=features_df,
            targets_df=targets_df,
            split_df=split_df,
            settings=settings,
            symbol="SPY",
        )
    except ValueError as exc:
        assert "validation" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty/non-usable validation set")