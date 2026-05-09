from __future__ import annotations          

from datetime import datetime              
import math                                

import pandas as pd                         
import pytest                               # Framework de testing, provee approx() para comparar floats

# Importaciones de las funciones del pipeline de features que vamos a testear
from quant_platform.features import (
    adapt_features_to_contract,             # Filtra y valida columnas según contrato
    build_base_features,                    # Construye todas las features desde datos normalizados
    get_enabled_feature_columns,            # Devuelve lista de columnas esperadas según settings
    validate_feature_output,                # Validación exhaustiva del DataFrame final
)

# ============================================================================
# CONFIGURACIÓN DE PRUEBA (TEST_SETTINGS)
# ============================================================================
# Este diccionario replica la estructura de un archivo de settings real,
# pero con valores fijos para que los tests sean deterministas.
TEST_SETTINGS = {
    "features": {
        "feature_version": "v1",                    # Versión de las features
        "price_column": "close",                    # Columna de precios principal
        "annualization_factor": 252,                # Días hábiles por año (para anualizar volatilidad)
        "drop_warmup_rows": False,                  # No eliminar filas iniciales en tests
        "include_intermediate_columns": False,      # No incluir columnas intermedias (solo las finales)
        "returns": {
            "enabled": True,
            "method": "log",
            "windows": [1, 5],                      # Ventanas de rendimiento: 1d y 5d
        },
        "volatility": {
            "enabled": True,
            "base_return_column": "log_ret_1d",     # Se basa en rendimiento diario
            "windows": [5, 10, 20, 60],             # Ventanas de volatilidad
            "annualize": True,                      # Convertir a volatilidad anualizada
        },
        "intraday_range": {
            "enabled": True,
            "include_hl_range": True,               # Incluir (high-low)/close
            "include_co_range": True,               # Incluir (close-open)/open
        },
        "atr": {
            "enabled": True,
            "window": 14,                           # Ventana estándar para ATR
            "normalize_by": "close",                # Normalizar ATR dividiendo por close
        },
        "momentum": {
            "enabled": True,
            "method": "log",                        # Mismo cálculo que log returns
            "windows": [10],                        # Solo una ventana para test
        },
        "moving_averages": {
            "enabled": True,
            "windows": [5, 20, 60],                 # Medias móviles simples
            "ratios": [[5, 20], [20, 60]],          # Ratios entre medias (5/20 y 20/60)
        },
        "drawdown": {
            "enabled": True,
            "windows": [20, 60],                    # Ventanas para drawdown máximo rodante
        },
        "context_market": {
            "enabled": False,                       # Feature deshabilitada (no se testea)
        },
    }
}

# ============================================================================
# FUNCIÓN AUXILIAR: GENERA DATOS NORMALIZADOS DE PRUEBA
# ============================================================================
def make_sample_normalized_df(n_rows: int = 80) -> pd.DataFrame:
    """
    Crea un DataFrame sintético con estructura similar a los datos normalizados
    que entrarían al pipeline: OHLC diario, volume, provider, ingested_at, etc.
    """
    # Genera fechas consecutivas en días hábiles (freq="B" = business days)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")

    # Precios: serie creciente linealmente (100, 100.5, 101, ...)
    close = pd.Series([100 + i * 0.5 for i in range(n_rows)], dtype=float)
    open_ = close - 0.2           # Apertura ligeramente inferior al cierre
    high = close + 0.8            # Máximo por encima del cierre
    low = close - 0.9             # Mínimo por debajo del cierre
    volume = pd.Series([1_000_000 + i * 1000 for i in range(n_rows)], dtype=float)

    # Construye el DataFrame con columnas requeridas por el pipeline
    df = pd.DataFrame(
        {
            "instrument_id": "spy_us",                         # Mismo símbolo para toda la serie
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "provider": "twelve_data",                         # Proveedor fijo
            "ingested_at": pd.Timestamp(datetime(2026, 4, 7, 12, 0, 0), tz="UTC"),
        }
    )
    return df

# ============================================================================
# TEST 1: Verificar que la lista de columnas generada coincide con el contrato
# ============================================================================
def test_get_enabled_feature_columns_matches_expected_contract() -> None:
    """
    Comprueba que get_enabled_feature_columns(settings) devuelve exactamente
    las columnas que esperamos según la configuración de prueba.
    Esto asegura que el contrato entre generación y consumo de features no cambie.
    """
    columns = get_enabled_feature_columns(TEST_SETTINGS)

    # Lista esperada según TEST_SETTINGS (todas las features habilitadas)
    expected = [
        "instrument_id",
        "date",
        "feature_version",
        "log_ret_1d",
        "log_ret_5d",
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "vol_60d",
        "hl_range",
        "co_range",
        "atr_14",
        "mom_10d",
        "ma_5",
        "ma_20",
        "ma_60",
        "ma_ratio_5_20",
        "ma_ratio_20_60",
        "drawdown_20",
        "drawdown_60",
    ]

    assert columns == expected   # Comparación exacta de listas

# ============================================================================
# TEST 2: Construcción completa de features produce columnas esperadas y datos coherentes
# ============================================================================
def test_build_base_features_creates_expected_columns() -> None:
    """
    Ejecuta build_base_features + adapt_features_to_contract y verifica:
    - Las columnas resultantes son las esperadas (contrato).
    - El número de filas se preserva (sin agregar ni eliminar).
    - La columna feature_version tiene el valor correcto para todas las filas.
    """
    df = make_sample_normalized_df()                      # Datos de entrada sintéticos
    feature_df = build_base_features(df, TEST_SETTINGS)   # Pipeline completo
    contracted_df = adapt_features_to_contract(feature_df, TEST_SETTINGS)  # Filtrado

    expected_columns = get_enabled_feature_columns(TEST_SETTINGS)

    assert list(contracted_df.columns) == expected_columns
    assert len(contracted_df) == len(df)                  # Misma cantidad de registros
    assert contracted_df["feature_version"].eq("v1").all() # Versión consistente

# ============================================================================
# TEST 3: Precisión numérica del cálculo de log_ret_1d
# ============================================================================
def test_log_ret_1d_matches_manual_calculation() -> None:
    """
    Toma dos filas consecutivas de los datos originales, calcula manualmente
    el rendimiento logarítmico (ln(close_t / close_{t-1})) y lo compara con
    el valor generado por el pipeline en la columna 'log_ret_1d'.
    """
    df = make_sample_normalized_df()
    feature_df = build_base_features(df, TEST_SETTINGS)
    contracted_df = adapt_features_to_contract(feature_df, TEST_SETTINGS)

    # Cálculo manual usando la fila 1 (índice 1) y la anterior (índice 0)
    manual_value = math.log(df.loc[1, "close"] / df.loc[0, "close"])
    model_value = contracted_df.loc[1, "log_ret_1d"]

    # pytest.approx permite tolerancia por errores de punto flotante
    assert model_value == pytest.approx(manual_value)

# ============================================================================
# TEST 4: Validación de output no lanza excepción con datos correctos
# ============================================================================
def test_validate_feature_output_passes_on_valid_dataset() -> None:
    """
    Verifica que validate_feature_output no lance ninguna excepción cuando se
    le entrega un DataFrame de features construido correctamente.
    """
    df = make_sample_normalized_df()
    feature_df = build_base_features(df, TEST_SETTINGS)
    contracted_df = adapt_features_to_contract(feature_df, TEST_SETTINGS)

    # Si hay algún problema, esta función lanza ValueError; si no, pasa silenciosamente.
    validate_feature_output(contracted_df, TEST_SETTINGS)

# ============================================================================
# TEST 5: Ordenamiento temporal se preserva aunque la entrada esté desordenada
# ============================================================================
def test_build_base_features_preserves_sorted_dates() -> None:
    """
    Construye un DataFrame de entrada con las filas en orden aleatorio (sample).
    Luego de pasar por build_base_features y adapt_features_to_contract,
    el DataFrame resultante debe estar ordenado por fecha (dentro de cada instrumento).
    Este test verifica que el pipeline reordena internamente (gracias a _sort_input).
    """
    df = make_sample_normalized_df()
    # Mezcla todas las filas aleatoriamente (frac=1.0) con semilla fija para reproducibilidad
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    feature_df = build_base_features(shuffled_df, TEST_SETTINGS)
    contracted_df = adapt_features_to_contract(feature_df, TEST_SETTINGS)

    # Orden esperado: fechas de menor a mayor
    sorted_dates = contracted_df["date"].sort_values().reset_index(drop=True)
    current_dates = contracted_df["date"].reset_index(drop=True)

    assert current_dates.equals(sorted_dates)