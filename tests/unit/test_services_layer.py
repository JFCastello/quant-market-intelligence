from __future__ import annotations

from quant_platform.services import (
    build_symbol_model_comparison_summary,
    build_symbol_market_forecast_snapshot,
    build_symbol_overview_snapshot,
    build_symbol_structural_break_summary,
    get_executive_summary_bundle,
    get_symbol_market_forecast_bundle,
    get_symbol_model_comparison_dashboard_bundle,
    get_symbol_model_comparison_bundle,
    get_symbol_overview_bundle,
    get_symbol_structural_changes_bundle,
    list_available_symbols,
    load_decision_summary,
)


def test_list_available_symbols_returns_expected_universe() -> None:
    """
    Verifica que la capa de servicios descubra exactamente el universo de
    símbolos esperado para el proyecto.

    Qué comprueba:
    - Que `list_available_symbols()` funcione.
    - Que el conjunto retornado coincida exactamente con el universo esperado.
    - Que el orden también sea el esperado.

    Por qué importa:
    Este test valida un contrato base de discovery. Si la lista cambia,
    puede significar una de estas cosas:
    - faltan artefactos o carpetas,
    - apareció un símbolo inesperado,
    - cambió el orden/canonización de salida.

    Observación:
    Como aquí se compara contra una lista exacta, este test es deliberadamente
    estricto. No solo verifica contenido, también verifica orden y tamaño.
    """
    symbols = list_available_symbols()

    # Se espera exactamente este universo de símbolos.
    assert symbols == ["GLD", "HYG", "SPY", "TLT"]


def test_load_decision_summary_returns_expected_shape() -> None:
    """
    Verifica que el artefacto global de decision summary cargue con la forma
    mínima esperada.

    Qué comprueba:
    - Que el DataFrame tenga exactamente 4 filas.
    - Que los símbolos presentes sean los cuatro esperados.
    - Que exista la columna `decision`.

    Por qué importa:
    Este test asegura que el artefacto global de decisiones:
    - existe,
    - tiene cobertura para todo el universo esperado,
    - conserva columnas clave que otras capas consumen después.

    Alcance:
    No valida todos los valores de todas las columnas; valida únicamente
    la forma mínima que hace al artefacto usable por la services layer.
    """
    decision_summary_df = load_decision_summary()

    # Debe haber una fila por símbolo del universo esperado.
    assert len(decision_summary_df) == 4

    # Los símbolos presentes deben coincidir exactamente con el universo esperado.
    assert sorted(decision_summary_df["symbol"].unique().tolist()) == [
        "GLD",
        "HYG",
        "SPY",
        "TLT",
    ]

    # La columna `decision` es esencial porque resume la decisión final por símbolo.
    assert "decision" in decision_summary_df.columns


def test_build_symbol_overview_snapshot_returns_consistent_snapshot() -> None:
    """
    Verifica que el snapshot de overview para SPY sea internamente consistente
    y contenga algunos valores esperados conocidos.

    Qué comprueba:
    - Que el snapshot corresponda realmente a SPY.
    - Que el instrument_id sea el esperado.
    - Que la decisión final esperada esté presente.
    - Que haya al menos un structural break reciente reportado.
    - Que la fecha más reciente de features no sea anterior a la de targets.

    Por qué importa:
    Este test valida una función de agregación de alto nivel.
    El snapshot no solo junta datos: también representa una vista final
    “lista para mostrar”. Por eso conviene verificar algunos valores concretos
    y algunas relaciones lógicas entre campos.
    """
    snapshot = build_symbol_overview_snapshot("SPY")

    # El snapshot debe identificarse correctamente con SPY.
    assert snapshot["symbol"] == "SPY"

    # El instrument_id esperado para SPY en este proyecto es `spy_us`.
    assert snapshot["instrument_id"] == "spy_us"

    # Se valida una decisión esperada conocida del artefacto de decision summary.
    assert snapshot["decision"] == "do_not_promote_ml"

    # El resumen debe reportar al menos un quiebre reciente.
    assert snapshot["recent_break_count"] >= 1

    # En este pipeline se espera que la última fecha de features sea igual
    # o posterior a la última fecha válida del target.
    assert snapshot["latest_feature_date"] >= snapshot["latest_target_date"]


def test_get_executive_summary_bundle_returns_semantically_correct_calibration_counts() -> (
    None
):
    """
    Verifica que el bundle ejecutivo corrija la semántica del conteo de calibración.

    Qué comprueba:
    - Que el bundle exponga las llaves esperadas.
    - Que no existan calibraciones pendientes en el artefacto actual.
    - Que las 4 filas actuales caigan en el bucket `not_evaluable_yet`.
    """
    bundle = get_executive_summary_bundle()

    assert sorted(bundle.keys()) == [
        "asset_summary_df",
        "decision_summary_df",
        "kpis",
    ]
    assert bundle["kpis"]["calibration_pending_count"] == 0
    assert bundle["kpis"]["calibration_not_evaluable_count"] == 4
    assert len(bundle["asset_summary_df"]) == 4


def test_get_symbol_overview_bundle_returns_expected_keys_and_rows() -> None:
    """
    Verifica que el bundle de overview para SPY tenga la estructura esperada
    y un tamaño conocido en su serie temporal.

    Qué comprueba:
    - Que el bundle tenga exactamente las llaves esperadas.
    - Que `timeseries_df` tenga el número de filas esperado.
    - Que el snapshot interno corresponda a SPY.

    Por qué importa:
    Este test asegura que la función bundle:
    - expone el contrato correcto hacia capas superiores,
    - contiene tanto resumen como datos tabulares de respaldo,
    - y que el artefacto temporal fusionado tiene el tamaño esperado.

    Nota:
    Este tipo de test es útil para detectar cambios silenciosos en la estructura
    del bundle o en la cobertura temporal de los datos.
    """
    bundle = get_symbol_overview_bundle("SPY")

    # Se verifica el contrato exacto de llaves del bundle.
    assert sorted(bundle.keys()) == [
        "decision_summary_row",
        "recent_break_events_df",
        "snapshot",
        "timeseries_df",
    ]

    # Se espera un número exacto de filas en la serie temporal fusionada.
    assert len(bundle["timeseries_df"]) == 2075

    # El snapshot incluido dentro del bundle debe corresponder a SPY.
    assert bundle["snapshot"]["symbol"] == "SPY"


def test_build_symbol_market_forecast_snapshot_returns_recent_market_and_forecast_state() -> (
    None
):
    """
    Verifica que el snapshot de Market & Forecast consolide precio, target y forecasts.
    """
    snapshot = build_symbol_market_forecast_snapshot("SPY")

    assert snapshot["symbol"] == "SPY"
    assert snapshot["latest_split_id"] == "split_007"
    assert snapshot["latest_dataset_role"] == "test"
    assert snapshot["latest_bar_date"] >= snapshot["latest_forecast_date"]


def test_get_symbol_market_forecast_bundle_returns_expected_focus_split() -> None:
    """
    Verifica que el bundle de Market & Forecast expose la vista más reciente del split.
    """
    bundle = get_symbol_market_forecast_bundle("SPY")

    assert sorted(bundle.keys()) == [
        "bars_df",
        "decision_summary_row",
        "focus_forecast_df",
        "focus_split_id",
        "forecast_df",
        "recent_break_events_df",
        "snapshot",
    ]
    assert bundle["focus_split_id"] == "split_007"
    assert set(bundle["focus_forecast_df"]["dataset_role"].unique().tolist()) == {
        "test",
        "validation",
    }


def test_build_symbol_model_comparison_summary_returns_expected_metrics() -> None:
    """
    Verifica que el resumen agregado de comparación de modelos para SPY
    contenga exactamente las métricas esperadas.

    Qué comprueba:
    - Que el summary tenga 5 filas.
    - Que las métricas únicas sean exactamente las esperadas.

    Por qué importa:
    Este test valida la capa agregada de model comparison.
    Si cambia el número de filas o el conjunto de métricas, puede significar:
    - que cambió el pipeline de evaluación,
    - que faltan métricas,
    - que se alteró la forma del resumen.

    Observación:
    Aquí se asume implícitamente que hay una fila agregada por métrica.
    """
    summary_df = build_symbol_model_comparison_summary("SPY")

    # Se espera una fila por cada métrica agregada relevante.
    assert len(summary_df) == 5

    # El conjunto exacto de métricas esperadas en el resumen.
    assert sorted(summary_df["metric_name"].unique().tolist()) == [
        "balanced_accuracy",
        "macro_f1",
        "mae",
        "qlike",
        "rmse",
    ]


def test_get_symbol_model_comparison_bundle_returns_expected_keys() -> None:
    """
    Verifica que el bundle de model comparison para SPY tenga la estructura
    esperada y tamaños conocidos en dos de sus DataFrames.

    Qué comprueba:
    - Que el bundle tenga exactamente las cuatro llaves previstas.
    - Que `metrics_df` tenga el número esperado de filas.
    - Que `panel_df` tenga el número esperado de filas.

    Por qué importa:
    Este test comprueba tanto el contrato estructural del bundle como el
    volumen esperado de datos en dos artefactos clave:
    - la tabla larga de métricas,
    - el panel detallado.

    Utilidad:
    Ayuda a detectar si cambió silenciosamente el número de splits, fechas,
    métricas o modelos considerados.
    """
    bundle = get_symbol_model_comparison_bundle("SPY")

    # Verifica las llaves exactas que promete la función bundle.
    assert sorted(bundle.keys()) == [
        "metrics_df",
        "panel_df",
        "pivot_df",
        "summary_df",
    ]

    # Tamaño esperado del artefacto largo de métricas.
    assert len(bundle["metrics_df"]) == 140

    # Tamaño esperado del panel detallado.
    assert len(bundle["panel_df"]) == 3510


def test_get_symbol_model_comparison_dashboard_bundle_returns_confusion_views() -> None:
    """
    Verifica que el bundle de comparación para UI incluya matrices de confusión agregadas.
    """
    bundle = get_symbol_model_comparison_dashboard_bundle("SPY")

    assert sorted(bundle.keys()) == [
        "confusion_df",
        "confusion_summary_df",
        "panel_df",
        "pivot_df",
        "role_summary_df",
        "summary_df",
    ]
    assert sorted(bundle["confusion_summary_df"]["dataset_role"].unique().tolist()) == [
        "test",
        "validation",
    ]
    assert sorted(bundle["confusion_summary_df"]["model_name"].unique().tolist()) == [
        "garch_11_student_t",
        "xgboost_regressor",
    ]


def test_build_symbol_structural_break_summary_returns_expected_counts() -> None:
    """
    Verifica que el resumen de structural breaks para SPY retorne conteos
    y campos clave esperados.

    Qué comprueba:
    - Que el resumen corresponda a SPY.
    - Que el número total de eventos detectados sea 9.
    - Que el número de filas de la señal sea 2069.
    - Que exista una fecha de quiebre reciente.

    Por qué importa:
    Este test valida una vista agregada importante del módulo de structural
    breaks. Si cambian estos conteos, podría haber cambiado:
    - la señal usada,
    - el algoritmo de segmentación,
    - el preprocesamiento,
    - o el universo temporal cubierto.
    """
    summary = build_symbol_structural_break_summary("SPY")

    # El símbolo reportado debe coincidir con el solicitado.
    assert summary["symbol"] == "SPY"

    # Se espera exactamente este número de quiebres detectados.
    assert summary["event_count"] == 9

    # Se espera exactamente este número de filas en la señal procesada.
    assert summary["signal_rows"] == 2069

    # Debe existir al menos un quiebre reciente.
    assert summary["most_recent_break_date"] is not None


def test_get_symbol_structural_changes_bundle_returns_expected_keys() -> None:
    """
    Verifica que el bundle de structural changes para SPY tenga la estructura
    esperada y tamaños coherentes en sus DataFrames principales.

    Qué comprueba:
    - Que el bundle tenga exactamente las llaves esperadas.
    - Que `signal_df` tenga 2069 filas.
    - Que `events_df` tenga 9 filas.
    - Que `recent_events_df` no supere el límite esperado de 10.

    Por qué importa:
    Este test valida tanto el contrato del bundle como una relación lógica:
    el subconjunto de eventos recientes debe ser como máximo el límite
    configurado, no más grande que eso.

    Nota:
    El último assert usa `<= 10` y no `== 10` porque podrían existir menos
    de 10 eventos totales; en ese caso sigue siendo correcto.
    """
    bundle = get_symbol_structural_changes_bundle("SPY")

    # Verifica el conjunto exacto de llaves del bundle.
    assert sorted(bundle.keys()) == [
        "events_df",
        "recent_events_df",
        "signal_df",
        "summary",
    ]

    # Tamaño esperado de la señal de structural breaks.
    assert len(bundle["signal_df"]) == 2069

    # Tamaño esperado de la tabla completa de eventos.
    assert len(bundle["events_df"]) == 9

    # El subconjunto de eventos recientes debe respetar el límite máximo.
    assert len(bundle["recent_events_df"]) <= 10
