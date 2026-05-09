from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ModelComparisonArtifacts:
    """
    Contenedor inmutable de los artefactos finales de evaluación.

    Attributes
    ----------
    evaluation_panel_df : pd.DataFrame
        Panel unificado a nivel fila que contiene, para cada observación out-of-sample
        (OOS), los valores reales, los pronósticos de cada modelo y columnas auxiliares
        de error.

    metrics_df : pd.DataFrame
        Tabla agregada de métricas por grupo (instrumento, símbolo, split, rol de dataset
        y nombre de modelo). Incluye métricas continuas y discretas.

    confusion_df : pd.DataFrame
        Tabla en formato largo con los conteos de matrices de confusión para la tarea
        de clasificación de regímenes.
    """
    evaluation_panel_df: pd.DataFrame
    metrics_df: pd.DataFrame
    confusion_df: pd.DataFrame


# ---------------------------------------------------------------------
# Columnas mínimas esperadas en cada una de las entradas principales.
# La idea es validar tempranamente la estructura para fallar con mensajes
# claros antes de hacer merges, métricas o comparaciones.
# ---------------------------------------------------------------------

REQUIRED_BENCHMARK_REGIME_COLS = {
    "date",
    "split_id",
    "dataset_role",
    "model_name",
    "future_rv_5d",
    "future_regime_5d",
    "yhat_future_rv_5d",
    "yhat_future_regime_5d",
    "threshold_low",
    "threshold_high",
    "regime_thresholds_source",
}

REQUIRED_ML_FORECAST_COLS = {
    "date",
    "split_id",
    "dataset_role",
    "model_name",
    "future_rv_5d",
    "yhat_future_rv_5d",
}

REQUIRED_REGIME_TARGET_COLS = {
    "instrument_id",
    "date",
    "split_id",
    "dataset_role",
    "future_rv_5d",
    "future_regime_5d",
    "threshold_low",
    "threshold_high",
    "regime_thresholds_source",
}


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: set[str],
    df_name: str,
) -> None:
    """
    Verifica que un DataFrame contenga todas las columnas obligatorias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    required_cols : set[str]
        Conjunto de nombres de columnas que deben existir en `df`.
    df_name : str
        Nombre lógico del DataFrame, usado únicamente para construir
        mensajes de error más claros.

    Raises
    ------
    ValueError
        Si falta al menos una de las columnas requeridas.

    Notes
    -----
    Esta función es una barrera defensiva muy importante:
    evita que errores de esquema aparezcan más adelante en operaciones
    más difíciles de diagnosticar, como merges o métricas agregadas.
    """
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _infer_single_instrument_id(regime_targets_df: pd.DataFrame) -> str:
    """
    Infiera el único `instrument_id` presente en `regime_targets_df`.

    Parameters
    ----------
    regime_targets_df : pd.DataFrame
        DataFrame de targets/regímenes. Debe corresponder a una evaluación
        por símbolo o instrumento, por lo que se espera exactamente un único
        `instrument_id`.

    Returns
    -------
    str
        El `instrument_id` inferido.

    Raises
    ------
    ValueError
        Si el DataFrame contiene cero o más de un `instrument_id`.

    Notes
    -----
    La lógica del módulo asume una construcción de artefactos por símbolo.
    Si aquí aparecen múltiples instrumentos, el resto del pipeline podría
    mezclar observaciones incompatibles.
    """
    instrument_ids = regime_targets_df["instrument_id"].dropna().unique().tolist()
    if len(instrument_ids) != 1:
        raise ValueError(
            "Expected exactly one instrument_id in regime_targets_df for a per-symbol evaluation build."
        )
    return str(instrument_ids[0])


def _infer_single_symbol(
    benchmark_regimes_df: pd.DataFrame,
    ml_forecasts_df: pd.DataFrame,
) -> str:
    """
    Intenta inferir un único símbolo compartido entre benchmark y modelo ML.

    Parameters
    ----------
    benchmark_regimes_df : pd.DataFrame
        DataFrame con pronósticos/regímenes del benchmark.
    ml_forecasts_df : pd.DataFrame
        DataFrame con pronósticos continuos del modelo ML.

    Returns
    -------
    str
        Símbolo único encontrado.
        Si ninguno de los DataFrames trae columna `symbol` o no hay valores
        válidos, retorna `"unknown_symbol"`.

    Raises
    ------
    ValueError
        Si se detecta más de un símbolo entre las entradas.

    Notes
    -----
    Esta función intenta ser tolerante:
    - Si no existe la columna `symbol`, no falla automáticamente.
    - Si no encuentra ningún símbolo, usa un placeholder.
    - Pero si encuentra varios símbolos, sí falla, porque eso implicaría
      una mezcla ambigua de activos.
    """
    benchmark_symbols = (
        benchmark_regimes_df["symbol"].dropna().unique().tolist()
        if "symbol" in benchmark_regimes_df.columns
        else []
    )
    ml_symbols = (
        ml_forecasts_df["symbol"].dropna().unique().tolist()
        if "symbol" in ml_forecasts_df.columns
        else []
    )

    symbols = sorted({str(x) for x in benchmark_symbols + ml_symbols})
    if len(symbols) == 1:
        return symbols[0]
    if len(symbols) == 0:
        return "unknown_symbol"

    raise ValueError(f"Expected a single symbol across inputs, found: {symbols}")


def _assign_regime_labels_from_thresholds(
    predicted_rv: pd.Series,
    threshold_low: pd.Series,
    threshold_high: pd.Series,
    labels: Sequence[str],
) -> pd.Series:
    """
    Convierte un pronóstico continuo de volatilidad en una etiqueta de régimen.

    Parameters
    ----------
    predicted_rv : pd.Series
        Serie con el valor pronosticado de volatilidad futura.
    threshold_low : pd.Series
        Umbral inferior que delimita la clase "calm".
    threshold_high : pd.Series
        Umbral superior que delimita la clase "stress".
    labels : Sequence[str]
        Secuencia de exactamente tres etiquetas en el orden:
        [calm, normal, stress].

    Returns
    -------
    pd.Series
        Serie de etiquetas categóricas inferidas a partir del valor pronosticado
        y los umbrales correspondientes fila a fila.

    Raises
    ------
    ValueError
        Si `labels` no tiene exactamente tres elementos.

    Logic
    -----
    - predicted_rv < threshold_low        -> calm
    - threshold_low <= predicted_rv <= threshold_high -> normal
    - predicted_rv > threshold_high       -> stress

    Notes
    -----
    La asignación se hace sólo sobre filas válidas, es decir, donde no haya
    faltantes en el pronóstico ni en los umbrales. Las filas inválidas quedan
    como `pd.NA`.
    """
    if len(labels) != 3:
        raise ValueError("labels must have exactly three entries: [calm, normal, stress]")

    calm_label, normal_label, stress_label = labels

    # Inicializamos la salida con NA para preservar explícitamente
    # las filas donde no se pueda hacer clasificación.
    out = pd.Series(pd.NA, index=predicted_rv.index, dtype="object")

    # Sólo se puede asignar régimen cuando existen los tres valores:
    # pronóstico, umbral inferior y umbral superior.
    valid_mask = predicted_rv.notna() & threshold_low.notna() & threshold_high.notna()

    out.loc[valid_mask & (predicted_rv < threshold_low)] = calm_label
    out.loc[valid_mask & (predicted_rv >= threshold_low) & (predicted_rv <= threshold_high)] = normal_label
    out.loc[valid_mask & (predicted_rv > threshold_high)] = stress_label

    return out


def _qlike_from_volatility_series(
    y_true_vol: pd.Series,
    y_pred_vol: pd.Series,
    eps: float = 1e-12,
) -> float:
    """
    Calcula la métrica QLIKE a partir de series de volatilidad.

    Parameters
    ----------
    y_true_vol : pd.Series
        Serie de volatilidad realizada/real.
    y_pred_vol : pd.Series
        Serie de volatilidad pronosticada.
    eps : float, default=1e-12
        Valor mínimo para recortar varianzas y evitar problemas numéricos
        como divisiones por cero o logaritmos de cero.

    Returns
    -------
    float
        Promedio de los términos QLIKE para las observaciones válidas.
        Devuelve `np.nan` si no hay observaciones comparables.

    Notes
    -----
    La función transforma volatilidades en varianzas mediante el cuadrado:

        var = vol^2

    y luego aplica:

        QLIKE = log(pred_var) + true_var / pred_var

    Es una métrica muy usada en forecasting de volatilidad porque penaliza
    fuertemente ciertas malas estimaciones de varianza y suele ser más
    apropiada que comparar volatilidades sólo con error cuadrático.
    """
    valid_mask = y_true_vol.notna() & y_pred_vol.notna()
    if not valid_mask.any():
        return np.nan

    y_true_var = np.clip(np.square(y_true_vol.loc[valid_mask].astype(float).to_numpy()), eps, None)
    y_pred_var = np.clip(np.square(y_pred_vol.loc[valid_mask].astype(float).to_numpy()), eps, None)

    qlike_terms = np.log(y_pred_var) + (y_true_var / y_pred_var)
    return float(np.mean(qlike_terms))


def _rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calcula el Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : pd.Series
        Valores reales.
    y_pred : pd.Series
        Valores predichos.

    Returns
    -------
    float
        RMSE sobre las filas válidas.
        Devuelve `np.nan` si no hay pares comparables.

    Notes
    -----
    El RMSE penaliza más fuertemente los errores grandes porque eleva
    al cuadrado las diferencias antes de promediarlas.
    """
    valid_mask = y_true.notna() & y_pred.notna()
    if not valid_mask.any():
        return np.nan

    errors = y_true.loc[valid_mask].astype(float).to_numpy() - y_pred.loc[valid_mask].astype(float).to_numpy()
    return float(np.sqrt(np.mean(np.square(errors))))


def _mae(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calcula el Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : pd.Series
        Valores reales.
    y_pred : pd.Series
        Valores predichos.

    Returns
    -------
    float
        MAE sobre las filas válidas.
        Devuelve `np.nan` si no hay pares comparables.

    Notes
    -----
    A diferencia del RMSE, el MAE no amplifica tanto los errores extremos.
    Por eso suele interpretarse como una medida más directa del error medio
    absoluto en las mismas unidades del target.
    """
    valid_mask = y_true.notna() & y_pred.notna()
    if not valid_mask.any():
        return np.nan

    errors = np.abs(
        y_true.loc[valid_mask].astype(float).to_numpy()
        - y_pred.loc[valid_mask].astype(float).to_numpy()
    )
    return float(np.mean(errors))


def _macro_f1(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Sequence[str],
) -> float:
    """
    Calcula el F1 macro para una clasificación multiclase de regímenes.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_pred : pd.Series
        Etiquetas predichas.
    labels : Sequence[str]
        Orden explícito de las clases a evaluar.

    Returns
    -------
    float
        F1 macro sobre las observaciones válidas.
        Devuelve `np.nan` si no hay observaciones comparables.

    Notes
    -----
    `average="macro"` significa que se calcula el F1 de cada clase y luego
    se promedian todos por igual, sin ponderar por frecuencia de clase.
    Esto es útil cuando importa tratar de manera equilibrada clases
    desbalanceadas.
    """
    valid_mask = y_true.notna() & y_pred.notna()
    if not valid_mask.any():
        return np.nan

    return float(
        f1_score(
            y_true.loc[valid_mask],
            y_pred.loc[valid_mask],
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def _balanced_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calcula balanced accuracy para la clasificación de regímenes.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_pred : pd.Series
        Etiquetas predichas.

    Returns
    -------
    float
        Balanced accuracy sobre las observaciones válidas.
        Devuelve `np.nan` si no hay datos comparables.

    Notes
    -----
    Balanced accuracy promedia el recall por clase, lo cual la hace útil
    cuando las clases no están balanceadas. Así evita que una clase dominante
    infle artificialmente el desempeño.
    """
    valid_mask = y_true.notna() & y_pred.notna()
    if not valid_mask.any():
        return np.nan

    return float(balanced_accuracy_score(y_true.loc[valid_mask], y_pred.loc[valid_mask]))


def _build_benchmark_panel(
    benchmark_regimes_df: pd.DataFrame,
    instrument_id: str,
    symbol: str,
) -> pd.DataFrame:
    """
    Estandariza el panel proveniente del benchmark para que tenga el esquema
    común usado en la comparación final.

    Parameters
    ----------
    benchmark_regimes_df : pd.DataFrame
        DataFrame original del benchmark con targets, pronósticos y regímenes.
    instrument_id : str
        Identificador del instrumento a anexar explícitamente.
    symbol : str
        Símbolo del activo a anexar explícitamente.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ordenadas y consistentes con el esquema de
        evaluación.

    Notes
    -----
    Esta función no recalcula pronósticos ni regímenes:
    simplemente toma lo que ya viene del benchmark, añade metadatos
    (`instrument_id`, `symbol`) y conserva únicamente las columnas relevantes.
    """
    benchmark_panel_df = benchmark_regimes_df.copy()

    # Se fuerzan estos metadatos para asegurar consistencia con el panel ML.
    benchmark_panel_df["instrument_id"] = instrument_id
    benchmark_panel_df["symbol"] = symbol

    keep_cols = [
        "instrument_id",
        "symbol",
        "date",
        "split_id",
        "dataset_role",
        "model_name",
        "future_rv_5d",
        "future_regime_5d",
        "yhat_future_rv_5d",
        "yhat_future_regime_5d",
        "threshold_low",
        "threshold_high",
        "regime_thresholds_source",
    ]
    benchmark_panel_df = benchmark_panel_df[keep_cols].copy()
    return benchmark_panel_df


def _build_ml_panel(
    ml_forecasts_df: pd.DataFrame,
    regime_targets_df: pd.DataFrame,
    instrument_id: str,
    symbol: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    """
    Construye el panel del modelo ML en el mismo formato que el benchmark.

    Parameters
    ----------
    ml_forecasts_df : pd.DataFrame
        Pronósticos continuos del modelo ML.
    regime_targets_df : pd.DataFrame
        DataFrame con targets reales y umbrales necesarios para derivar
        la clase de régimen pronosticada.
    instrument_id : str
        Identificador del instrumento.
    symbol : str
        Símbolo del activo.
    labels : Sequence[str]
        Etiquetas de clases en el orden [calm, normal, stress].

    Returns
    -------
    pd.DataFrame
        Panel ML con targets reales, pronóstico continuo, pronóstico discreto
        de régimen y metadatos estandarizados.

    Workflow
    --------
    1. Selecciona de `regime_targets_df` sólo las columnas necesarias para el join.
    2. Hace merge one-to-one con los pronósticos ML usando llaves OOS.
    3. Añade `instrument_id` y `symbol`.
    4. Deriva `yhat_future_regime_5d` a partir de `yhat_future_rv_5d`
       y de los umbrales reales de cada fila.
    5. Reordena y filtra columnas.

    Notes
    -----
    La validación `validate="one_to_one"` en el merge es importante:
    obliga a que cada observación ML se empareje con una única fila de targets,
    evitando duplicaciones silenciosas.
    """
    join_keys = ["date", "split_id", "dataset_role"]

    regime_join_df = regime_targets_df[
        [
            "instrument_id",
            "date",
            "split_id",
            "dataset_role",
            "future_rv_5d",
            "future_regime_5d",
            "threshold_low",
            "threshold_high",
            "regime_thresholds_source",
        ]
    ].copy()

    # Se incorporan al forecast ML los targets reales y los umbrales
    # necesarios para convertir un forecast continuo en régimen discreto.
    ml_panel_df = ml_forecasts_df.merge(
        regime_join_df,
        on=join_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_from_regime_targets"),
    )

    ml_panel_df["instrument_id"] = instrument_id
    ml_panel_df["symbol"] = symbol

    # A partir del pronóstico continuo de RV, se crea la clase predicha
    # usando los umbrales por fila.
    ml_panel_df["yhat_future_regime_5d"] = _assign_regime_labels_from_thresholds(
        predicted_rv=ml_panel_df["yhat_future_rv_5d"],
        threshold_low=ml_panel_df["threshold_low"],
        threshold_high=ml_panel_df["threshold_high"],
        labels=labels,
    )

    keep_cols = [
        "instrument_id",
        "symbol",
        "date",
        "split_id",
        "dataset_role",
        "model_name",
        "future_rv_5d",
        "future_regime_5d",
        "yhat_future_rv_5d",
        "yhat_future_regime_5d",
        "threshold_low",
        "threshold_high",
        "regime_thresholds_source",
    ]
    ml_panel_df = ml_panel_df[keep_cols].copy()
    return ml_panel_df


def _validate_oos_alignment(
    benchmark_panel_df: pd.DataFrame,
    ml_panel_df: pd.DataFrame,
) -> None:
    """
    Verifica que benchmark y ML estén perfectamente alineados en sus llaves OOS.

    Parameters
    ----------
    benchmark_panel_df : pd.DataFrame
        Panel ya estandarizado del benchmark.
    ml_panel_df : pd.DataFrame
        Panel ya estandarizado del modelo ML.

    Raises
    ------
    ValueError
        Si alguno de los paneles tiene llaves duplicadas o si los conjuntos
        de llaves no coinciden exactamente.

    Notes
    -----
    La comparación justa entre modelos exige que ambos hayan sido evaluados
    exactamente sobre las mismas observaciones OOS. Esta función impone eso.

    Las llaves de alineación son:
    - date
    - split_id
    - dataset_role
    """
    key_cols = ["date", "split_id", "dataset_role"]

    benchmark_dup_count = int(benchmark_panel_df.duplicated(subset=key_cols).sum())
    ml_dup_count = int(ml_panel_df.duplicated(subset=key_cols).sum())

    if benchmark_dup_count != 0:
        raise ValueError(f"benchmark_panel_df has duplicated OOS keys: {benchmark_dup_count}")
    if ml_dup_count != 0:
        raise ValueError(f"ml_panel_df has duplicated OOS keys: {ml_dup_count}")

    benchmark_keys = benchmark_panel_df[key_cols].sort_values(key_cols).reset_index(drop=True)
    ml_keys = ml_panel_df[key_cols].sort_values(key_cols).reset_index(drop=True)

    if not benchmark_keys.equals(ml_keys):
        raise ValueError("Benchmark and ML OOS keys do not align exactly.")


def _add_row_level_error_columns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de error a nivel observación para el target continuo.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel de evaluación que debe contener al menos:
        - future_rv_5d
        - yhat_future_rv_5d

    Returns
    -------
    pd.DataFrame
        Copia del panel original con columnas extra:
        - error
        - abs_error
        - sq_error
        - qlike_term

    Notes
    -----
    Estas columnas sirven para análisis más finos a nivel fila, inspección
    de errores por fecha/split y futuras agregaciones personalizadas.
    """
    out = panel_df.copy()

    # Error firmado: positivo si el modelo sobreestima, negativo si subestima.
    out["error"] = out["yhat_future_rv_5d"] - out["future_rv_5d"]

    # Error absoluto: magnitud del error sin signo.
    out["abs_error"] = out["error"].abs()

    # Error cuadrático: útil para RMSE o penalizar más los errores grandes.
    out["sq_error"] = np.square(out["error"])

    # Término por fila de la métrica QLIKE.
    eps = 1e-12
    true_var = np.clip(np.square(out["future_rv_5d"].astype(float)), eps, None)
    pred_var = np.clip(np.square(out["yhat_future_rv_5d"].astype(float)), eps, None)
    out["qlike_term"] = np.log(pred_var) + (true_var / pred_var)

    return out


def _build_metrics_df(
    evaluation_panel_df: pd.DataFrame,
    evaluation_version: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    """
    Construye la tabla agregada de métricas por grupo de evaluación.

    Parameters
    ----------
    evaluation_panel_df : pd.DataFrame
        Panel consolidado con benchmark y ML, incluyendo targets y predicciones.
    evaluation_version : str
        Versión lógica de la evaluación, útil para trazabilidad.
    labels : Sequence[str]
        Etiquetas de clase usadas para las métricas discretas.

    Returns
    -------
    pd.DataFrame
        Tabla en formato largo donde cada fila representa una métrica
        para un grupo específico.

    Grouping
    --------
    Las métricas se calculan por:
    - instrument_id
    - symbol
    - split_id
    - dataset_role
    - model_name

    Metrics
    -------
    Continuas:
    - qlike
    - rmse
    - mae

    Discretas:
    - macro_f1
    - balanced_accuracy

    Notes
    -----
    El resultado queda en formato largo porque es mucho más flexible
    para:
    - exportar,
    - graficar,
    - pivotear después,
    - comparar modelos de forma uniforme.
    """
    metric_rows: list[dict[str, object]] = []

    group_cols = ["instrument_id", "symbol", "split_id", "dataset_role", "model_name"]

    for group_keys, group_df in evaluation_panel_df.groupby(group_cols, dropna=False):
        instrument_id, symbol, split_id, dataset_role, model_name = group_keys

        # Número de observaciones válidas para métricas continuas.
        continuous_n = int((group_df["future_rv_5d"].notna() & group_df["yhat_future_rv_5d"].notna()).sum())

        # Número de observaciones válidas para métricas discretas.
        discrete_n = int(
            (group_df["future_regime_5d"].notna() & group_df["yhat_future_regime_5d"].notna()).sum()
        )

        metric_rows.extend(
            [
                {
                    "evaluation_version": evaluation_version,
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": model_name,
                    "target_family": "continuous",
                    "metric_name": "qlike",
                    "metric_value": _qlike_from_volatility_series(
                        group_df["future_rv_5d"],
                        group_df["yhat_future_rv_5d"],
                    ),
                    "n_obs": continuous_n,
                },
                {
                    "evaluation_version": evaluation_version,
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": model_name,
                    "target_family": "continuous",
                    "metric_name": "rmse",
                    "metric_value": _rmse(
                        group_df["future_rv_5d"],
                        group_df["yhat_future_rv_5d"],
                    ),
                    "n_obs": continuous_n,
                },
                {
                    "evaluation_version": evaluation_version,
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": model_name,
                    "target_family": "continuous",
                    "metric_name": "mae",
                    "metric_value": _mae(
                        group_df["future_rv_5d"],
                        group_df["yhat_future_rv_5d"],
                    ),
                    "n_obs": continuous_n,
                },
                {
                    "evaluation_version": evaluation_version,
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": model_name,
                    "target_family": "discrete",
                    "metric_name": "macro_f1",
                    "metric_value": _macro_f1(
                        group_df["future_regime_5d"],
                        group_df["yhat_future_regime_5d"],
                        labels=labels,
                    ),
                    "n_obs": discrete_n,
                },
                {
                    "evaluation_version": evaluation_version,
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "split_id": split_id,
                    "dataset_role": dataset_role,
                    "model_name": model_name,
                    "target_family": "discrete",
                    "metric_name": "balanced_accuracy",
                    "metric_value": _balanced_accuracy(
                        group_df["future_regime_5d"],
                        group_df["yhat_future_regime_5d"],
                    ),
                    "n_obs": discrete_n,
                },
            ]
        )

    return pd.DataFrame(metric_rows)


def _build_confusion_df(
    evaluation_panel_df: pd.DataFrame,
    evaluation_version: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    """
    Construye una tabla en formato largo con matrices de confusión.

    Parameters
    ----------
    evaluation_panel_df : pd.DataFrame
        Panel consolidado con etiquetas reales y predichas.
    evaluation_version : str
        Versión lógica de la evaluación.
    labels : Sequence[str]
        Orden de clases a respetar en la matriz de confusión.

    Returns
    -------
    pd.DataFrame
        Tabla en formato largo donde cada fila representa una celda
        de la matriz de confusión:
        (`y_true`, `y_pred`, `count`) por grupo de evaluación.

    Notes
    -----
    En vez de devolver una matriz 2D por grupo, se descompone todo en filas.
    Esto facilita persistencia, filtrado, pivot posterior y visualización.
    """
    confusion_rows: list[dict[str, object]] = []

    group_cols = ["instrument_id", "symbol", "split_id", "dataset_role", "model_name"]

    for group_keys, group_df in evaluation_panel_df.groupby(group_cols, dropna=False):
        instrument_id, symbol, split_id, dataset_role, model_name = group_keys

        # Sólo se consideran filas donde haya tanto etiqueta real como predicha.
        valid_df = group_df[
            group_df["future_regime_5d"].notna() & group_df["yhat_future_regime_5d"].notna()
        ].copy()

        if valid_df.empty:
            continue

        matrix = confusion_matrix(
            valid_df["future_regime_5d"],
            valid_df["yhat_future_regime_5d"],
            labels=list(labels),
        )

        # Convertimos la matriz a formato largo para mayor flexibilidad.
        for i, y_true in enumerate(labels):
            for j, y_pred in enumerate(labels):
                confusion_rows.append(
                    {
                        "evaluation_version": evaluation_version,
                        "instrument_id": instrument_id,
                        "symbol": symbol,
                        "split_id": split_id,
                        "dataset_role": dataset_role,
                        "model_name": model_name,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "count": int(matrix[i, j]),
                    }
                )

    return pd.DataFrame(confusion_rows)


def build_model_comparison_artifacts(
    benchmark_regimes_df: pd.DataFrame,
    ml_forecasts_df: pd.DataFrame,
    regime_targets_df: pd.DataFrame,
    evaluation_version: str = "v1",
    labels: Sequence[str] = ("calm", "normal", "stress"),
) -> ModelComparisonArtifacts:
    """
    Orquesta la construcción completa de los artefactos de comparación
    entre un benchmark y un modelo ML.

    Parameters
    ----------
    benchmark_regimes_df : pd.DataFrame
        DataFrame del benchmark que ya incluye:
        - target continuo,
        - target discreto,
        - forecast continuo,
        - forecast discreto,
        - umbrales.
    ml_forecasts_df : pd.DataFrame
        DataFrame del modelo ML con forecast continuo y metadatos OOS.
    regime_targets_df : pd.DataFrame
        DataFrame con targets reales y umbrales por observación, usado para
        enriquecer el panel ML y derivar su forecast discreto.
    evaluation_version : str, default="v1"
        Identificador/versionado de la evaluación.
    labels : Sequence[str], default=("calm", "normal", "stress")
        Etiquetas de clases usadas en clasificación de regímenes.

    Returns
    -------
    ModelComparisonArtifacts
        Objeto con:
        - evaluation_panel_df
        - metrics_df
        - confusion_df

    Pipeline
    --------
    1. Valida que las entradas tengan el esquema mínimo requerido.
    2. Infiera `instrument_id` y `symbol`.
    3. Construye el panel estandarizado del benchmark.
    4. Construye el panel estandarizado del modelo ML.
    5. Verifica alineación exacta OOS entre benchmark y ML.
    6. Concatena ambos paneles.
    7. Añade columnas de error a nivel fila.
    8. Calcula métricas agregadas.
    9. Construye matrices de confusión en formato largo.
    10. Devuelve todo empaquetado en un dataclass inmutable.

    Notes
    -----
    Esta es la función principal del módulo. Su propósito no es entrenar modelos,
    sino producir artefactos limpios y comparables para evaluación downstream,
    reporting o persistencia.
    """
    # -----------------------------------------------------------------
    # 1) Validación de esquema de entrada
    # -----------------------------------------------------------------
    _validate_required_columns(
        benchmark_regimes_df,
        REQUIRED_BENCHMARK_REGIME_COLS,
        "benchmark_regimes_df",
    )
    _validate_required_columns(
        ml_forecasts_df,
        REQUIRED_ML_FORECAST_COLS,
        "ml_forecasts_df",
    )
    _validate_required_columns(
        regime_targets_df,
        REQUIRED_REGIME_TARGET_COLS,
        "regime_targets_df",
    )

    # -----------------------------------------------------------------
    # 2) Inferencia de metadatos principales
    # -----------------------------------------------------------------
    instrument_id = _infer_single_instrument_id(regime_targets_df)
    symbol = _infer_single_symbol(benchmark_regimes_df, ml_forecasts_df)

    # -----------------------------------------------------------------
    # 3) Construcción de paneles homogéneos
    # -----------------------------------------------------------------
    benchmark_panel_df = _build_benchmark_panel(
        benchmark_regimes_df=benchmark_regimes_df,
        instrument_id=instrument_id,
        symbol=symbol,
    )
    ml_panel_df = _build_ml_panel(
        ml_forecasts_df=ml_forecasts_df,
        regime_targets_df=regime_targets_df,
        instrument_id=instrument_id,
        symbol=symbol,
        labels=labels,
    )

    # -----------------------------------------------------------------
    # 4) Validación de alineación exacta OOS entre modelos
    # -----------------------------------------------------------------
    _validate_oos_alignment(
        benchmark_panel_df=benchmark_panel_df,
        ml_panel_df=ml_panel_df,
    )

    # -----------------------------------------------------------------
    # 5) Unificación de benchmark + ML en un solo panel de evaluación
    # -----------------------------------------------------------------
    evaluation_panel_df = pd.concat(
        [benchmark_panel_df, ml_panel_df],
        axis=0,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # 6) Cálculo de errores fila a fila para el target continuo
    # -----------------------------------------------------------------
    evaluation_panel_df = _add_row_level_error_columns(evaluation_panel_df)

    # -----------------------------------------------------------------
    # 7) Construcción de métricas agregadas y matrices de confusión
    # -----------------------------------------------------------------
    metrics_df = _build_metrics_df(
        evaluation_panel_df=evaluation_panel_df,
        evaluation_version=evaluation_version,
        labels=labels,
    )
    confusion_df = _build_confusion_df(
        evaluation_panel_df=evaluation_panel_df,
        evaluation_version=evaluation_version,
        labels=labels,
    )

    # -----------------------------------------------------------------
    # 8) Empaquetado final de artefactos
    # -----------------------------------------------------------------
    return ModelComparisonArtifacts(
        evaluation_panel_df=evaluation_panel_df,
        metrics_df=metrics_df,
        confusion_df=confusion_df,
    )