from __future__ import annotations

from typing import Any

import pandas as pd

from .artifact_loaders import (
    load_symbol_model_comparison_confusion_matrix,
    load_symbol_model_comparison_metrics,
    load_symbol_model_comparison_panel,
)


def get_symbol_model_comparison_metrics(symbol: str) -> pd.DataFrame:
    """
    Carga y ordena el artefacto de métricas de comparación de modelos para
    un símbolo.

    Propósito:
    Esta función devuelve la tabla "larga" de métricas tal como sale del
    pipeline de evaluación, pero ordenada de forma estable y predecible.
    Eso facilita consumo posterior por otras funciones del módulo y por
    capas superiores.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un DataFrame ordenado por:
      1. split_id
      2. dataset_role
      3. metric_name
      4. model_name

    Idea de diseño:
    La función no transforma la semántica del artefacto; solo lo carga y lo
    deja en un orden consistente. Es una función de acceso limpio a los datos.
    """
    # Carga el parquet de métricas del símbolo y trabaja sobre una copia
    # para evitar efectos colaterales sobre el DataFrame original.
    df = load_symbol_model_comparison_metrics(symbol).copy()

    # Ordena las filas para que la tabla tenga un orden determinista:
    # primero por split, luego por rol del dataset, después por métrica
    # y finalmente por modelo.
    return df.sort_values(
        ["split_id", "dataset_role", "metric_name", "model_name"]
    ).reset_index(drop=True)


def get_symbol_model_comparison_panel(symbol: str) -> pd.DataFrame:
    """
    Carga y ordena el panel de comparación de modelos para un símbolo.

    Propósito:
    Exponer el artefacto tipo panel en una forma ya normalizada temporalmente,
    con la columna `date` convertida explícitamente a datetime y con orden
    cronológico estable.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un DataFrame ordenado por:
      1. date
      2. split_id
      3. dataset_role
      4. model_name

    Importancia:
    Este panel suele ser útil para inspección temporal, visualización o
    análisis por fecha. Por eso aquí sí tiene sentido forzar el parseo
    correcto de `date`.
    """
    # Carga el panel base.
    df = load_symbol_model_comparison_panel(symbol).copy()

    # Convierte la columna date a datetime de forma estricta. Si algo falla,
    # se considera que el artefacto está mal formado.
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # Devuelve el panel ordenado cronológicamente y con índice limpio.
    return df.sort_values(
        ["date", "split_id", "dataset_role", "model_name"]
    ).reset_index(drop=True)


def get_symbol_model_comparison_confusion_matrix(symbol: str) -> pd.DataFrame:
    """
    Carga y ordena la confusion matrix de model comparison para un símbolo.

    Propósito:
    Exponer los conteos de clasificación por modelo, split y dataset_role en
    una forma estable para capas superiores.
    """
    df = load_symbol_model_comparison_confusion_matrix(symbol).copy()

    return df.sort_values(
        ["dataset_role", "model_name", "split_id", "y_true", "y_pred"]
    ).reset_index(drop=True)


def get_symbol_model_comparison_pivot(symbol: str) -> pd.DataFrame:
    """
    Construye una vista pivotada de las métricas de comparación para un símbolo.

    Propósito:
    Transformar la tabla larga de métricas en una tabla ancha donde las métricas
    de los dos modelos queden lado a lado, permitiendo comparar benchmark vs ML
    de manera directa y calcular deltas e improvement relativo.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un DataFrame pivotado con:
      - identificación del instrumento/símbolo/split/rol/métrica
      - nombre del benchmark y del modelo ML
      - valor de la métrica para cada modelo
      - dirección de optimización
      - diferencias absolutas
      - mejora relativa contra benchmark

    Contrato importante:
    Esta función asume que, tras pivotar, habrá exactamente dos columnas
    de modelos:
    - una para benchmark
    - una para ML

    Si aparecen más o menos de dos, lanza un error porque la lógica de
    comparación binaria dejaría de ser válida.
    """
    # Parte de la tabla larga de métricas ya ordenada.
    metrics_df = get_symbol_model_comparison_metrics(symbol)

    # Pivot table:
    # - cada fila representa una combinación única de
    #   instrumento, símbolo, split, rol de dataset y nombre de métrica
    # - cada columna de modelo contiene el valor de la métrica para ese modelo
    #
    # aggfunc="first" asume que cada combinación debería producir a lo sumo
    # un único valor; si hubiera duplicados, toma el primero.
    pivot_df = metrics_df.pivot_table(
        index=["instrument_id", "symbol", "split_id", "dataset_role", "metric_name"],
        columns="model_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    # Elimina el nombre del eje de columnas que deja pivot_table para que
    # el resultado sea más limpio y fácil de manipular.
    pivot_df.columns.name = None

    # Métricas donde un valor menor indica mejor desempeño.
    lower_is_better = {"qlike", "rmse", "mae"}

    # Métricas donde un valor mayor indica mejor desempeño.
    higher_is_better = {"macro_f1", "balanced_accuracy"}

    # Identifica cuáles columnas representan modelos.
    # Excluye las columnas de identificación/base.
    model_cols = [
        c
        for c in pivot_df.columns
        if c
        not in {"instrument_id", "symbol", "split_id", "dataset_role", "metric_name"}
    ]

    # La lógica posterior asume comparación entre exactamente dos modelos.
    if len(model_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 model columns in pivot output, found: {model_cols}"
        )

    # Asigna los nombres de columnas a benchmark y ML de forma determinista.
    # Aquí no se usan etiquetas semánticas explícitas desde el archivo, sino
    # un orden alfabético estable.
    benchmark_col, ml_col = sorted(model_cols)

    # Guarda explícitamente los nombres de los modelos dentro del DataFrame
    # final para que la salida sea autoexplicativa.
    pivot_df["benchmark_model_name"] = benchmark_col
    pivot_df["ml_model_name"] = ml_col

    # Copia los valores de la métrica de cada modelo a columnas con nombres
    # semánticos más claros.
    pivot_df["benchmark_metric_value"] = pivot_df[benchmark_col]
    pivot_df["ml_metric_value"] = pivot_df[ml_col]

    # Determina si para cada métrica "más bajo es mejor" o "más alto es mejor".
    #
    # Observación crítica:
    # Si aparece una métrica no contemplada en ninguno de los dos sets,
    # esta expresión la marcará como "higher_is_better", lo cual puede ser
    # una simplificación arriesgada.
    pivot_df["optimization_direction"] = pivot_df["metric_name"].map(
        lambda x: "lower_is_better" if x in lower_is_better else "higher_is_better"
    )

    # Diferencia directa: ML - benchmark.
    #
    # Interpretación:
    # - positiva: ML mayor que benchmark
    # - negativa: ML menor que benchmark
    #
    # Esta columna sola no basta para decidir si ML es mejor, porque eso
    # depende de la dirección de optimización de la métrica.
    pivot_df["ml_minus_benchmark"] = (
        pivot_df["ml_metric_value"] - pivot_df["benchmark_metric_value"]
    )

    # Diferencia inversa: benchmark - ML.
    #
    # Esta columna resulta especialmente útil para métricas donde
    # "lower is better", porque una reducción frente al benchmark
    # se ve como valor positivo aquí.
    pivot_df["benchmark_minus_ml"] = (
        pivot_df["benchmark_metric_value"] - pivot_df["ml_metric_value"]
    )

    # Construye un denominador seguro usando el valor absoluto del benchmark.
    # Reemplaza 0.0 por NA para evitar divisiones por cero.
    #
    # Nota:
    # Esta variable luego no se usa directamente en la implementación actual.
    # Probablemente quedó de una versión previa o como preparación para una
    # fórmula más compacta.
    denom = pivot_df["benchmark_metric_value"].abs().replace(0.0, pd.NA)

    # Calcula la mejora relativa del modelo ML frente al benchmark.
    #
    # Lógica:
    # - Si lower_is_better:
    #     improvement = (benchmark - ml) / |benchmark|
    #   porque ML mejora cuando reduce el valor de la métrica.
    #
    # - Si higher_is_better:
    #     improvement = (ml - benchmark) / |benchmark|
    #   porque ML mejora cuando aumenta el valor.
    #
    # - Si benchmark es nulo o cero:
    #     retorna NA para evitar resultados inválidos.
    #
    # Se usa apply fila a fila porque la fórmula depende del tipo de métrica.
    pivot_df["relative_improvement_vs_benchmark"] = pivot_df.apply(
        lambda row: (
            row["benchmark_minus_ml"] / abs(row["benchmark_metric_value"])
            if row["metric_name"] in lower_is_better
            and pd.notna(row["benchmark_metric_value"])
            and row["benchmark_metric_value"] != 0
            else (
                row["ml_minus_benchmark"] / abs(row["benchmark_metric_value"])
                if row["metric_name"] in higher_is_better
                and pd.notna(row["benchmark_metric_value"])
                and row["benchmark_metric_value"] != 0
                else pd.NA
            )
        ),
        axis=1,
    )

    # Selecciona solo las columnas finales que interesan para consumo externo.
    keep_cols = [
        "instrument_id",
        "symbol",
        "split_id",
        "dataset_role",
        "metric_name",
        "benchmark_model_name",
        "ml_model_name",
        "benchmark_metric_value",
        "ml_metric_value",
        "optimization_direction",
        "ml_minus_benchmark",
        "benchmark_minus_ml",
        "relative_improvement_vs_benchmark",
    ]

    # Devuelve la tabla final ordenada de manera estable.
    return (
        pivot_df[keep_cols]
        .sort_values(["split_id", "dataset_role", "metric_name"])
        .reset_index(drop=True)
    )


def build_symbol_model_comparison_summary(symbol: str) -> pd.DataFrame:
    """
    Construye un resumen agregado de comparación entre modelos para un símbolo.

    Propósito:
    Pasar de la tabla pivotada por split/rol/métrica a una tabla resumida
    que agregue resultados a nivel de métrica, promediando benchmark, ML,
    delta y mejora relativa.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un DataFrame resumido con columnas como:
      - benchmark_mean
      - ml_mean
      - mean_delta
      - mean_relative_improvement
      - comparisons

    Significado:
    Esta tabla responde preguntas del tipo:
    - “En promedio, ¿cómo le fue al ML frente al benchmark en RMSE?”
    - “¿Cuántas comparaciones entraron en ese promedio?”
    """
    # Parte de la tabla pivotada ya enriquecida con deltas e improvement relativo.
    pivot_df = get_symbol_model_comparison_pivot(symbol)

    # Agrupa por símbolo/métrica y nombres de modelos para resumir múltiples
    # observaciones provenientes de distintos splits y roles de dataset.
    summary_df = (
        pivot_df.groupby(
            [
                "instrument_id",
                "symbol",
                "metric_name",
                "benchmark_model_name",
                "ml_model_name",
            ],
            as_index=False,
        )
        .agg(
            # Promedio del benchmark para esa métrica.
            benchmark_mean=("benchmark_metric_value", "mean"),
            # Promedio del modelo ML para esa métrica.
            ml_mean=("ml_metric_value", "mean"),
            # Promedio de la diferencia directa ML - benchmark.
            mean_delta=("ml_minus_benchmark", "mean"),
            # Promedio de la mejora relativa contra benchmark.
            mean_relative_improvement=("relative_improvement_vs_benchmark", "mean"),
            # Número de comparaciones individuales que alimentaron el agregado.
            comparisons=("metric_name", "size"),
        )
        .sort_values(["symbol", "metric_name"])
        .reset_index(drop=True)
    )

    return summary_df


def build_symbol_model_comparison_role_summary(symbol: str) -> pd.DataFrame:
    """
    Resume la comparación benchmark vs ML por `dataset_role` y métrica.

    Propósito:
    Ofrecer una vista compacta para UI donde se quiera separar validation y
    test sin exponer la granularidad completa por split.
    """
    pivot_df = get_symbol_model_comparison_pivot(symbol)

    summary_df = (
        pivot_df.groupby(
            [
                "instrument_id",
                "symbol",
                "dataset_role",
                "metric_name",
                "benchmark_model_name",
                "ml_model_name",
                "optimization_direction",
            ],
            as_index=False,
        )
        .agg(
            benchmark_mean=("benchmark_metric_value", "mean"),
            ml_mean=("ml_metric_value", "mean"),
            mean_delta=("ml_minus_benchmark", "mean"),
            mean_relative_improvement=("relative_improvement_vs_benchmark", "mean"),
            comparisons=("metric_name", "size"),
        )
        .sort_values(["dataset_role", "metric_name"])
        .reset_index(drop=True)
    )

    return summary_df


def build_symbol_model_comparison_confusion_summary(symbol: str) -> pd.DataFrame:
    """
    Agrega las confusion matrices por modelo y dataset_role.

    Propósito:
    Colapsar los conteos por split a una sola matriz por modelo y por rol,
    facilitando heatmaps compactos en el dashboard.
    """
    confusion_df = get_symbol_model_comparison_confusion_matrix(symbol)

    summary_df = (
        confusion_df.groupby(
            [
                "instrument_id",
                "symbol",
                "dataset_role",
                "model_name",
                "y_true",
                "y_pred",
            ],
            as_index=False,
        )
        .agg(count=("count", "sum"))
        .sort_values(["dataset_role", "model_name", "y_true", "y_pred"])
        .reset_index(drop=True)
    )

    row_totals = summary_df.groupby(["dataset_role", "model_name", "y_true"])[
        "count"
    ].transform("sum")
    safe_totals = row_totals.where(row_totals != 0, pd.NA)
    summary_df["row_total"] = row_totals
    summary_df["row_share"] = summary_df["count"] / safe_totals

    return summary_df


def get_symbol_model_comparison_dashboard_bundle(symbol: str) -> dict[str, Any]:
    """
    Devuelve un bundle orientado específicamente a la UI comparativa.

    Propósito:
    Reunir resúmenes agregados, vistas por role, panel de forecasts y matrices
    de confusión en una sola estructura lista para visualización.
    """
    return {
        "summary_df": build_symbol_model_comparison_summary(symbol),
        "role_summary_df": build_symbol_model_comparison_role_summary(symbol),
        "pivot_df": get_symbol_model_comparison_pivot(symbol),
        "panel_df": get_symbol_model_comparison_panel(symbol),
        "confusion_df": get_symbol_model_comparison_confusion_matrix(symbol),
        "confusion_summary_df": build_symbol_model_comparison_confusion_summary(symbol),
    }


def get_symbol_model_comparison_bundle(symbol: str) -> dict[str, Any]:
    """
    Devuelve un bundle completo con todas las vistas principales de comparación
    de modelos para un símbolo.

    Propósito:
    Reunir en una sola estructura:
    - la tabla larga de métricas,
    - la tabla pivotada comparativa,
    - el resumen agregado,
    - el panel temporal/detallado.

    Parámetros:
    - symbol:
      símbolo a consultar.

    Retorna:
    - Un diccionario con cuatro entradas:
      * "metrics_df"
      * "pivot_df"
      * "summary_df"
      * "panel_df"

    Utilidad:
    Esta función es conveniente para capas superiores que necesiten tanto
    una vista resumida como tablas detalladas de respaldo sin llamar varias
    funciones por separado.
    """
    return {
        # Tabla larga original de métricas.
        "metrics_df": get_symbol_model_comparison_metrics(symbol),
        # Tabla comparativa pivotada benchmark vs ML.
        "pivot_df": get_symbol_model_comparison_pivot(symbol),
        # Resumen agregado por métrica.
        "summary_df": build_symbol_model_comparison_summary(symbol),
        # Panel detallado/temporal de comparación.
        "panel_df": get_symbol_model_comparison_panel(symbol),
    }
