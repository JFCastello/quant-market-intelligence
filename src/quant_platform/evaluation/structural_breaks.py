from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import ruptures as rpt


@dataclass(frozen=True)
class StructuralBreakArtifacts:
    """
    Contenedor inmutable de los artefactos generados por el pipeline
    de detección de quiebres estructurales.

    Attributes
    ----------
    signal_df : pd.DataFrame
        DataFrame ya preparado y ordenado temporalmente, que contiene las señales
        usadas para la detección de quiebres. Dependiendo de la configuración,
        puede incluir columnas originales y también columnas estandarizadas
        (`z_*`).

    events_df : pd.DataFrame
        Tabla de eventos de quiebre detectados. Cada fila representa un cambio
        estructural identificado en la serie/señal analizada, junto con metadatos
        de trazabilidad y segmentación temporal.
    """
    signal_df: pd.DataFrame
    events_df: pd.DataFrame


# ---------------------------------------------------------------------
# Columnas base mínimas esperadas en las entradas.
#
# Este módulo trabaja a partir de dos fuentes:
# - features_df
# - targets_df
#
# Ambas deben compartir como mínimo:
# - instrument_id
# - date
#
# A partir de ahí, según la configuración, se exigirán columnas adicionales
# para construir la señal de detección.
# ---------------------------------------------------------------------

REQUIRED_FEATURE_BASE_COLS = {
    "instrument_id",
    "date",
}

REQUIRED_TARGET_BASE_COLS = {
    "instrument_id",
    "date",
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
        Conjunto de nombres de columnas que deben existir.
    df_name : str
        Nombre lógico del DataFrame, usado para construir mensajes de error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Si falta al menos una columna requerida.

    Notes
    -----
    Esta validación permite fallar temprano cuando el contrato de entrada
    no se cumple, evitando errores más ambiguos en etapas posteriores
    como merges, estandarización o detección de quiebres.
    """
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _infer_single_instrument_id(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
) -> str:
    """
    Infiera el único `instrument_id` compartido entre `features_df` y `targets_df`.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame de features.
    targets_df : pd.DataFrame
        DataFrame de targets.

    Returns
    -------
    str
        El único `instrument_id` presente y consistente en ambas tablas.

    Raises
    ------
    ValueError
        Si alguna de las tablas contiene cero o múltiples `instrument_id`,
        o si ambos DataFrames contienen IDs distintos.

    Notes
    -----
    El diseño de este módulo asume una detección de quiebres por instrumento.
    Por eso exige que cada corrida procese exactamente un activo/instrumento.
    """
    feature_ids = sorted(features_df["instrument_id"].dropna().unique().tolist())
    target_ids = sorted(targets_df["instrument_id"].dropna().unique().tolist())

    if len(feature_ids) != 1:
        raise ValueError(
            f"Expected exactly one instrument_id in features_df, found: {feature_ids}"
        )
    if len(target_ids) != 1:
        raise ValueError(
            f"Expected exactly one instrument_id in targets_df, found: {target_ids}"
        )
    if feature_ids[0] != target_ids[0]:
        raise ValueError(
            f"features_df and targets_df instrument_id mismatch: {feature_ids[0]} vs {target_ids[0]}"
        )

    return str(feature_ids[0])


def _infer_symbol_from_instrument_id(instrument_id: str) -> str:
    """
    Deriva un símbolo legible a partir de `instrument_id`.

    Parameters
    ----------
    instrument_id : str
        Identificador interno del instrumento, por ejemplo `spy_us`.

    Returns
    -------
    str
        Símbolo derivado, en mayúsculas. Por ejemplo:
        `spy_us` -> `SPY`

    Notes
    -----
    La lógica actual toma la parte anterior al primer guion bajo `_`.
    Es una convención simple y útil para reporting, aunque depende del formato
    del identificador interno.
    """
    return instrument_id.split("_")[0].upper()


def _zscore_series(series: pd.Series) -> pd.Series:
    """
    Estandariza una serie usando z-score.

    Parameters
    ----------
    series : pd.Series
        Serie numérica a estandarizar.

    Returns
    -------
    pd.Series
        Serie transformada como:

            (x - media) / desviación_estándar

    Raises
    ------
    ValueError
        Si la serie es constante y por tanto su desviación estándar es cero.

    Notes
    -----
    Se usa `ddof=0`, es decir, desviación estándar poblacional.
    Esta transformación resulta útil cuando se combinan señales con escalas
    distintas dentro de una señal conjunta multivariada.
    """
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if std == 0.0:
        raise ValueError(f"Cannot standardize constant signal column: {series.name}")
    mean = float(values.mean())
    return (values - mean) / std


def _prepare_signal_df(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    return_signal_column: str,
    volatility_signal_column: str,
    dropna: bool,
    min_required_rows: int,
    standardize_joint_signal: bool,
    use_joint_signal: bool,
    joint_signal_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Construye y prepara el DataFrame de señal que se usará para detectar
    quiebres estructurales.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame de features.
    targets_df : pd.DataFrame
        DataFrame de targets.
    return_signal_column : str
        Nombre de la columna de señal de retornos, por ejemplo `log_ret_1d`.
    volatility_signal_column : str
        Nombre de la columna de señal de volatilidad, por ejemplo `future_rv_5d`.
    dropna : bool
        Indica si se deben eliminar filas con nulos en las columnas de señal.
    min_required_rows : int
        Número mínimo de filas requerido después del preprocesamiento.
    standardize_joint_signal : bool
        Indica si las señales seleccionadas deben ser estandarizadas mediante z-score.
    use_joint_signal : bool
        Si es `True`, usa varias columnas de señal; si es `False`, usa sólo la
        señal de retornos.
    joint_signal_columns : Sequence[str]
        Lista/tupla de columnas que componen la señal conjunta.

    Returns
    -------
    pd.DataFrame
        DataFrame fusionado, ordenado y listo para ser convertido en matriz
        numérica para el algoritmo de detección.

    Workflow
    --------
    1. Valida columnas requeridas en features y targets.
    2. Hace merge one-to-one por `instrument_id` y `date`.
    3. Ordena por fecha.
    4. Determina qué columnas forman la señal.
    5. Opcionalmente elimina filas con nulos en esas señales.
    6. Verifica que queden suficientes filas.
    7. Convierte `date` a datetime.
    8. Opcionalmente crea columnas estandarizadas `z_*`.

    Notes
    -----
    Esta función no detecta quiebres todavía; prepara la base limpia y coherente
    sobre la que operará el algoritmo.
    """
    required_feature_cols = REQUIRED_FEATURE_BASE_COLS | {return_signal_column}
    required_target_cols = REQUIRED_TARGET_BASE_COLS | {volatility_signal_column}

    _validate_required_columns(features_df, required_feature_cols, "features_df")
    _validate_required_columns(targets_df, required_target_cols, "targets_df")

    # Se hace un merge exacto por instrumento y fecha para alinear
    # la señal de retornos con la de volatilidad.
    merged_df = features_df[
        ["instrument_id", "date", return_signal_column]
    ].merge(
        targets_df[["instrument_id", "date", volatility_signal_column]],
        on=["instrument_id", "date"],
        how="inner",
        validate="one_to_one",
    )

    merged_df = merged_df.sort_values("date", ascending=True).reset_index(drop=True)

    # Si use_joint_signal=True se usan varias señales; si no, sólo la de retornos.
    signal_cols = list(joint_signal_columns) if use_joint_signal else [return_signal_column]
    missing_signal_cols = [c for c in signal_cols if c not in merged_df.columns]
    if missing_signal_cols:
        raise ValueError(
            f"Prepared signal dataframe is missing configured signal columns: {missing_signal_cols}"
        )

    if dropna:
        merged_df = merged_df.dropna(subset=signal_cols).reset_index(drop=True)

    if len(merged_df) < min_required_rows:
        raise ValueError(
            f"Signal dataframe has insufficient rows after preprocessing: "
            f"{len(merged_df)} < {min_required_rows}"
        )

    merged_df["date"] = pd.to_datetime(merged_df["date"], errors="raise")

    # Si se solicita estandarización, se crea una columna z-score por cada señal.
    if standardize_joint_signal:
        for col in signal_cols:
            merged_df[f"z_{col}"] = _zscore_series(merged_df[col])

    return merged_df


def _select_signal_matrix_columns(
    signal_df: pd.DataFrame,
    use_joint_signal: bool,
    joint_signal_columns: Sequence[str],
    return_signal_column: str,
    standardize_joint_signal: bool,
) -> list[str]:
    """
    Determina qué columnas del `signal_df` deben convertirse en la matriz
    numérica que consumirá el algoritmo de detección de quiebres.

    Parameters
    ----------
    signal_df : pd.DataFrame
        DataFrame ya preparado.
    use_joint_signal : bool
        Si es `True`, usa múltiples señales; si es `False`, sólo una.
    joint_signal_columns : Sequence[str]
        Columnas base configuradas para la señal conjunta.
    return_signal_column : str
        Columna de retorno a usar cuando no se emplea señal conjunta.
    standardize_joint_signal : bool
        Si es `True`, se usarán las columnas `z_*`; en caso contrario,
        se usarán las columnas originales.

    Returns
    -------
    list[str]
        Lista ordenada de nombres de columnas que formarán la matriz de señal.

    Notes
    -----
    Esta función no usa directamente `signal_df` salvo como contexto conceptual;
    decide sólo los nombres de columnas a seleccionar aguas abajo.
    """
    base_cols = list(joint_signal_columns) if use_joint_signal else [return_signal_column]
    if standardize_joint_signal:
        return [f"z_{col}" for col in base_cols]
    return base_cols


def _detect_breakpoints(
    signal_matrix: np.ndarray,
    algorithm: str,
    cost_model: str,
    penalty: float,
    min_size: int,
    jump: int,
) -> list[int]:
    """
    Ejecuta el algoritmo de detección de quiebres estructurales sobre la señal.

    Parameters
    ----------
    signal_matrix : np.ndarray
        Matriz numérica de señal con forma `(n_rows, n_features)`.
    algorithm : str
        Nombre del algoritmo a usar. En esta versión sólo se soporta `pelt`.
    cost_model : str
        Modelo de costo de `ruptures`, por ejemplo `rbf`.
    penalty : float
        Penalización usada por el algoritmo para decidir el número de quiebres.
    min_size : int
        Tamaño mínimo permitido para un segmento.
    jump : int
        Paso de exploración del algoritmo.

    Returns
    -------
    list[int]
        Lista de breakpoints devueltos por `ruptures`, convertidos a enteros.

    Raises
    ------
    ValueError
        Si se solicita un algoritmo no soportado en esta versión.

    Notes
    -----
    `ruptures` devuelve breakpoints como índices de final de segmento, incluyendo
    normalmente el punto terminal `n_rows`. Este último es importante porque el
    resto del pipeline lo usa para reconstruir segmentos y eventos.
    """
    algorithm_normalized = algorithm.lower()

    if algorithm_normalized != "pelt":
        raise ValueError(f"Unsupported algorithm for v1: {algorithm}")

    model = rpt.Pelt(
        model=cost_model,
        min_size=min_size,
        jump=jump,
    ).fit(signal_matrix)

    breakpoints = model.predict(pen=penalty)
    return [int(x) for x in breakpoints]


def _empty_events_df() -> pd.DataFrame:
    """
    Construye un DataFrame vacío con el esquema esperado para `events_df`.

    Returns
    -------
    pd.DataFrame
        DataFrame vacío con todas las columnas finales del artefacto de eventos.

    Notes
    -----
    Esto permite que el pipeline conserve un contrato estable de salida incluso
    cuando no se detectan quiebres estructurales.
    """
    return pd.DataFrame(
        columns=[
            "event_id",
            "break_version",
            "instrument_id",
            "symbol",
            "breakpoint_index",
            "break_date",
            "previous_segment_start_date",
            "previous_segment_end_date",
            "next_segment_start_date",
            "next_segment_end_date",
            "method",
            "algorithm",
            "cost_model",
            "penalty",
            "min_size",
            "jump",
            "signal_row_count",
            "detected_at",
        ]
    )


def _build_events_df(
    signal_df: pd.DataFrame,
    breakpoints: Sequence[int],
    break_version: str,
    instrument_id: str,
    symbol: str,
    algorithm: str,
    cost_model: str,
    penalty: float,
    min_size: int,
    jump: int,
) -> pd.DataFrame:
    """
    Construye la tabla de eventos de quiebre a partir de la señal y de los
    breakpoints detectados.

    Parameters
    ----------
    signal_df : pd.DataFrame
        DataFrame de señal ya preparado y ordenado por fecha.
    breakpoints : Sequence[int]
        Secuencia de índices de quiebre devueltos por el detector.
    break_version : str
        Versión lógica de la detección de quiebres.
    instrument_id : str
        Identificador interno del instrumento.
    symbol : str
        Símbolo legible del instrumento.
    algorithm : str
        Algoritmo usado.
    cost_model : str
        Modelo de costo usado.
    penalty : float
        Penalización usada en la detección.
    min_size : int
        Tamaño mínimo de segmento.
    jump : int
        Paso de exploración.

    Returns
    -------
    pd.DataFrame
        Tabla de eventos donde cada fila representa un quiebre detectado,
        con fechas de los segmentos anterior y siguiente.

    Raises
    ------
    ValueError
        Si el último breakpoint no coincide con el largo de la señal.

    Notes
    -----
    El detector devuelve límites de segmento. Esta función traduce esos límites
    en eventos de quiebre interpretables, incluyendo:
    - fecha del quiebre,
    - rango temporal del segmento anterior,
    - rango temporal del segmento siguiente,
    - metadatos de trazabilidad.
    """
    n_rows = len(signal_df)
    if n_rows == 0:
        return _empty_events_df()

    # Se antepone 0 como inicio del primer segmento.
    boundaries = [0] + [int(bp) for bp in breakpoints]

    # Se exige que el breakpoint terminal coincida con la longitud de la señal.
    if boundaries[-1] != n_rows:
        raise ValueError(
            f"Expected terminal breakpoint to equal signal length ({n_rows}), got {boundaries[-1]}"
        )

    event_rows: list[dict[str, object]] = []
    detected_at = datetime.now(timezone.utc)

    # Se excluye el breakpoint inicial (0) y el terminal (n_rows),
    # porque los eventos reales son sólo los cortes intermedios.
    for event_number, bp in enumerate(boundaries[1:-1], start=1):
        previous_start_idx = boundaries[event_number - 1]
        previous_end_idx = bp - 1
        next_start_idx = bp
        next_end_idx = boundaries[event_number + 1] - 1

        event_rows.append(
            {
                "event_id": f"{instrument_id}_break_{event_number:03d}",
                "break_version": break_version,
                "instrument_id": instrument_id,
                "symbol": symbol,
                "breakpoint_index": bp,
                "break_date": signal_df.iloc[next_start_idx]["date"],
                "previous_segment_start_date": signal_df.iloc[previous_start_idx]["date"],
                "previous_segment_end_date": signal_df.iloc[previous_end_idx]["date"],
                "next_segment_start_date": signal_df.iloc[next_start_idx]["date"],
                "next_segment_end_date": signal_df.iloc[next_end_idx]["date"],
                "method": f"{algorithm}_{cost_model}",
                "algorithm": algorithm,
                "cost_model": cost_model,
                "penalty": float(penalty),
                "min_size": int(min_size),
                "jump": int(jump),
                "signal_row_count": int(n_rows),
                "detected_at": detected_at,
            }
        )

    if not event_rows:
        return _empty_events_df()

    return pd.DataFrame(event_rows)


def build_structural_break_artifacts(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    break_version: str = "v1",
    return_signal_column: str = "log_ret_1d",
    volatility_signal_column: str = "future_rv_5d",
    use_joint_signal: bool = True,
    joint_signal_columns: Sequence[str] = ("log_ret_1d", "future_rv_5d"),
    algorithm: str = "pelt",
    cost_model: str = "rbf",
    dropna: bool = True,
    min_required_rows: int = 252,
    standardize_joint_signal: bool = True,
    penalty: float = 8.0,
    min_size: int = 20,
    jump: int = 1,
) -> StructuralBreakArtifacts:
    """
    Orquesta la construcción completa de los artefactos de quiebres estructurales.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame de features del instrumento.
    targets_df : pd.DataFrame
        DataFrame de targets del instrumento.
    break_version : str, default="v1"
        Versión lógica del pipeline de quiebres.
    return_signal_column : str, default="log_ret_1d"
        Columna de retornos usada como señal base.
    volatility_signal_column : str, default="future_rv_5d"
        Columna de volatilidad usada como señal complementaria.
    use_joint_signal : bool, default=True
        Si es `True`, usa señal multivariada; si es `False`, usa sólo la señal
        de retornos.
    joint_signal_columns : Sequence[str], default=("log_ret_1d", "future_rv_5d")
        Columnas que forman la señal conjunta.
    algorithm : str, default="pelt"
        Algoritmo de detección. En esta versión sólo se soporta `pelt`.
    cost_model : str, default="rbf"
        Modelo de costo de `ruptures`.
    dropna : bool, default=True
        Si es `True`, elimina filas con nulos en las señales seleccionadas.
    min_required_rows : int, default=252
        Número mínimo de filas requerido para proceder con la detección.
    standardize_joint_signal : bool, default=True
        Si es `True`, estandariza las señales antes de detectar quiebres.
    penalty : float, default=8.0
        Penalización usada por el algoritmo.
    min_size : int, default=20
        Tamaño mínimo de cada segmento.
    jump : int, default=1
        Paso de exploración del algoritmo.

    Returns
    -------
    StructuralBreakArtifacts
        Objeto con:
        - `signal_df`: señal preparada
        - `events_df`: eventos de quiebre detectados

    Workflow
    --------
    1. Valida columnas base mínimas.
    2. Infiera `instrument_id` y `symbol`.
    3. Prepara el DataFrame de señal.
    4. Determina qué columnas forman la matriz numérica de detección.
    5. Ejecuta el detector de quiebres.
    6. Convierte los breakpoints en una tabla interpretable de eventos.
    7. Empaqueta los resultados en un dataclass inmutable.

    Notes
    -----
    Esta es la función principal del módulo. No entrena modelos ni calcula
    targets; toma series ya preparadas y produce una lectura estructural
    del comportamiento temporal del instrumento.
    """
    _validate_required_columns(features_df, REQUIRED_FEATURE_BASE_COLS, "features_df")
    _validate_required_columns(targets_df, REQUIRED_TARGET_BASE_COLS, "targets_df")

    instrument_id = _infer_single_instrument_id(features_df, targets_df)
    symbol = _infer_symbol_from_instrument_id(instrument_id)

    signal_df = _prepare_signal_df(
        features_df=features_df,
        targets_df=targets_df,
        return_signal_column=return_signal_column,
        volatility_signal_column=volatility_signal_column,
        dropna=dropna,
        min_required_rows=min_required_rows,
        standardize_joint_signal=standardize_joint_signal,
        use_joint_signal=use_joint_signal,
        joint_signal_columns=joint_signal_columns,
    )

    signal_matrix_cols = _select_signal_matrix_columns(
        signal_df=signal_df,
        use_joint_signal=use_joint_signal,
        joint_signal_columns=joint_signal_columns,
        return_signal_column=return_signal_column,
        standardize_joint_signal=standardize_joint_signal,
    )

    # Se extrae la matriz numérica final que consumirá `ruptures`.
    signal_matrix = signal_df[signal_matrix_cols].to_numpy(dtype=float)

    breakpoints = _detect_breakpoints(
        signal_matrix=signal_matrix,
        algorithm=algorithm,
        cost_model=cost_model,
        penalty=penalty,
        min_size=min_size,
        jump=jump,
    )

    events_df = _build_events_df(
        signal_df=signal_df,
        breakpoints=breakpoints,
        break_version=break_version,
        instrument_id=instrument_id,
        symbol=symbol,
        algorithm=algorithm,
        cost_model=cost_model,
        penalty=penalty,
        min_size=min_size,
        jump=jump,
    )

    return StructuralBreakArtifacts(
        signal_df=signal_df,
        events_df=events_df,
    )