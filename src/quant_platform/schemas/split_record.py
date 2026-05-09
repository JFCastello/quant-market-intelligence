from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class SplitRecord(BaseModel):
    """
    Modelo Pydantic que representa una partición (split) de datos para entrenamiento,
    validación y prueba de modelos de series temporales.
    
    Almacena las fechas de inicio y fin de cada segmento, así como metadatos
    como la versión del split, identificadores y la fuente de umbrales de régimen.
    """

    # Versión del esquema o proceso de split (ej: "v1", "v2")
    split_version: str

    # Identificador único de este split (ej: "time_based_001")
    split_id: str

    # Identificador del instrumento financiero al que pertenece este split
    instrument_id: str

    # Fecha de inicio del conjunto de entrenamiento
    train_start: date
    # Fecha de fin del conjunto de entrenamiento
    train_end: date

    # Fecha de inicio del conjunto de validación (opcional, puede ser None)
    validation_start: date | None = None
    # Fecha de fin del conjunto de validación (opcional)
    validation_end: date | None = None

    # Fecha de inicio del conjunto de prueba
    test_start: date
    # Fecha de fin del conjunto de prueba
    test_end: date

    # Número de filas/observaciones en el conjunto de entrenamiento
    train_rows: int

    # Número de filas en el conjunto de validación (opcional)
    validation_rows: int | None = None

    # Número de filas en el conjunto de prueba
    test_rows: int

    # Fuente utilizada para calcular los umbrales de régimen (ej: "train_only", "expanding")
    regime_thresholds_source: str