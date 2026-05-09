from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ForecastRecord(BaseModel):
    forecast_id: str
    run_id: str
    instrument_id: str
    as_of_date: date
    horizon: int
    y_pred: float
    regime_pred: str | None = None
    probability_calm: float | None = None
    probability_normal: float | None = None
    probability_stress: float | None = None
