from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class TargetRow(BaseModel):
    instrument_id: str
    date: date
    target_version: str
    future_rv_5d: float | None = None
    future_regime_5d: str | None = None
