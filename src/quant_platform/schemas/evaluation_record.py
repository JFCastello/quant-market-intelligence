from __future__ import annotations

from pydantic import BaseModel


class EvaluationRecord(BaseModel):
    run_id: str
    split_id: str
    instrument_id: str
    model_name: str
    metric_name: str
    metric_value: float
