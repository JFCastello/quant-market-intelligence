from __future__ import annotations

from pydantic import BaseModel


class Instrument(BaseModel):
    instrument_id: str
    symbol: str
    asset_class: str
    exchange: str
    currency: str
    timezone: str
    is_active: bool = True
