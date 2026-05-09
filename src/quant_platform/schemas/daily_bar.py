from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class DailyBar(BaseModel):
    instrument_id: str
    date: date   #-yo opino que mejor-mas poderoso datetime.... mas en un futuro si intraday, pero ok, <sanchez>-
    open: float
    high: float
    low: float
    close: float
#    adjusted_close: float | None = None   #-En Twelve Data sí hay ajuste, pero no como un campo separado 
                                           #  'adjusted_close' de modo que este no se puede solicitar como tal y por 
                                           #  ello no queda en el payload que se logra en 
                                           #  providers/twelve_data_client.py; Twelve Data dice que para datos 
                                           #  daily, weekly y monthly los precios vienen ajustados por splits; los 
                                           #  datos intraday no vienen ajustados; y para hacer ajustes adicionales 
                                           #  del lado cliente, mencionan usar los endpoints de /splits y /dividends. 
                                           #  De modo que el close diario de Twelve Data ya estaría split-adjusted 
                                           #  (With the current Twelve Data daily payload, prices are split-adjusted 
                                           #  at the series level) i,e el close actual ya viene ajustado por splits en 
                                           #  daily (incorpora cierto ajuste por splits a nivel de serie), pero eso no 
                                           #  implica que tengamos un adjusted_close tipo 
                                           #  Yahoo-style / total-return-style como columna separada (una columna 
                                           #  explicita de cierre ajustado ), y menos aún una serie close ajustado por 
                                           #  dividendos de forma explícita en el payload actual (an explicit 
                                           #  dividend-adjusted total-return close series). Es por eso que cosas como 
                                           #  una serie verdaderamente o totalmente ajustada (fully adjusted series) 
                                           #  i,e que este ajustanda tanto por dividendos como por splits (como minimo) 
                                           #  donde sí tocaría construirla del lado cliente (i,e aca de nuestro lado 
                                           #  donde la recibimos-despues de recibirla info) usando información adicional 
                                           #  (eventos corporativos) como por ejm dividends y splits (additional 
                                           #  corporate actions data like for example, splits/dividends); o obtener los 
                                           #  datos sin ningun ajuste y obtener una columna de ajuste por aparte son 
                                           #  cosas que se dejaran para despues pues posiblemente requieran redesign the 
                                           #  contract around a dedicated 'adjustment pipeline' (un pipeline para el 
                                           #  adjusted_close ya sea como columna aparte y que el close sea el 
                                           #  bruto-crudo o que el close sea split y dividend adjusted y no haya nada
                                           #  en crudo-bruto.... y todo esto quede asi ademas trazable), <sanchez>
    volume: float | None = None
    provider: str
    ingested_at: datetime
