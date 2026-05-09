from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from quant_platform.providers import TwelveDataClient
from quant_platform.schemas.daily_bar import DailyBar      #-en el init de schemas estaba, no habia nececidad, <castello>-
from quant_platform.services.settings import load_settings


#INSTRUMENT_MAP = {     #-mapa nombre instrumento a nombre interno... una especie de "id" interno para los 
#    "SPY": "spy_us",   #  diferentes simbolos, <castello>-
#    "TLT": "tlt_us",
#}

INSTRUMENT_MAP = {
    "SPY": "spy_us",
    "TLT": "tlt_us",
    "GLD": "gld_us",
    "HYG": "hyg_us",
}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)  #-🧠 ¿Qué hace path.mkdir(parents=True, exist_ok=True)?
                                             #     🔍 Desglose
                                             #        .mkdir() es un metodo de instancia de la clase Path de pathlib en
                                             #        Python y sirve para crear carpetas:
                                             #     
                                             #       🧩 mkdir()
                                             #          -> crea un directorio (carpeta)
                                             #       🌳 parents=True
                                             #          -> crea todas las carpetas necesarias en la ruta si no existe
                                             #             (analogo a mkdir -p etc/etc/etc que se hace por consola en 
                                             #              linux)
                                             #       🛑 exist_ok=True
                                             #          -> evita error si la carpeta ya existe
                                             #          Sin esto: FileExistsError ❌
                                             #          Con esto: no pasa nada, sigue normal ✔️, <castello>-

def save_raw_payload(symbol: str, payload: dict, raw_root: Path, start_date: str, end_date: str) -> Path:
    outdir = raw_root / symbol.lower()
    ensure_directory(outdir)  #-nos verifica y certifica que si exista esa direccion de directorio, <castello>-

    outfile = outdir / f"{symbol.lower()}_{start_date}_{end_date}_raw.json"
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return outfile


def adapt_provider_df_to_daily_bar_contract(df: pd.DataFrame, instrument_id: str) -> pd.DataFrame:
    adapted = df.copy()

    adapted["instrument_id"] = instrument_id

    adapted = adapted[
        [
            "instrument_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "provider",
            "ingested_at",
        ]
    ]  #-asegura el orden y bota cosas innecesarias que entregase .get_provider_daily_bars_df(), <castello>-

    # Validacion ligera fila por fila contra el contrato
    records = adapted.to_dict(orient="records")  #-De pandas y clave cuando trabajas con APIs o JSON
    validated = [DailyBar(**record).model_dump() for record in records]
                                                 #  🧠 ¿Qué hace?
    validated_df = pd.DataFrame(validated)
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="raise") #-date -> pandas datetime64[ns], <castello>-
    validated_df["ingested_at"] = pd.to_datetime(validated_df["ingested_at"], errors="raise", utc=True) #-datetime -> pandas datetime64[ns, UTC].... aunque ya los hacia bien automaticamente pandas a diferencia de los dtype date i,e no los dejaba como tipo obj sino como datetime64[ns, UTC], pero igual con esto aseguramos, <castello>-
                                                 #          df.to_dict(orient="records")
                                                 #      👉 Convierte un DataFrame en una lista de diccionarios,
    return validated_df
                                                 #          donde cada fila es un diccionario.
                                                 #          Traduccion:
                                                 #          “vuelva cada fila un diccionario y métalos en una lista”
                                                 #      💥 Ejemplo claro
                                                 #           import pandas as pd
                                                 #           
                                                 #           df = pd.DataFrame({
                                                 #               "nombre": ["Ana", "Luis"],
                                                 #               "edad": [25, 30]
                                                 #           })
                                                 #           
                                                 #           df.to_dict(orient="records")
                                                 #           👉
                                                 #           [
                                                 #               {"nombre": "Ana", "edad": 25},
                                                 #               {"nombre": "Luis", "edad": 30}
                                                 #           ]
                                                 #  🔍 ¿Por qué “records”?
                                                 #       👉 porque cada fila = un “registro” (record)
                                                 #  🔥 Para qué sirve
                                                 #     enviar datos a APIs; convertir a JSON; trabajar con listas en Python
                                                 # 
                                                 #  🎯 Comparación rápida
                                                 #  - orient="records" → lista de dicts (🔥 más útil)
                                                 #                       list[donde cada fila un dict-obj tipo json]
                                                 #  - orient="dict"    → dict de columnas: {key_campos:{indices:registros}}
                                                 #                       columnas con índices como diccionarios
                                                 #  - orient="list"    → dict con listas: {key_campos:[registros]}
                                                 #                       columnas como listas
                                                 #  - orient="index"   → filas con índice como clave: 
                                                 #                       {key_indice:{fila i,e camps:registrs}}
                                                 #                       i,e igual que records pero dict no list 
                                                 #                       cuyas keys seran indices aoutincrement por asi
                                                 #                       decirlo
                                                 #
                                                 #  Colorario, [DailyBar(**record).model_dump() for record in records]: 
                                                 #  pille entonces que vamos a recorrer la lista de records-resgistros
                                                 #  donde cada record-registro en records es un dict, de manera que
                                                 #  cuando hacemos DailyBar(**record) donde si por ejm-suponga-hipotetico no real
                                                 #  record = {"open":100,"close":105} eso se traduciria en un 
                                                 #  desempaquetado **record en la clase como 
                                                 #  DailyBar(open=100,close=105). De modo que:
                                                 #  1. Usa **record para desempaquetar el dict como argumentos del 
                                                 #     modelo DailyBar (equivale a llamar 
                                                 #     DailyBar(campo1=valor1, campo2=valor2, ...))
                                                 #  2. Crea una instancia del modelo (Pydantic), lo que:
                                                 #     - valida los tipos (ej: float, date, datetime, etc.)
                                                 #     - convierte datos automáticamente si es posible
                                                 #     - lanza error si algo no cumple el esquema
                                                 #  3. .model_dump() convierte el objeto validado de vuelta a dict,
                                                 #     ya limpio, consistente y con los tipos correctos -y se guarda 
                                                 #     en lista.... asi fila de registros a filade registros-
                                                 # 
                                                 #  👉 En resumen: toma cada fila, la valida contra el modelo y devuelve
                                                 #     un diccionario "sanitizado" listo para usar (API, DB, etc.)
                                                 #     .... y las guarda en la lista validates, luego esa lista
                                                 #     se transforma en data frame again con DataFrame, <castello>-


def save_normalized_df(symbol: str, df: pd.DataFrame, normalized_root: Path, start_date: str, end_date: str) -> Path:
    outdir = normalized_root / symbol.lower()
    ensure_directory(outdir)

    outfile = outdir / f"{symbol.lower()}_{start_date}_{end_date}_daily_bars.parquet"
    df.to_parquet(outfile, index=False)  #-Note que Parquet es un formato... pero no cualquier formato 🔥
                                         #  🧠 ¿Qué es Parquet?
                                         #    👉 Apache Parquet es un formato de almacenamiento de datos:
                                         #        - columnar (por columnas)
                                         #        - comprimido
                                         #        - optimizado para análisis
                                         #
                                         #      🔍 ¿Qué significa “columnar”?                                         
                                         #          En lugar de guardar por filas (como CSV), guarda los datos por 
                                         #          columnas.... ejemplo conceptual:
                                         #           * Normal (como CSV-por filas):
                                         #               fila 1: nombre, edad, ciudad
                                         #               fila 2: nombre, edad, ciudad
                                         #             👉 guarda por filas
                                         #           * Parquet:
                                         #             columna nombre -> [Ana, Luis]
                                         #             columna edad   -> [25, 30]
                                         #             columna ciudad -> [Medellín, Bogotá]
                                         #             👉 guarda por columnas
                                         #      💥 ¿Por qué eso es brutal?
                                         #           Porque en análisis tú haces cosas como:
                                         #              df["edad"]
                                         #           👉 Parquet solo lee ESA columna
                                         #           👉 no todo el archivo
                                         #           🚀 mucho más rápido
                                         #
                                         #           i,e como permite leer solo las columnas necesarias (es más rápido y 
                                         #           eficiente en análisis)
                                         #
                                         #        🔥 Ventajas:
                                         #           - ⚡ más rápido que CSV
                                         #           - 💾 ocupa menos espacio (compresión)
                                         #           - 📊 ideal para big data
                                         #           - 🧠 mantiene tipos (no como CSV que todo es texto)
                                         #        ⚠️ Desventajas:
                                         #           - ❌ no es legible a ojo (no lo abres como texto)
                                         #           - ❌ necesitas herramientas (pandas, spark, etc.)
                                         #
                                         #      Por ultimo, mencionar que en Python con pandas los comandos mas comunes 
                                         #      asociados son:
                                         #           df.to_parquet("data.parquet")
                                         #           df = pd.read_parquet("data.parquet")
                                         #  🎯 Resumen:
                                         #  👉 Parquet = “CSV pero pro, rápido y optimizado pa data grande”
                                         #              = formato optimizado para análisis de datos grandes, <castello>-

    return outfile


def main() -> None:
    settings = load_settings()

    symbols = settings["data"]["universe"]
    raw_root = Path(settings["paths"]["raw_path"])  #-vuleve la ruta una instancia de la clase Path de pathlib, <castello>-
    normalized_root = Path(settings["paths"]["normalized_path"])  #-same, <castello>-
    interval = settings["data"]["interval"]

    #start_date = "2026-02-01"
    #end_date = "2026-03-01"

    start_date = "2018-01-01"
    end_date = "2026-04-05"    #-mas adelante -> datetime.now(timezone.utc).date().isoformat() pero toca poner el UTC de NY pues nos estamos movinendo en esa bolsa-calendario, <sanchez>-

    client = TwelveDataClient()

    for symbol in symbols:
        instrument_id = INSTRUMENT_MAP[symbol]

        payload = client.get_time_series_raw(               #-ine__, <castello>-
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            #outputsize=200,
            outputsize=5000,
        )

        raw_file = save_raw_payload(
            symbol=symbol,
            payload=payload,
            raw_root=raw_root,
            start_date=start_date,
            end_date=end_date,
        )                               #-como .json, <castello>-

        provider_df = client.provider_daily_bars_df_from_payload(  #-solved ine__ficiencia, <castello>-
            payload=payload,
            fallback_symbol=symbol,
        )

        #provider_df = client.get_provider_daily_bars_df(   #-__ficiente.... dos request de lo mismo, <castello>-
        #    symbol=symbol,
        #    interval=interval,
        #    start_date=start_date,
        #    end_date=end_date,
        #    outputsize=200,
        #)

        daily_bar_df = adapt_provider_df_to_daily_bar_contract(
            df=provider_df,
            instrument_id=instrument_id,
        )

        normalized_file = save_normalized_df(
            symbol=symbol,
            df=daily_bar_df,
            normalized_root=normalized_root,
            start_date=start_date,
            end_date=end_date,
        )                               #-como .parquet, <castello>-

        print(f"symbol={symbol}")       #-note que nunca usamos el 'symbol' de providers/twelve_data_client.py
                                        #  sino el de configs/base.yaml.... queda solo guardado en
                                        #  el .json del raw, <castello>-
        print(f"instrument_id={instrument_id}")
        print(f"raw_file={raw_file}")
        print(f"normalized_file={normalized_file}")
        print(daily_bar_df.head(3))
        print(daily_bar_df.dtypes)
        print(daily_bar_df.shape)
        print("-" * 60)


if __name__ == "__main__":
    main()
