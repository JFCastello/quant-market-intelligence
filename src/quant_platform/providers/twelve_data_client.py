from __future__ import annotations

from datetime import datetime  #-Sirve para la trazabilidad mínima que pide el SDD, que de hecho al hablar de 
                               #  normalización menciona explícitamente columnas como provider e ingested_at
                               #  i,e que toca poner cuándo ingirió el sistema esos datos y de quien-donde, <sanchez>-

from datetime import timezone  #-adicion 100% mia, toco porque datetime.utcnow() solo devuleve la fecha naive i,e sin 
                               #  UTC explicito i,e no aware... por eso se usara datetime.now(timezone.utc), <sanchez>-

from typing import Any  #-Se usa para anotar diccionarios tipo dict[str, Any] porque la respuesta JSON del proveedor 
                        #  puede contener strings, listas, diccionarios, etc, <sanchez>-

import pandas as pd  #-porque la respuesta JSON del proveedor puede contener strings, listas, diccionarios, etc, <sanchez>-

import requests      #-Es la librería HTTP que realmente habla con la API externa, <sanchez>-

from quant_platform.services.settings import load_settings  #-pa carga nuestro loader centralizado de settings
                                                            #  encargado de: configs/ + .env, <sanchez>-


class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com"  #-todas las llamadas del cliente parten de esa URL base, <sanchez>-

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        settings = load_settings()                                       #-carga el contexto de funcionamiento, <sanchez>-

        self.api_key = api_key or settings["env"]["twelve_data_api_key"] #-si por argumento no se pasa la key al crea 
                                                                         #  el obj, entonces mira en el diccionario 
                                                                         #  settings, posiblemente se coloca
                                                                         #  porque da dos modos de uso:
                                                                         #  modo normal del proyecto: credencial desde 
                                                                         #                            config
                                                                         #  modo manual o test: credencial pasada 
                                                                         #                      directamente, <sanchez>-

        self.timeout = timeout   #-Timeout máximo de espera para cada request HTTP al proveedor
                                 #  sirve para evitar que el cliente quede bloqueado indefinidamente
                                 #  si la API responde muy lento o la conexión se cuelga.
                                 #
                                 #  En este caso, por defecto tenemos un tiempo de espera de 30 segundos, 
                                 #  esto signfica un
                                 #  "No se quede esperando indefinidamente, si la API tarda demasiado en responder o la 
                                 #   conexión queda colgada, por t > a 30 s, entonces la libreria requests lanza un 
                                 #   error de timeout y el programa falla en vez de quedarse congelado"
                                 # 
                                 #  Ojo: para un novato, esto NO debe entenderse como
                                 #  "toda la descarga completa 'todo el request' siempre debe terminar en 30 segundos".
                                 #  En requests, el timeout se usa como protección frente a esperas
                                 #  excesivas de red / respuesta del servidor; si el servidor no responde
                                 #  a tiempo, se corta la espera y se lanza una excepción.
                                 # 
                                 #  En todo caso, mire que se expone como argumento en el constructor por lo que permite 
                                 #  ajustar fácilmente el tiempo de espera al crear el cliente, por ejemplo:
                                 #  TwelveDataClient(timeout=10) o TwelveDataClient(timeout=60)
                                 # 
                                 #  Eso es útil porque:
                                 #  - en desarrollo local quizá quieres fallar rápido
                                 #  - en una red lenta quizá quieres dar más margen, <sanchez>-

        if not self.api_key:   #-si not None i,e None por arg construc y None en .env.... good porque mejor explotar al 
            raise ValueError(  #  comienzo con un mensaje claro que dejar que falle después en una request oscura
                               #  .... pille que solo estamos tratando el caso en que es None, no en que es invalida 
                               #  per se, <sanchez>-
                "Missing Twelve Data API key. Set TWELVE_DATA_API_KEY in your local .env file."
            )  #-good mensaje, de acuerdo al flujo real del proyecto pues manda a la persona a .env que es donde se 
               #  encarga y por ello se debe modificar esto.... lo manda a donde es, <sanchez>-


    #-metodo privado
    #  recuerde que al incio de las cosas los _ → pieza auxiliar / interna / pensada para implementación interna
    #  , <sanchez>-
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:  
        """
        Método interno para hacer requests GET a la API de Twelve Data de forma uniforme.

        ¿Por qué existe este método?
        ----------------------------
        Porque no queremos repetir en muchos lugares la misma lógica de (i,e centraliza):
        - construir la URL final,
        - agregar la API key-autenticación con API key,
        - enviar la request,
        - validar errores HTTP,
        - interpretar errores que vengan en el JSON del proveedor-validación de errores reportados por el propio 
          proveedor

        La idea es centralizar toda esa responsabilidad aquí, y que los demás métodos del cliente (por ejemplo 
        get_time_series_raw) solo tengan que preocuparse por pasar el endpoint correcto y sus parámetros.
        """
        url = f"{self.BASE_URL}/{endpoint}"          #-Arma la URL final, si por ejm endpoint = "time_series", queda:
                                                     #  https://api.twelvedata.com/time_series, <sanchez>-

        params = {**params, "apikey": self.api_key}  #-Hace una copia del diccionario de parámetros y le añade la clave
                                                     # API = implementacion autenticación por query param
                                                     # ¿autenticacion con quien? -> con key ante api xD con un query
                                                     # parametrizado o que le inyectamos eso en el param O_O, <sanchez>-

        response = requests.get(url, params=params, timeout=self.timeout)  #-hace la llamada HTTP real-como tal
                                                                           #  params=params: requests convierte este 
                                                                           #  diccionario en query params de la URL, por 
                                                                           #  ejemplo 
                                                                           #  ?symbol=SPY&interval=1day&apikey=..., <sanchez>-
                                                                           
        response.raise_for_status()  #-Esto es importante, por ejm si la respuesta HTTP es algo como:
                                     #  400 ó 401 ó 403 ó 500
                                     #  entonces requests lanza una excepción inmediatamente..... con esto de 
                                     #  aca, cabe notar, se manejan errores a dos niveles:
                                     #    errores HTTP ^ errores “de negocio” reportados por Twelve Data en JSON, <sanchez>-

        payload = response.json()  #-Convierte la respuesta a diccionario Python, <sanchez>-

        #-Algunas APIs pueden responder con status code de 200-299 (por lo que no pasa nada-.raise_for_status no lanza 
        #  nada), en nuestro caso, digamos responde con HTTP 200 (o sea, "la request llegó"), pero aun así reportar un 
        #  error en el contenido JSON.
        #  
        #  Por eso aquí (en este pedazo siguiente de codigo) hacemos una segunda validación: no solo revisamos 
        #  la capa HTTP, sino también la capa lógica del proveedor
        #  En resumen, tenemos:
        #  Validación en dos niveles:
        #   1. HTTP (raise_for_status)
        #   2. Lógica del proveedor (campo "status" en el JSON), <sanchez>-
        if "status" in payload and payload["status"] == "error":  

            code = payload.get("code", "unknown") #-el codigo en general, puede ser, de manera no necesaria, por ejm 
                                                  #  200 o 202 o en general 2xx i,e un status code 
                                                  #  pero pille que nunca sera un 4xx (ejm 404), 5xx (ejm 500)
                                                  #  pues si fuese el caso 2-3 intrucciones atras se caia, <sanchez>-

            message = payload.get("message", "Unknown Twelve Data error")

            raise RuntimeError(f"Twelve Data error [{code}]: {message}")

        return payload  #-devuelve payload ya parseado como dict
                        #  colorario: háblame con Twelve Data de forma uniforme, con auth, timeout y errores claros, <sanchez>-


    def get_time_series_raw(
        self,
        symbol: str,                    #-ticker o instrumento a consultar, por ejemplo "SPY" o "TLT", <sanchez>-
        interval: str = "1day",         #-eso quiere decir que nuestras barritas son de 1 dia por defecto, <sanchez>-
        start_date: str | None = None,  #-pa sacar de una ventana
        end_date: str | None = None,    #  temporal especifica entonces usted da la fechas, <sanchez>-
        outputsize: int = 5000,         #-eso quiere decir que vamos a tener max 5000 dias por sacada-request, <sanchez>-
    ) -> dict[str, Any]:
        """
        Descarga el payload crudo del endpoint `time_series` de Twelve Data.

        Este método es una envoltura específica del endpoint de series temporales
        del proveedor. Su responsabilidad es construir la consulta (con el _get anterior auth y da errores) y devolver 
        la respuesta JSON tal como viene de la API, sin convertirla todavía al esquema
        interno del proyecto.

        Se usa para inspección, trazabilidad del raw payload y como insumo del
        proceso posterior de normalización.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "format": "JSON",
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._get("time_series", params)  #-recuerde que nuestro 
                                                 #  Dominio temporal de observación: días
                                                 #  Frecuencia de actualización: diaria
                                                 #  Objetivo min: forecast de volatilidad futura a 5 días, <sanchez>-


    def get_metadata(self, symbol: str, interval: str = "1day") -> dict[str, Any]:
        
        payload = self.get_time_series_raw(symbol=symbol, interval=interval, outputsize=2) #-pedimos una pequeña 
                                                                                        #  muestra del endpoint y extrae 
                                                                                        #  el bloque "meta"
                                                                                    #  pues para metadatos no 
                                                                                    #  necesitamos bajar toda la serie
                                                                                    #  i,e hacemos una optimización 
                                                                                    #  chiquita ahi pero sensata, <sanchez>-

        meta = payload.get("meta", {})  #-Busca la sección de metadatos del JSON, si no existe, devuelve {}, <sanchez>-

        if not meta:                                                        #-si no {} = en python a si no False i,e si 
            raise RuntimeError(f"No metadata returned for symbol={symbol}") #  hubieron metadatos, lanza ese error, <sanchez>-

        return meta  #-Esto probablmente servira para recuperar info del símbolo desde el proveedor, por ejemplo:
                     #     símbolo resuelto, intervalo, exchange, timezone, tipo de instrumento 
                     #  según lo que el proveedor devuelva, <sanchez>-
    

    def provider_daily_bars_df_from_payload(  #-saca las dailybars del provider desde el payload y las pone en df, <sanchez>-
        self,
        payload: dict[str, Any],              #-se hace de ante mano la requests GET a la API de Twelve Data de forma 
                                              #  uniforme, con auth, timeout y errores claros via el envoltorio del 
                                              #  endpoint `time_series` de Twelve Data, de modo que traemos el JSON 
                                              #  crudo via el _get envuelto por el .get_time_series_raw()..... y pues 
                                              #  se pasa como argumento aca pa aca no hacer la request obvio, <sanchez>-
        fallback_symbol: str | None = None,   #-pone el simbolo con que hizo la request, <sanchez>-
    ) -> pd.DataFrame:

        values = payload.get("values", [])  #-contiene las filas OHLCV, <sanchez>-
        meta = payload.get("meta", {})      #-xd, entonces pa que el metodo anterior?... bueno igual ahi queda 
                                            #  disponible quien sabe pa que.... es que igual no tenia sentido el
                                            #  anterior habiendo traido estos de aca no ?, <sanchez>-

        if not values:
            raise RuntimeError(f"No OHLCV rows returned in payload for symbol={symbol}")

        df = pd.DataFrame(values).copy()    #-ese copy creo sobre, pero bueno, hiper mega precabido pa no hacer cambios 
                                            #  en los datos hiper mega orginales por algun in_place, <sanchez>-

        rename_map = {                      #-pille que aca lo que 
            "datetime": "date",             #  estamos haciendo, es 
        }                                   #  simplemente renombrar el nombre de
        df = df.rename(columns=rename_map)  #  una columna-campo de los datos obtenidos, <sanchez>-

        numeric_cols = ["open", "high", "low", "close", "volume"]  #-pille que aca simplemente convierte columnas a 
        for col in numeric_cols:                                   #  columnas numéricas i,e nos aseguramos que lo 
            if col in df.columns:                                  #  numerico este en formato numerico (que si esta en 
                df[col] = pd.to_numeric(df[col], errors="coerce")  #  str, usual en APIS, quede en reales) y el 
                                                                   #  errors="coerce" significa: si algo no se puede 
                                                                   #  convertir, entonces pandas lo vuelve NaN, <sanchez>-

        # En la capa provider mantenemos date como Python date para no pelear
        # con el contrato actual del dominio; luego la capa normalized final
        # puede convertirlo a pandas datetime si hace falta para parquet/tablas.
        df["date"] = pd.to_datetime(df["date"]).dt.date                     #-aca simplmente convierte-asegura el 
        df = df.sort_values("date", ascending=True).reset_index(drop=True)  #  formato a fecha y ordena ascendente
                                                                            #  again probablemente str a date
                                                                            #  y el orden es importante porque muchos 
                                                                            #  proveedores devuelven series del día más 
                                                                            #  reciente al más antiguo entonces toca
                                                                            #  asegurar el pasado → presente, <sanchez>-

        df["symbol"] = meta.get("symbol", fallback_symbol)  #-creamos ese campo symbol pa datar el ticker o instrumento 
                                                            #  que se consulto.... si no hay symbol en metadatos, pone 
                                                            #  el symbol que le pasamos a la función, <sanchez>-
        df["provider"] = "twelve_data"                      #-creamos ese campo pa dejar trazabilidad del origen de datos, <sanchez>-
        #df["ingested_at"] = datetime.utcnow()              #-creamos ese campo pa marca cuándo el sistema los descargó
                                                            #  i,e representar cuándo ingirió el sistema esos datos
                                                            #  i,e mas trazabilidad, <sanchez>-
        df["ingested_at"] = datetime.now(timezone.utc)      #-toco porque datetime.utcnow() daba la naive sin UTC, aca
                                                            #  con esto tenemos la aware i,e con UTC, en particular, con 
                                                            #  UTC+0 i,e hora global estandar, <sanchez>-

        expected_cols = [                                            #-y  aca simplmente lo que estamos haciendo
            "date",                                                  #  es definir esta lista de columnas-campos
            "open",                                                  #  esperados, de tal manera que a continuacion
            "high",                                                  #  miramos con un for list comprehension, que no 
            "low",                                                   #  no falte ninguna de las que pensamos necesarias
            "close",                                                 #  para poder hacer las cosas
            "volume",                                                #  ....
            "symbol",                                                #  Es por asi decirlo, 
            "provider",                                              #  una especie de contrato-esquema mínimo local del 
            "ingested_at",                                           #  DataFrame normalizado o que se va a normalizar
        ]                                                            #  ...
                                                                     #  El caso es que validamos entonces que no falten 
        missing = [c for c in expected_cols if c not in df.columns]  #  columnas en el df
        if missing:                                                  #  y si missing = [] entonces tendriamos if [] 
                                                                     #  que equivale a if False i,e si missing list 
            raise RuntimeError(f"Standardized OHLCV DataFrame from provider missing columns: {missing}")  #  no vacia 
                                                                     #  i,e algo falta, entonces se lanza este error de 
                                                                     #  que nos faltan cositas casi que vitales xD, toca
                                                                     #  ver que y porque y con esto se muestra cual, <sanchez>-

        return df[expected_cols]  #-Devuelve solo el subconjunto estándar i,e forzamos a que el resultado final tenga 
                                  #  solo esas columnas y en ese orden, <sanchez>-


    #-2do a leer: debido al nuevo orquestador scripts/ingest_market_data.py (orquestador serio del pipeline), este metodo 
    #  queda relegado a ser un wrapper conveniente-util (pierde su protagonismo como orquetador del pipeline).... 
    #  usable mas que todo para herramienta para pruebas rápidas, uso en notebooks, o smoke tests manuales.... 
    #  pruebas rapidas de si todo funciona al combinar por ejm.... por ello, no lo llamaría “fósil inútil”, pero
    #  ya no es el camino principal del pipeline formal....... razonaria yo al incio de este cambiode arquitectura:
    #   "ese get_provider_daily_bars_df(...) se medio cayo al meter lo de scripts/ingest_market_data.py porque 
    #    basicamente el lo que intentaba era unir todas las partes construidas en twelve_data_client.py pero al quedar 
    #    el nuevo unidor-orquetador pues xD no? ya actualemnte usted dice que dejarlo como wrapper 
    #    covniente............ pues bueno, creo que no lo vamos a usar pa nada pues rompo la logica de pipeline, pero 
    #    bueno, lo podemos dejar como fosil xD...... un fosil igual muy educativo en el sentido de que acabo de exponer 
    #    pues (porque pille que pa testeo tmapoco srive, pues la buena practica es mirar primero partes y luego la cosa 
    #    complea...... pero eso, bueno, si,, puede servir como medio testeo no ? como "testeo de orquestaje via 
    #    antecesor orquetador-orquestador mini")"
    #  el caso es que mire que esto simplmente es lo mismo qe el original pero mas troceado, troceado en dos procesos
    #  el fetch y luego la tranformacion:, <sanchez>-
    def get_provider_daily_bars_df(   
        self,
        symbol: str,
        interval: str = "1day",
        start_date: str | None = None,
        end_date: str | None = None,
        outputsize: int = 5000,
    ) -> pd.DataFrame:
        payload = self.get_time_series_raw(                #-fetch raw-payload, <sanchez>-
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            outputsize=outputsize,
        )

        return self.provider_daily_bars_df_from_payload(   #-transform provider payload -> provider-normalized df, <sanchez>-
            payload=payload,
            fallback_symbol=symbol, #-pille que pasamos como argumento el symbolo que se usa para el request, <sanchez>-
        )


#    #-1ro leer: este es el equivalente arquelogico-predesesor del de arriba antes de que el orquestaje estuviese 
#    #  delegado a scripts/ingest_market_data.py.... este metodo orquestaba request y organizado de los datos en DF y el 
#    #  return del DF i,e era central-tenia un papel principal
#    #  pero de esta manera la ingest se compliacaba pues no se podia guardar los raw y los normalizados
#    #  de modo que en ingest_market_data.py tocaba hacer doble request llamando este metodo y 
#    #  get_time_series_raw......... entonces para tener los procesos bien divididos y definidos y posiblemente
#    #  usables desde otro lado i,e por un orquestador i,e para conservar la logica de-arquitectira de:
#    #   - la lógica de transformar un payload de Twelve Data a provider-normalized df pertenece al provider adapter
#    #   - la lógica de leer config, iterar universo, guardar raw, guardar parquet, loggear pertenece al script
#    #  
#    #  ...
#    #  me acabo de dar cuenta que una sencilla era que esta funcion devolviera en el return tambien el raw....
#    #  pero bueno, semanticamente es mas apropiado que no, pero eso, <sanchez>-
#    def get_provider_daily_bars_df(#get_daily_bars_df(
#        self,
#        symbol: str,
#        interval: str = "1day",
#        start_date: str | None = None,
#        end_date: str | None = None,
#        outputsize: int = 5000,
#    ) -> pd.DataFrame:
#
#        payload = self.get_time_series_raw(
#            symbol=symbol,
#            interval=interval,
#            start_date=start_date,
#            end_date=end_date,
#            outputsize=outputsize,
#        )  #-se hace la requests GET a la API de Twelve Data de forma uniforme, con auth, timeout y errores claros 
#           #  via el envoltorio del endpoint `time_series` de Twelve Data, de modo que traemos el JSON crudo, <sanchez>-
#
#        values = payload.get("values", [])  #-contiene las filas OHLCV, <sanchez>-
#        meta = payload.get("meta", {})      #-xd, entonces pa que el metodo anterior?... bueno igual ahi queda 
#                                            #  disponible quien sabe pa que.... es que igual no tenia sentido el anterior
#                                            #  habiendo traido estos de aca no ?, <sanchez>-
#
#        if not values:
#            raise RuntimeError(f"No OHLCV rows returned for symbol={symbol}")
#
#        df = pd.DataFrame(values).copy()    #-ese copy creo sobre, pero bueno, hiper mega precabido pa no hacer cambios 
#                                            #  en los datos hiper mega orginales por algun in_place, <sanchez>-
#
#        rename_map = {                      #-pille que aca lo que 
#            "datetime": "date",             #  estamos haciendo, es 
#        }                                   #  simplemente renombrar el nombre de
#        df = df.rename(columns=rename_map)  #  una columna-campo de los datos obtenidos, <sanchez>-
#
#        numeric_cols = ["open", "high", "low", "close", "volume"]  #-pille que aca simplemente convierte columnas a 
#        for col in numeric_cols:                                   #  columnas numéricas i,e nos aseguramos que lo 
#            if col in df.columns:                                  #  numerico este en formato numerico (que si esta en 
#                df[col] = pd.to_numeric(df[col], errors="coerce")  #  str, usual en APIS, quede en reales) y el 
#                                                                   #  errors="coerce" significa: si algo no se puede 
#                                                                   #  convertir, entonces pandas lo vuelve NaN, <sanchez>-
#
#        df["date"] = pd.to_datetime(df["date"]).dt.date                     #-aca simplmente convierte-asegura el 
#        df = df.sort_values("date", ascending=True).reset_index(drop=True)  #  formato a fecha y ordena ascendente
#                                                                            #  again probablemente str a date
#                                                                            #  y el orden es importante porque muchos 
#                                                                            #  proveedores devuelven series del día más 
#                                                                            #  reciente al más antiguo entonces toca
#                                                                            #  asegurar el pasado → presente, <sanchez>-
#
#        df["symbol"] = meta.get("symbol", symbol)       #-creamos ese campo symbol pa datar el ticker o instrumento 
#                                                        #  que se consulto.... si no hay symbol en metadatos, pone 
#                                                        #  el symbol que le pasamos a la función, <sanchez>-
#        df["provider"] = "twelve_data"                  #-creamos ese campo pa dejar trazabilidad del origen de datos, <sanchez>-
#        #df["ingested_at"] = datetime.utcnow()          #-creamos ese campo pa marca cuándo el sistema los descargó
#                                                        #  i,e representar cuándo ingirió el sistema esos datos
#                                                        #  i,e mas trazabilidad, <sanchez>-
#        df["ingested_at"] = datetime.now(timezone.utc)  #-toco porque datetime.utcnow() daba la naive sin UTC, aca
#                                                        #  con esto tenemos la aware i,e con UTC, en particular, con 
#                                                        #  UTC+0 i,e hora global estandar, <sanchez>-
#
#        expected_cols = [                                            #-y  aca simplmente lo que estamos haciendo
#            "date",                                                  #  es definir esta lista de columnas-campos
#            "open",                                                  #  esperados, de tal manera que a continuacion
#            "high",                                                  #  miramos con un for list comprehension, que no 
#            "low",                                                   #  no falte ninguna de las que pensamos necesarias
#            "close",                                                 #  para poder hacer las cosas
#            "volume",                                                #  ....
#            "symbol",                                                #  Es por asi decirlo, 
#            "provider",                                              #  una especie de contrato-esquema mínimo local del 
#            "ingested_at",                                           #  DataFrame normalizado o que se va a normalizar
#        ]                                                            #  ...
#                                                                     #  El caso es que validamos entonces que no falten 
#        missing = [c for c in expected_cols if c not in df.columns]  #  columnas en el df
#        if missing:                                                  #  y si missing = [] entonces tendriamos if [] 
#                                                                     #  que equivale a if False i,e si missing list 
#            raise RuntimeError(f"Standardized OHLCV DataFrame missing columns: {missing}")  #  no vacia i,e algo falta,
#                                                                     #  entonces se lanza este error de que nos faltan
#                                                                     #  cositas casi que vitales xD, toca ver que
#                                                                     #  y porque y con esto se muestra cual, <sanchez>-
#
#        return df[expected_cols]  #-Devuelve solo el subconjunto estándar i,e forzamos a que el resultado final tenga 
#                                  #  solo esas columnas y en ese orden, <sanchez>-

