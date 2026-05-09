#-objetivo unico en este archivo: optencion de un diccionario con las configuraciones pertinentes
#  i,e centralizacion de las configuraciones via un loader central (por ello) de settings que maneja 
#  los archivos en la carpeta config y en el .env -(o los, aunque no es claro eso de los, mirar critica del final)-
#  .... cabe notar que nuestra filosofia es: existe una config base y luego en un fase 1 o 2 o idk hay cosas 
#  extras o que cambian algunos aspectos de la base, por lo que se "sobre escribe" (las comillas es porque no se 
#  modifican los archviso base ni los otros) y se añade lo extra
#  por sobre lo base, esa idea es la que sigue el diccionario resultante con las configuracion pertienente al conexto,
#  cabe notar que, como vera, soporta casi que un cambio en la totalidad del base (entonces el rol del base
#  es mas que todo para que no se tenga que reescribir todo en los otros sino solo lo nuevo-lo que cambia
#  ...... a priori, de una fase a otra o idk), <sanchez>-

from __future__ import annotations

import os
from pathlib import Path
from typing import Any     #-Se usa para anotar diccionarios tipo dict[str, Any] i,e para dar la flexibilidad
                           #  de que a priori se espera que contenga strings pero tambien puede contener listas, 
                           #  diccionarios, etc..... igual recuerde que esto es para esas vainas de claridad
                           #  de python, de decir que tipo recibe algo y de que tipo devuelve i,e eso es para
                           #  las type hints-annotations de python..... aunque gpt dice que FastAPI como
                           #  que si le para bolas a eso, entonces demas que por eso lo estamos viendo
                           #  so, lets see lets see, lets see if it is as Pydantic que pille que si hace
                           #  cosas, pues tiene efectos en compilado-ejecucion, <sanchez>-

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[3]  #-__file__ DUNDER del modulo del archivo settings.py, que apunta a 
                                                #  la direccion en que esta este archivo settings.py
                                                #  .resolve() obtiene la ruta absoluta real
                                                #  .parents[3] sube desde: 
                                                #    settings.py -> services -> quant_platform -> src -> raiz del repo
                                                #  , <sanchez>-
CONFIG_DIR = ROOT_DIR / "configs"


#-func auxiliar de la ultima, <sanchez>-
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Mezcla dos diccionarios de forma profunda (deep merge).

    Ejemplo:
    - base puede traer toda la config común
    - override puede traer solo lo que cambia para 'local' o 'prod'

    Si una clave existe en ambos y ambos valores son diccionarios,
    se mezclan recursivamente en vez de reemplazarse por completo.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
    
    #-pa entender rapido este codigo, piense que todo esto es para los .yaml, y que estos cumplen con una 
    #  esturctura que puede ser t.q asi:
    #     cosa 1:
    #       etc 1
    #     cosa 2:
    #       etc 1:
    #         etc 1.0
    #         etc 1.1
    #       etc 2
    #     ...[continua]...
    #  por ello, piense que los .yaml los podemos visualizar como diccionarios con llaves y valores, y como diccionarios
    #  de diccionarios en general.
    #
    #  Entonces, note, que lo que se hace, es mirar las llaves y valores de override (el diccionario que en principio
    #  complementa el base) y de ahi separamos el proceso en dos partes:
    #  - mirar llaves que comparte base con override 
    #    y que estan asociadas en base como en override a valores que son diccionarios, en simultaneo
    #    -> se llama recursivamente la funcion
    #  - caso contrario: 
    #    - si override tiene algo nuevo (es decir una llave que no tiene base)
    #      entonces simplememte se añade esa nueva llave y valor 
    #    - si para una llave o caracteristica compartida en los yamls, en uno es diccionario y en el otro no
    #      o viceversa
    #      entonces simplemnete se sobre escribe-reemplaza lo que diga en override sobre lo que diga en base
    #       Ejemplo:
    #        base = {
    #            "logging": {
    #                "level": "INFO",
    #                "format": "short"
    #            }
    #        }
    #        
    #        override = {
    #            "logging": "apagado"
    #        }
    #
    #       resultado:
    #       {
    #           "logging": "apagado"
    #       }
    #    - se realizo recursivamente el proceso hasta escudriñar todo el diccionario y llegar a una llave
    #      compartida que en ambos archivos estan asociadas a uno diccionarios
    #      entonces simplemnete se sobre escribe-reemplaza lo que diga en override sobre lo que diga en base
    #      para esa llave
    #       Ejemplos:
    #        base = {
    #            "app": {
    #                "name": "quant-market-intelligence",
    #                "phase": "phase-1",
    #                "environment": "local"
    #            }
    #        }
    #        
    #        override = {
    #            "app": {
    #                "environment": "prod"
    #            }
    #        }
    #        
    #        resultado:
    #        {
    #            "app": {
    #                "name": "quant-market-intelligence",
    #                "phase": "phase-1",
    #                "environment": "prod"
    #            }
    #        }
    #
    #  entonces eso, la idea es simplemente que estamos rastreando las keys compartidas para sobrescribir
    #  lo de override en el base y sacar un diccionario de resultado (imponer la version de override), lo no compartido 
    #  simplemente se añade, <sanchez>-

#-func auxiliar de la ultima, <sanchez>-
def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Carga un archivo YAML y lo devuelve como dict.

    Si el archivo no existe, devolvemos {} para no romper el flujo.
    Eso permite que, por ejemplo, base.yaml exista siempre,
    pero local.yaml o prod.yaml sean opcionales al principio.
    """
    if not path.exists():                        #-si no existe el archivo entonces devuelve diccionario vacio pa que no
        return {}                                #   caiga, <sanchez>-
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}           #-safe_load convierte YAML -> estructuras Python seguras
                                                 #  Si el YAML está vacío y safe_load devuelve None, usamos {}, 
                                                 #  <sanchez>-

#-Explicacion preludio (sobre como funcionan las variables de entorno):
#  1er caso: en WSL con Bash, una variable de entorno se puede cargar solo para la terminal actual con 
#              export MI_VARIABLE="valor"; 
#
#            en ese caso queda disponible en esa shell y en los procesos que esa shell lance, pero no “se instala” de 
#            forma permanente ni modifica otras terminales ya abiertas -por asi decirlo, queda durante lo que dure esa 
#            sesion de terminal-..... eso sucede porque eso afecta la shell actual y los programas que lances desde ahí, 
#            porque Bash exporta variables al entorno de los procesos hijos; pero un comando ejecutado en ese entorno 
#            separado no puede modificar la shell padre (en el 3er caso esto se medio repite).
#            
#  2do caso: también se puede dejar persistente una variable de entorno para futuras shells del usuario escribiendo el 
#            export en archivos de inicio como ~/.bashrc o ~/.bash_profile: 
#              nano ~/.bashrc  --> y dentro copiar: export MI_VARIABLE="lo_que_sea"
#
#            Bash lee ~/.bashrc al arrancar una shell interactiva no-login, y lee /etc/profile y luego 
#            ~/.bash_profile/~/.bash_login/~/.profile en shells login; 
#            además, es típico que ~/.bash_profile cargue ~/.bashrc, 
#            así que, por inferencia, toda nueva terminal que lea ese archivo verá la variable, mientras que las 
#            terminales ya abiertas no cambian hasta que hagas source ~/.bashrc o abras una nueva.
#            Por lo que para hacerlo efectivo en la terminal en que lo hizo, tendria que hacer:
#              nano ~/.bashrc  --> y dentro copiar: export MI_VARIABLE="lo_que_sea"
#              source ~/.bashrc
#             
#            Resumen: Bash lee ~/.bashrc en shells interactivas no-login, 
#                     y en shells login lee /etc/profile y luego ~/.bash_profile, ~/.bash_login o ~/.profile; además, 
#                     es habitual que ~/.bash_profile cargue ~/.bashrc. 
#                     Por eso, en la práctica, ponerlo en ~/.bashrc suele hacer que aparezca en nuevas terminales 
#                     interactivas.
#
#  3er caso: en Conda, además, puedes asociar variables (que queras o penses como de entorno, variables de entorno) a 
#            un entorno concreto con 
#              conda env config vars set MI_VAR=valor -n mi_env; 
#              
#            después debes reactivar el entorno (para que se haga efectivo el cambio pues), y esa variable solo existe 
#            mientras ese entorno esté activo en esa sesión, no de forma global, y desaparece al desactivarlo (pero 
#            vuelve a aparecer una vez vuelve a activar el entorno obvio..... es decir, la variable queda asociada al 
#            entorno y se aplica cada vez que ese entorno está activo.... de modo que en por ejm terminales
#            que no tenga el entorno activado que esten al mismo tiempo que una o unas que si, no tendran acceso
#            a dichas variables por ejm.... pues again, Bash además solo pasa variables exportadas a los procesos hijos 
#            que lanza esa-una shell, no a shells hermanas o independientes), <sanchez>-

def load_settings(env: str | None = None) -> dict[str, Any]:
    load_dotenv(ROOT_DIR / ".env")  #-load_dotenv(ROOT_DIR / ".env") intenta leer el archivo .env en esa direccion, 
                                    #  dicho archivo en principio, seria un .env local (es decir, de ese tipo).
                                    #  Ahora bien tenemos dos caso:
                                    #  * Si el archivo existe    --> lo que este dentro de este (las-sus variables) 
                                    #                                se cargan en el entorno del proceso actual de 
                                    #                                Python (os.environ)..., y luego pueden leerse con 
                                    #                                os.getenv(...).
                                    #                                
                                    #                                Y ademas la funcion devuelve un bool: True si logró 
                                    #                                cargar al menos una variable de entorno
                                    #                                En caso de que el archivo exista pero este vacio
                                    #                                entonces no lograra cargar al menos una variable 
                                    #                                de entorno, por lo que devolvera bool: False.... 
                                    #                                igual, note que en nuestro caso no estamos 
                                    #                                capturando-guardando lo que devuelve la funcion
                                    #                                (con por ejm alguna variable o idk)
                                    #
                                    #                                en caso de estar vacio, no lanza error
                                    #                                y de manera efectiva no carga ninguna variable
                                    #                                pues no hay nada dentro del archivo para cargar
                                    #
                                    #  * Si el archivo no existe --> no lance error
                                    #                                no cargue nada
                                    #                                y la función simplemente devuelva bool: False
                                    #  
                                    #  Importante:
                                    #  - No hace falta asignar el resultado a una variable.
                                    #  - Se usa mas que todo por su efecto secundario: poblar el entorno del proceso.
                                    #  - Si el archivo .env no existe, normalmente no revienta; simplemente
                                    #    no carga nada desde ahí, <sanchez>-

    env_name = env or os.getenv("APP_ENV", "local")  #-Sobre os.getenv(...)
                                                     #  --------------------
                                                     #  os.getenv("NOMBRE", "default") busca una variable de entorno,
                                                     #  en este caso "NOMBRE" (la asi denominada pues) en el entorno 
                                                     #  del proceso actual de Python.
                                                     #  - Si existe, devuelve su valor.
                                                     #  - Si no existe, devuelve el default, <sanchez>-
                                                     #
                                                     #-El caso, es que el orden logico que llevamos hasta ahora ha sido
                                                     #  Orden lógico:
                                                     #   1. intenta cargar variables desde .env (con la 1ra instruccion)
                                                     #   2. decide qué ambiente usar (con esta instruccion), <sanchez>-
                                                     #  
                                                     #  .... o sea, en 2. decide si usar el str que puede pasar uno al
                                                     #  invocar la funcion; o si no se paso nada y env=None, entonces
                                                     #  buscaria en la variable de entorno "APP_ENV" que se supone 
                                                     #  fue inyectada en el paso 1. (o estaba inyectada en la terminal
                                                     #  desde antes... mas adelante hablamos de eso); o si esta 
                                                     #  no esta, entonces por defecto se usara "local" (ese seria
                                                     #  nuestro fallback por defecto), <sanchez>-

    base_cfg = _load_yaml(CONFIG_DIR / "base.yaml")        #-3. carga base.yaml, <sanchez>-
    env_cfg = _load_yaml(CONFIG_DIR / f"{env_name}.yaml")  #-4. carga el YAML del ambiente (local.yaml, prod.yaml, 
                                                           #    etc.), <sanchez>-

    settings = _deep_merge(base_cfg, env_cfg)              #-5. mezcla ambas configuraciones (las ve como diccionarios
                                                           #    y las mezcla), <sanchez>-

    settings["env"] = {                                              #-6. inyecta variables sensibles leídas del entorno, 
        "app_env": env_name,                                         #    de modo que el "nuevo documento" setting 
        "twelve_data_api_key": os.getenv("TWELVE_DATA_API_KEY", ""), #    queda o se le agrega algo como 
    }                                                                #     env:
                                                                     #         app_env: [inserte el del env partc]
                                                                     #         twelve_data_api_key: [inserte la key esa]
                                                                     #  , <sanchez>-
    return settings   #-7. devuelve el settings final, <sanchez>-

    #-Acotacion: Casos posibles
    #  --------------
    #  anteriomente señale que las variables de entorno podian estar inyectadas en la terminal desde antes, por eso
    #  veo los siguientes casos-contextos posibles en que se puede ejecutar este codigo:
    #  
    #  1. Existe .env y NO existe la variable exportada en el entorno:
    #     -> se usa lo cargado desde .env
    #  
    #  2. NO existe .env pero SÍ existe la variable exportada en el entorno:
    #     -> se usa la variable del entorno
    #  
    #  3. NO existe .env y NO existe la variable en el entorno:
    #     -> se usa el valor por defecto dado a os.getenv(...)
    #  
    #  4. Existen ambas:
    #     -> normalmente la variable ya exportada en el entorno tiene prioridad,
    #        y load_dotenv no la sobreescribe por defecto
    #
    #  Se sigue por tanto, que parece ser que, o bajo mi interpretacion del codigo en su contexto y por como esta hecho,
    #  lo que se espera o lo correcto es que NO HAYAN VARIABLES DE ENTRONO CARGADAS "A MANO", sino que se espera que 
    #  todo esto se maneje via archivos .env ....... lo cual tiene sentido desde una perspectiva luego de lanzado a 
    #  produccion......... y notese que pa que el codigo caiga por eso esta complicado, por lo que no se si seria
    #  ideal capturar ese error de alguna manera, por ahora, sera primordial mantenerle un ojo encima a ese detalle.
    #
    #  Critica: actualmente el codigo esta hecho para concebir un env_name i,e para que este codigo pueda funcionar 
    #           tanto para un caso de una configuracion de entorno como para otro (fase 1 y 2). Aun asi, veo 
    #           problematico que la direccion es unica y que solo existe un .env con su unico "APP_ENV", por lo que 
    #           entonces la colocada del "APP_ENV" en .env se automatiza con otro codigo o idk (tenemos un dibujo de 
    #           propuesta aun no efectiva).... o se hace manual..... que pues bueno, parece que la idea es mas 
    #           automatica, pero como solo son dos fases, ta bien, <sanchez>-
