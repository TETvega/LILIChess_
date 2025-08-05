# src/utils.py

import requests
from pathlib import Path
import time
import os

def download_lichess_games(usernames, output_dir="data/raw", max_games=1000, 
                          rated=False, perf_type=None, format="pgn"):
    """
    Descarga partidas de jugadores de Lichess.
    
    Parámetros:
    - usernames: lista de str (ej: ["magnuscarlsen", "hikaru"])
    - output_dir: carpeta donde guardar los PGNs
    - max_games: cuántas partidas descargar por jugador (máx ~3000, pero 1000 es seguro)
    - rated: solo partidas clasificadas
    - perf_type: "bullet", "blitz", "classical", "rapid", etc.
    - format: "pgn" (único soportado por ahora)
    """
    
    # Crear carpeta si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Endpoint base
    base_url = "https://lichess.org/api/games/user/"
    # visita https://lichess.org/api?spm=a2ty_o01.29997173.0.0.7c1ec921r0wME8 par amas detalles

    # Parámetros
    # Parametros para la API de Lichess
    # https://lichess.org/api#tag/Games/operation/apiGamesUser
    # max: máximo de partidas, rated: si son clasificadas, perfType: tipo de partida, format: formato de salida
    params = {
    "max": max_games,
    "rated": str(rated).lower()
    }
    # Si se especifica un tipo de partida, lo añadimos
    # perfType es opcional, si no se especifica, se descargan todas las partidas
    # format es opcional, pero por ahora solo soportamos "pgn"
    if perf_type:
        params["perfType"] = perf_type
    if format:
        params["format"] = format
    # Headers para evitar bloqueos por scraping
    # Lichess permite scraping, pero es mejor identificarse
    # https://lichess.org/api#section/Introduction/Headers
    # User-Agent es importante para evitar bloqueos

    headers = {
        "User-Agent": "LILICHESS v1.0 - Proyecto de IA educativo"
    }
    
    print(f"Descargando partidas de {len(usernames)} jugadores...")
    
    # Iterar sobre cada usuario
    for username in usernames:
        output_path = Path(output_dir) / f"{username.lower()}.pgn"
        print(f"{username}: solicitando hasta {max_games} partidas...")

        try:
            response = requests.get(
                f"{base_url}{username}",
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                size_kb = len(response.text) // 1000
                print(f"Guardado: {output_path} ({size_kb} KB)\n")
            elif response.status_code == 404:
                print(f" Usuario no encontrado: {username}\n")
            elif response.status_code == 429:
                print(f" Límite de solicitudes alcanzado. Esperando 60 segundos...")
                time.sleep(60)
                continue
            else:
                print(f" Error {response.status_code} al obtener datos de {username}:\n{response.text[:100]}\n")
        
        except requests.RequestException as e:
            print(f"Error al conectar con {username}: {e}\n")
        except Exception as e:
            print(f"Error inesperado al procesar {username}: {e}\n")
        
        time.sleep(1.5)  # Evita saturar la API
    
    print(" Descarga completada.\n")




# top playes
# aqui hay unas referencias de la FIDE pero ensi estos de la lista siguiente son los de LILIchest y su user name en Lichess
# https://lichess.org/player/top/players
# https://lichess.org/player

top_playersEX = [
    "yuuki-asuna",    
    "Vlad_Lazarev79",
    "magnuscarlsen",
    "Hikaru",
    "Lucon2000",
    "matteorf2b",
    "platinumcrown",
    "CoD_Dragon",
    "Oakley666",
    "tommy_pug",
    "Cyberdrunk",
    "avcs",
    "Unstoppable_Rob",
    "White_Robot"

]



def eliminar_archivos_vacios(ruta):
    """
    Elimina todos los archivos vacíos en la ruta especificada.

    Parámetros:
    ruta (str): Ruta del directorio donde buscar archivos vacíos.
    """
    if not os.path.isdir(ruta):
        print(f"La ruta '{ruta}' no es válida.")
        return

    archivos_eliminados = 0

    for archivo in os.listdir(ruta):
        ruta_completa = os.path.join(ruta, archivo)
        if os.path.isfile(ruta_completa) and os.path.getsize(ruta_completa) == 0:
            os.remove(ruta_completa)
            archivos_eliminados += 1
            print(f"Eliminado: {ruta_completa}")

    print(f"Total de archivos vacíos eliminados: {archivos_eliminados}")














if __name__ == "__main__":
    print("Menu")
    print("1. Descargar partidas de jugadores específicos")
    print("2. Eliminar archivos vacíos en un directorio")
    print("3. Salir")
    choice = input("Elige una opción: ")
    if choice == "1":
        usernames = input("Introduce los nombres de usuario separados por comas: ").split(",")
        usernames = [username.strip() for username in usernames] 
        if not usernames:
            print("No se han introducido nombres de usuario. Usando jugadores populares por defecto.")
            usernames = top_playersEX
        output_dir = input("Introduce el directorio de salida (por defecto 'data/raw'): ") or "data/raw"
        max_games = int(input("Máximo de partidas por jugador (por defecto 1000): ") or 1000)
        rated = input("Solo partidas clasificadas? (s/n, por defecto n): ").lower() == 'n'
        perf_type = input("Tipo de partida (bullet, blitz, classical, rapid, etc., por defecto None): ") or None
        download_lichess_games(
            usernames=top_playersEX,
            output_dir="data/raw",
            max_games=1000,
            rated=False,           
            perf_type=None
        )
    elif choice == "2":
        print(Path.cwd())
        dir_path = input("Introduce el directorio donde eliminar archivos vacíos: ")
        eliminar_archivos_vacios(dir_path)
    elif choice == "3":
        print("Saliendo...")
        exit()

