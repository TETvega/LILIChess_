# --- PASO 1: Instalar dependencias ---
#pip install requests

# --- PASO 2: Código completo con LOG ---
import requests
import time
from pathlib import Path

# === CONFIGURACIÓN ===
OUTPUT_DIR = "data/raw"
LOG_FILE = "download_log.txt"
MAX_GAMES_PER_PLAYER = 2000
RATED_ONLY = True
PERF_TYPES = ["bullet", "blitz", "rapid", "classical"]


# Lista de jugadores (ejemplo con algunos, puedes extenderla)
USERNAMES = [
"rockon_123",
"ajkumar",
"crichero2024",
"anonymousblitz",
"qucani",
"minhnm2020",
"sashalemish",
"just_the-king2011",
"nimzo5",
"vasudev9",
"hulscher_1979",
"miwi2",
"davidmark",
"vladimirsakun",
"cryptopanda",
"conan_the_barbarian8",
"neelunani5",
"cod_dragon",
"nmiq",
"chessyclub",
"padurean_daniel",
"prettyfishy",
"tommy_pug",
"sidfandx",
"lokiy47",
"mintrefreshments",
"pyama_ninja",
"star_arya",
"chessinsomniac",
"killer311",
"shaheus",
"lovely_fella",
"wildmountain2020",
"jbking07",
"haunting_games",
"am31",
"ice_energyteam",
"aleks-lexa",
"underdelo",
"onlyrapid11",
"wangmuyuan123",
"paxavarica",
"k-georgiev",
"chesstrial2013",
"shpittsik",
"xii_marshal",
"dragso",
"bestromano",
"art_vandelay1998",
"drawdenied_twitch",
"massterofmayhem",
"opening_master9",
"freestyler1999",
"tintirito",
"attackingbeast",
"gmshloky",
"re-born",
"shasi12345",
"aldmuo",
"lucon2000",
"ema_roma87",
"feelingliberated",
"bkamber",
"flyinglemon",
"joejan",
"onlyvariantmaster_qx",
"blindjagulep",
"alexhas22",
"jaremac00l",
"familychess1",
"yspchess",
"hideseeker",
"vaynebot",
"diarytraining",
"chesstheory64",
"tiberiandawn",
"elconceto",
"newaccwhothis",
"giobra",
"cruise97",
"brokeno_smekla",
"aotxleviackerman",
"fastforyou",
"silkthewanderer",
"aleatorio00",
"daddy_economics",
"uzkuzk",
"cris2016",
"snakeice",
"salomon_v",
"paopaomate",
"avcs",
"ilikeknightmost",
"anton_maximov",
"temus22",
"jb2906",
"daaleksandrov",
"pinkwafflepig",
"vostanin",
"nosporchess",
"m24883",
"mr_baronmunchausen",
"cutemouse83",
"bhavya33",
"dr_tiger",
"larrythecabledude",
"vadim0507001012",
"ganeshgd",
"sportik_shark"
]
BLITZ_TOP_200 = [

]
# Combinar todas las listas sin duplicados
all_players = list(set(USERNAMES + BLITZ_TOP_200))
# Crear carpetas y limpiar log
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
log_path = Path(LOG_FILE)
#if log_path.exists():
 #   log_path.unlink()  # Borrar log anterior

# Función para escribir en el log
def log_entry(username, perf_type, status, count=None, message=""):
    with open(log_path, 'a', encoding='utf-8') as log:
        line = f"Jugador: {username} | Partida: {perf_type} | Estado: {status}"
        if count is not None:
            line += f" | Descargadas: {count}"
        if message:
            line += f" | {message}"
        log.write(line + "\n")
        print(line)  # También mostrar en pantalla

# === DESCARGAR PARTIDAS ===
def download_games(username, perf_type):
    base_url = "https://lichess.org/api/games/user/"
    params = {
        "max": MAX_GAMES_PER_PLAYER,
        "rated": str(RATED_ONLY).lower(),
        "perfType": perf_type,
        "format": "pgn"
    }
    headers = {"User-Agent": "ChessAI-Project v1.0 - Educational Use"}

    output_path = Path(OUTPUT_DIR) / f"{username.lower()}_{perf_type}.pgn"

    # Evitar descarga duplicada
    if output_path.exists():
        log_entry(username, perf_type, "Saltado", message="Archivo ya existe")
        return

    print(f"📥 Descargando {username} - {perf_type}...")
    try:
        response = requests.get(
            f"{base_url}{username}",
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            content = response.text.strip()
            if content:
                # Contar partidas: cada partida empieza con [Event "..."]
                game_count = content.count("[Event ")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                log_entry(username, perf_type, "Éxito", count=game_count)
            else:
                log_entry(username, perf_type, "Vacío", message="No hay partidas")
        elif response.status_code == 404:
            log_entry(username, perf_type, "Error", message="Usuario no encontrado")
        elif response.status_code == 429:
            log_entry(username, perf_type, "Error", message="Límite de API alcanzado")
            time.sleep(60)
            return False
        else:
            log_entry(username, perf_type, "Error", message=f"HTTP {response.status_code}")

    except Exception as e:
        log_entry(username, perf_type, "Error", message=f"Excepción: {str(e)}")

    time.sleep(1.5)
    return True

# === EJECUCIÓN ===
print("🚀 Iniciando descarga de partidas...\n")

for username in all_players:
    for perf_type in PERF_TYPES:
        success = download_games(username, perf_type)
        if success is False:  # Si hubo 429, espera más
            time.sleep(60)

print(f"\n🎉 ¡Descarga completada! Revisa el log: {LOG_FILE}")