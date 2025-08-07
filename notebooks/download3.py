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
"asacopaco",
"emmanueljimenez",
"suhov549",
"fabsid",
"pawneatingzombie",
"jumangee",
"igorkowalski",
"chessclubkomotini",
"yorkster",
"abyin2000",
"starkid20",
"irfanharithrosli",
"lu_shanglei",
"aa2017",
"nilete",
"viktor_tahirov555",
"i_love_1c",
"jakebr8",
"mistheoretical",
"m_t_h",
"victory_biber",
"schachstratege",
"chess_for_some",
"rubinsteinsmonster",
"white_robot",
"eldarmv",
"alex_chi",
"molt7n",
"qls-xagavan",
"qcfxi_dizsdicx",
"blackmist2025",
"idvan",
"sergeirk",
"imangalikhafiz",
"fasmc17",
"bandhawk",
"grrrrrrr_r",
"matteorf2b",
"silent-killer2265",
"thecablebox",
"fatcatsat",
"kawaciukov",
"faznaz83",
"alpha_777",
"vistagusta",
"mrtactic2008",
"delebarre",
"sobino106",
"trenersahnutyi",
"schachinator999",
"conwycastle",
"mkenenisa",
"pauark",
"akshak",
"simchev",
"antonpraim",
"dynamicus",
"wizard1983",
"mw1966",
"artem_3000",
"nobita2345678",
"stn_chess",
"platinumcrown",
"henryshen2024",
"stairsarecool",
"stefan_95",
"xicxoc",
"bishop1984",
"gk1963",
"nikolarasss",
"tavli",
"nitrogue",
"dr-cro",
"yakamuzosan",
"brokylovekfc",
"steva23",
"truemasterme",
"felicity187",
"stallion28",
"drudim",
"artistendo",
"avslugin80",
"extermo44",
"bida1992",
"hansi_flick",
"bagera9",
"hematom87",
"unbroken_warrior",
"whenrooksfly",
"rajibmr",
"alex1-victory20",
"sheep1965",
"lada23new",
"realdavidnavara"
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