# --- PASO 1: Instalar dependencias ---
#!pip install requests

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
 "Dr_Tiger"
]
BLITZ_TOP_200 = [
    "Dr_Tiger",
    "Sportik_Shark",
    "Fran4477",
    "Yarebore",
    "cutemouse83",
    "teem1",
    "FeelingLiberated",
    "athena-pallada",
    "delta_horsey",
    "Schachmachine",
    "InvisibleGuest2023",
    "Tuzakli_Egitim",  # Ya estaba en rapid, pero muy fuerte
    "Mr_BaronMunchausen",
    "TigerSiberian",
    "Ilikeknightmost",
    "lqevsll",
    "Tintirito",
    "Ciderdrinker",
    "Arjun2002",
    "BrittaBagels",
    "ilqar_7474",  # Muy fuerte en múltiples formatos
    "elconceto",
    "NeverEnough",
    "Tsoi_Dima",
    "AVS2000",
    "ChessLab_Online",
    "Lintchevski_Daniil",
    "Batko2003",
    "Amateur_IM",
    "BackToJune",
    "Hotprimus",
    "vistagausta",
    "RealDavidNavara",
    "MatthewG-p4p",  # Ya en otras listas
    "am31",
    "LoveFifi",
    "Juan_Delasala",
    "Lu_Shanglei",
    "Secret_Spion",
    "FeegLood",
    "Lukianoo",
    "Mlchael",
    "evg85",
    "Bllitzer11",
    "KAPUTVSEMU",
    "MJRFRNLM",
    "Freestyler1999",
    "hideseeker",
    "QuarterPawn",
    "vistagusta",  # Duplicado con vistagausta, corregido
    "arturchix",
    "gmmoranda",
    "LeelooKorbenDallas",
    "lovely_fella",
    "Chewbacca18",
    "flower9503",
    "pozvonochek",
    "nrvisakh",
    "BenjaminBokTwitch",
    "platinumcrown",  # Ya en otras listas
    "IVK88",
    "EvilGenius94",
    "gefuehlter_FM",
    "DiaryTraining",
    "raf1310",
    "Bamboccione",
    "Zangyglobal",
    "TarmosanCheese",
    "WhiteRabbit00735",
    "MikeGScarn",
    "Jakebr8",
    "chessed70",
    "vaynebot",
    "AlexTriapishko",
    "DeutcheArtist3000",
    "BrightFalcon",
    "Solidifying",
    "cjota95",
    "Buhmann",
    "Azzir",
    "DrawDenied_Twitch",
    "SF-46",
    "Shpittsik",
    "Xmaskerino_Student",
    "NINDJAxx8",
    "bishop1984",
    "Christened",
    "TacticsMan23",
    "Elda64",
    "Cann_Karo",
    "Cruise97",
    "MOTAR0",
    "Dragso",
    "Iron_Man1703",
    "chesswithmra",
    "BrokyLoveKFC",
    "Rakhmanov_Aleksandr",  # Fuerte en todas partes
    "GK1963",
    "K-Georgiev",
    "ixci",
    "ReluctantCannibal",
    "anonymousblitz",
    "temus22",
    "paniculata",
    "BENED7707",
    "woshigeshagua",
    "DingLiren7777777",
    "Samid2002",
    "Nana_jede_cevape",
    "Heisenberg01",
    "bulletnik",
    "Hanskai",
    "TryhardTrainingTiger",
    "Unbroken_Warrior",
    "Coach13",
    "Andrey11976",  # Repetido, pero relevante
    "jake_jortles",
    "Popovich2020",
    "abudabi22840",
    "BatmanBinSuperman",
    "dr_XXL",
    "Baidetskyi_Valentin",
    "horsebishop123",
    "Conan_The_Barbarian8",
    "Chesstoday",
    "Sigma_Tauri",
    "Pyama_Ninja",
    "FBINC",
    "Dr-CRO",
    "Ignathor",
    "Chariteawithambition",
    "PauArk",
    "artem64",
    "Sobino106",
    "Vostanin",
    "karagioules",
    "Fabsid",
    "forgiants",
    "Molt7n",
    "MrTactic2008",
    "TBold3",
    "gan06",
    "mzh1995",
    "Epoch11",
    "Alpha_777",
    "hansi_flick",
    "vodanou01",
    "Mikhailov_Viacheslav",
    "Jamorris94",
    "Nefer-Pitou",
    "FishhingForFish",
    "efourwhitewins",
    "IgorKowalski",
    "Train_account",
    "AnIndianChessplayer",  # Ya en otras listas
    "LeNoobsInLeTactics",
    "hristiyan06",
    "abrakadabra02",
    "Pavel-Vorontsov",
    "Sergeiaza",
    "FastforYou",
    "Woland87",
    "rasulovvugar",
    "Brokeno_smekla",
    "onepoundrook",
    "Finite_Incantatem777",
    "enmQ",
    "ReynaMatias",
    "Jumangee",
    "Lyonbeast21",
    "Kinryln",
    "tjychess",
    "AttackingBeast",
    "CrazySage",
    "Koshulyan_Egor",
    "stairsarecool",
    "DarkAlekhine",
    "Art_Vandelay1998",
    "Alexandr_KhleBovich",
    "Pblu35",
    "rfelgaer",
    "mitsutszkisszucs",
    "AASP1124",
    "kingofspeedd",
    "M24883",
    "sapphire_phoenix",
    "SavvaVetokhin2009",
    "Forester_19",
    "Carlsen_brat",
    "IsolatedMushroom",
    "paopaomate",
    "Tractor19702025",
    "Misi_95",
    "nickVet",
    "June31",
    "Yakamuzosan",
    "Biriyani",
    "okriak",
    "justantan",
    "BocchoiBoi"
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