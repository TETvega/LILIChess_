# data_processor_optimized.py

import numpy as np
import chess
import chess.pgn
from pathlib import Path
import os
import io
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import re

# === CONFIGURACIÓN DE LOGGING ===
LOGS_DIR = "logs"
LOG_FILE = Path(LOGS_DIR) / "processing.log"

# Crear directorio de logs
Path(LOGS_DIR).mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def fen_to_8x8x29(fen: str, last_moves: list = None):
    """
    Convierte un FEN a un array 8x8x29.
    Cada plano representa una característica del tablero.
    """
    if last_moves is None:
        last_moves = []
    
    board = chess.Board(fen)
    planes = np.zeros((8, 8, 29), dtype=np.float32)
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    
    # 0-11: Piezas (blancas 0-5, negras 6-11)
    for i in range(8):
        for j in range(8):
            sq = chess.square(j, 7 - i)
            piece = board.piece_at(sq)
            if piece:
                idx = piece_to_index[piece.symbol().upper()]
                if piece.color == chess.WHITE:
                    planes[i, j, idx] = 1
                else:
                    planes[i, j, idx + 6] = 1
    
    # 12: Turno (1 si es blanco)
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1
    
    # 13-16: Enroque
    if board.has_kingside_castling_rights(chess.WHITE): planes[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): planes[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): planes[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): planes[:, :, 16] = 1
    
    # 17: Al paso
    if board.ep_square:
        file = chess.square_file(board.ep_square)
        rank = 7 - chess.square_rank(board.ep_square)
        planes[rank, file, 17] = 1
    
    # 18: Jaque
    if board.is_check():
        planes[:, :, 18] = 1
    
    # 19: 50 movimientos sin progreso
    planes[:, :, 19] = 1 if board.halfmove_clock >= 50 else 0
    
    # 20-21: Últimos 2 movimientos
    recent_moves = last_moves[-2:]
    for k, move_uci in enumerate(recent_moves):
        try:
            move = chess.Move.from_uci(move_uci)
            fr = move.from_square
            to = move.to_square
            i_fr, j_fr = 7 - chess.square_rank(fr), chess.square_file(fr)
            i_to, j_to = 7 - chess.square_rank(to), chess.square_file(to)
            planes[i_fr, j_fr, 20 + k] = 1
            planes[i_to, j_to, 20 + k] = 1
        except:
            pass
    
    # 22: Distancia del rey blanco al centro
    w_king = board.king(chess.WHITE)
    if w_king:
        dist = ((chess.square_file(w_king) - 3.5)**2 + (7 - chess.square_rank(w_king) - 3.5)**2)**0.5
        planes[:, :, 22] = dist / 6
    
    # 23: Distancia del rey negro al centro
    b_king = board.king(chess.BLACK)
    if b_king:
        dist = ((chess.square_file(b_king) - 3.5)**2 + (7 - chess.square_rank(b_king) - 3.5)**2)**0.5
        planes[:, :, 23] = dist / 6
    
    # 24: Control del centro (d4, d5, e4, e5)
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    for sq in center:
        i, j = 7 - chess.square_rank(sq), chess.square_file(sq)
        attackers = len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq))
        planes[i, j, 24] = np.tanh(attackers / 3)
    
    # 25: Estructura de peones (dobles, pasados)
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        # Peón doblado
        if any(chess.square_file(p) == file and chess.square_rank(p) < rank for p in board.pieces(chess.PAWN, chess.WHITE)):
            planes[7 - rank, file, 25] = 0.5
        # Peón pasado
        is_passed = True
        for f in [file-1, file, file+1]:
            if 0 <= f <= 7:
                for r in range(rank+1, 8):
                    if board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color == chess.BLACK:
                        is_passed = False
        if is_passed:
            planes[7 - rank, file, 25] = 1.0
    
    # 26: Piezas desarrolladas (N y B fuera de la fila inicial)
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        if chess.square_rank(sq) > 1:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        if chess.square_rank(sq) > 1:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        if chess.square_rank(sq) < 6:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        if chess.square_rank(sq) < 6:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    
    # 27: Mapa de ataques
    for i in range(8):
        for j in range(8):
            sq = chess.square(j, 7 - i)
            diff = len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq))
            planes[i, j, 27] = np.tanh(diff / 5)
    
    # 28: Movilidad (número de movimientos legales por casilla)
    mobility = np.zeros((8, 8))
    for move in board.legal_moves:
        to_sq = move.to_square
        i_to, j_to = 7 - chess.square_rank(to_sq), chess.square_file(to_sq)
        mobility[i_to, j_to] += 1
    planes[:, :, 28] = np.tanh(mobility / 10)
    
    return planes

# === CONFIGURACIÓN GENERAL ===
RAW_DATA_DIR = "notebooks/data/raw"
PROCESSED_DIR = "data/processed"
PROCESSED_LOG_FILE = Path(LOGS_DIR) / "processed_files.txt"
MAX_WORKERS = max(1, mp.cpu_count() - 4)

# Patrón para extraer: jugador_tipo_año.pgn
PATTERN = re.compile(r'(?P<player>[\w\-]+)_(?P<time_control>\w+)_\d{4}')

def get_metadata_from_filename(filename: str):
    """Extrae jugador y tipo de partida del nombre del archivo.
    El tipo de partida es lo que está después del último '_', antes de '.pgn'."""
    name = Path(filename).stem  # Quita .pgn
    if '_' not in name:
        return "unknown", "unknown"
    
    # Divide solo por el último guion bajo
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        player = parts[0]  # Todo antes del último _
        time_control = parts[1].lower()  # Lo último (bullet, blitz, etc.)
        return player, time_control
    return "unknown", "unknown"

def process_single_game(game_content: str):
    X_local = []
    y_local = []
    try:
        game = chess.pgn.read_game(io.StringIO(game_content))
        if game is None:
            return [], []
        board = game.board()
        move_history = []
        for move in game.mainline_moves():
            try:
                fen = board.fen()
                uci_move = move.uci()
                if len(fen) < 10 or '[]' in fen:
                    board.push(move)
                    move_history.append(uci_move)
                    continue
                board_array = fen_to_8x8x29(fen, last_moves=move_history[-2:])
                X_local.append(board_array)
                y_local.append(uci_move)
            except Exception:
                pass
            board.push(move)
            move_history.append(uci_move)
    except Exception:
        pass
    return X_local, y_local

def process_pgn_file(args):
    pgn_path, processed_log = args
    X_batch = []
    y_batch = []

    # Extraer metadatos
    filename = pgn_path.name
    player, time_control = get_metadata_from_filename(filename)

    # Verificar si ya existe el .npz
    temp_file = Path(PROCESSED_DIR) / f"temp_{pgn_path.stem}.npz"
    if temp_file.exists():
        logger.warning(f"⚠️  Saltado (archivo .npz ya existe): {filename} | {player} | {time_control}")
        return 0, player, time_control, 0, pgn_path.stat().st_size

    # Verificar si ya está en el log
    if filename in processed_log:
        logger.warning(f"⚠️  Saltado (registrado en log): {filename} | {player} | {time_control}")
        return 0, player, time_control, 0, pgn_path.stat().st_size

    try:
        file_size = pgn_path.stat().st_size / (1024 * 1024)  # MB
        with open(pgn_path, 'r', encoding='utf-8') as f:
            content = f.read()
        games = [g for g in content.strip().split("\n\n\n") if g.strip()]
        num_games = len(games)

        logger.info(f"📄 Procesando: {filename} | Jugador: {player} | Tipo: {time_control} | Partidas: {num_games} | Tamaño: {file_size:.2f} MB")

        for game_str in games:
            X_game, y_game = process_single_game(game_str)
            X_batch.extend(X_game)
            y_batch.extend(y_game)

        if X_batch and y_batch:
            X_array = np.array(X_batch, dtype=np.float32)
            y_array = np.array(y_batch, dtype=object)
            np.savez_compressed(temp_file, X=X_array, y=y_array)
            npz_size = temp_file.stat().st_size / (1024 * 1024)  # en MB
            logger.info(f"✅ Guardado: {temp_file.name} | Posiciones: {len(X_array)} | Tamaño: {npz_size:.2f} MB")
            try:
                test_load = np.load(temp_file, allow_pickle=True)
                assert 'X' in test_load and 'y' in test_load
                assert len(test_load['X']) == len(test_load['y'])
                test_load.close()
                logger.info(f"✅ Validación exitosa: {temp_file.name}")
            except Exception as e:
                logger.error(f"❌ Guardado corrupto: {temp_file.name} → {e}")
                temp_file.unlink()  # borrar si está corrupto
                return 0, player, time_control, 0, 0
            # Registrar en log
            with open(PROCESSED_LOG_FILE, "a", encoding='utf-8') as log_f:
                log_f.write(f"{filename}\n")

            return len(X_array), player, time_control, num_games, file_size

        else:
            logger.warning(f"⚠️  Sin datos útiles: {filename}")
            return 0, player, time_control, num_games, file_size

    except Exception as e:
        logger.error(f"❌ Error grave con {filename}: {e}")
        return 0, player, time_control, 0, 0

def process_all_games():
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    input_path = Path(RAW_DATA_DIR)
    if not input_path.exists():
        logger.critical(f"❌ Carpeta de datos crudos no encontrada: {input_path}")
        sys.exit(1)

    pgn_files = list(input_path.glob("*.pgn"))
    if not pgn_files:
        logger.critical(f"❌ No se encontraron archivos .pgn en {RAW_DATA_DIR}")
        sys.exit(1)

    # Leer log de procesados
    processed_log = set()
    if PROCESSED_LOG_FILE.exists():
        with open(PROCESSED_LOG_FILE, "r", encoding='utf-8') as f:
            processed_log = {line.strip() for line in f if line.strip()}

    # Contar cuántos .npz ya existen
    existing_npz = {f.name[5:-4] for f in Path(PROCESSED_DIR).glob("temp_*.npz")}  # quita "temp_" y ".npz"

    # Determinar qué archivos faltan
    remaining_files = []
    for f in pgn_files:
        if f.name not in processed_log and f.stem not in existing_npz:
            remaining_files.append(f)

    total_files = len(pgn_files)
    already_processed_by_log = len([f for f in pgn_files if f.name in processed_log])
    already_processed_by_npz = len([f for f in pgn_files if f.stem in existing_npz])
    to_process = len(remaining_files)

    # === LOG INICIAL ===
    logger.info("=" * 80)
    logger.info("🚀 INICIO DEL PROCESAMIENTO DE PARTIDAS DE AJEDREZ")
    logger.info("=" * 80)
    logger.info(f"📁 Archivos PGN encontrados:      {total_files}")
    logger.info(f"✅ Ya procesados (log):           {already_processed_by_log}")
    logger.info(f"✅ Ya procesados (archivo .npz):  {already_processed_by_npz}")
    logger.info(f"🔁 Por procesar:                  {to_process}")
    logger.info(f"⚙️  Núcleos utilizados:            {MAX_WORKERS}")
    logger.info(f"📤 Salida:                        {PROCESSED_DIR}")
    logger.info(f"📄 Log detallado:                 {LOG_FILE}")
    logger.info("-" * 80)

    if to_process == 0:
        logger.info("✅ Todos los archivos ya han sido procesados. Nada que hacer.")
        return

    total_positions = 0
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pgn_file, (f, processed_log)) for f in remaining_files]
        for future in as_completed(futures):
            count, player, time_control, num_games, size_mb = future.result()
            total_positions += count
            results.append({
                'player': player,
                'time_control': time_control,
                'games': num_games,
                'positions': count,
                'size_mb': size_mb
            })

    # === RESUMEN FINAL ===
    logger.info("-" * 80)
    logger.info("📊 RESUMEN FINAL")
    logger.info("-" * 80)
    logger.info(f"📌 Total de posiciones procesadas: {total_positions:,}")
    logger.info(f"📁 Archivos procesados:            {len(results)}")
    logger.info(f"✅ Archivos .npz generados:        {len([r for r in results if r['positions'] > 0])}")
    logger.info(f"⚠️  Archivos sin datos:             {len([r for r in results if r['positions'] == 0])}")
    logger.info("🎉 ¡Procesamiento completado con éxito!")
    logger.info("=" * 80)

if __name__ == "__main__":
    process_all_games()