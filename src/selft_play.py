# src/self_play.py

import sys
import os

# Añade el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import chess
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from src.agent import ChessAgent
from src.mcts import MCTS
import time
import logging
from datetime import datetime

# --- Configuración de logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"logs/self_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuración segura para 4GB de VRAM ---
NUM_PROCESSES = 1                  # Usa 1 proceso para evitar OOM
BATCH_SIZE_SAVE = 5                # Guarda cada 5 partidas
TEMPERATURE = 1.0                  # Exploración en self-play
NUM_SIMULATIONS = 150              # Simulaciones por movimiento
NUM_GAMES = 30                     # Número total de partidas


def play_single_game(game_idx, agent_path="models/current/best_model.keras", vocab_path="data/processed/move_vocab.pkl"):
    """
    Juega una partida de self-play.
    Retorna: (data, stats) con posiciones y estadísticas.
    """
    start_time = time.time()
    logger.info(f"🎮 Iniciando partida {game_idx + 1}")

    try:
        # Crea agente dentro del proceso
        agent = ChessAgent(model_path=agent_path, vocab_path=vocab_path)
        mcts = MCTS(agent, num_simulations=NUM_SIMULATIONS)

        board = chess.Board()
        game_history = []

        while not board.is_game_over() and len(game_history) < 150:
            fen = board.fen()
            move_history = [m.uci() for m in board.move_stack[-2:]]

            # Ejecutar MCTS
            mcts_policy = mcts.run(board)

            # Política con temperatura
            visit_counts = np.array([count for count, _ in mcts_policy])
            policy = visit_counts ** (1 / TEMPERATURE)
            policy = policy / (policy.sum() + 1e-8)

            # Elegir movimiento
            if len(board.move_stack) < 10:
                move_idx = np.random.choice(len(mcts_policy), p=policy)
            else:
                move_idx = np.argmax(policy)
            chosen_move = mcts_policy[move_idx][1]

            # Guardar estado
            policy_dict = {move.uci(): prob for prob, move in mcts_policy}
            game_history.append({
                "fen": fen,
                "policy": policy_dict,
                "turn": "white" if board.turn == chess.WHITE else "black"
            })

            board.push(chosen_move)

        # Resultado final
        result = board.result()
        if result == "1-0":
            winner = "white"
        elif result == "0-1":
            winner = "black"
        else:
            winner = "draw"

        # Asignar valor a cada estado
        data = []
        for state in game_history:
            value = 1.0 if state["turn"] == winner else -1.0 if winner != "draw" else 0.0
            data.append({
                "fen": state["fen"],
                "policy": state["policy"],
                "value": value
            })

        # Estadísticas
        duration = time.time() - start_time
        num_moves = len(game_history)
        outcome = result

        logger.info(f"✅ Partida {game_idx + 1} completada | Movimientos: {num_moves} | Ganador: {winner.upper()} | Duración: {duration:.1f}s")

        return data, {
            "game_idx": game_idx + 1,
            "duration": duration,
            "num_moves": num_moves,
            "winner": winner,
            "outcome": outcome
        }

    except Exception as e:
        logger.error(f"❌ Error en partida {game_idx + 1}: {e}")
        return [], {
            "game_idx": game_idx + 1,
            "duration": time.time() - start_time,
            "num_moves": 0,
            "winner": "error",
            "outcome": "error"
        }


def run_self_play_parallel(num_games=NUM_GAMES, num_processes=NUM_PROCESSES, batch_size_save=BATCH_SIZE_SAVE):
    """
    Ejecuta self-play en paralelo con progreso y logging.
    """
    logger.info("🚀 Iniciando self-play paralelo")
    logger.info(f"🔹 Partidas: {num_games} | Procesos: {num_processes} | Simulaciones: {NUM_SIMULATIONS} | Lotes: {batch_size_save}")

    func = partial(play_single_game)
    all_data = []
    all_stats = []

    with Pool(processes=num_processes) as pool:
        for i in range(0, num_games, batch_size_save):
            batch_games = min(batch_size_save, num_games - i)
            game_indices = list(range(i, i + batch_games))

            logger.info(f"📦 Iniciando lote {i//batch_size_save + 1} | Partidas: {i+1} a {i+batch_games}")

            with tqdm(total=batch_games, desc=f"Lote {i//batch_size_save + 1}", unit="partida") as pbar:
                for data, stats in pool.imap_unordered(func, game_indices):
                    if data:  # Si no hubo error
                        all_data.extend(data)
                        all_stats.append(stats)
                    pbar.set_postfix({
                        "Movs": f"{stats['num_moves']}",
                        "Ganador": stats['winner'][:2].upper(),
                        "Duración": f"{stats['duration']:.1f}s"
                    })
                    pbar.update()

            # Guardar lote
            os.makedirs("data/self_play", exist_ok=True)
            save_path = f"data/self_play/self_play_data_batch_{i//batch_size_save}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(all_data, f)
            logger.info(f"💾 Lote {i//batch_size_save + 1} guardado: {len(all_data)} posiciones en {save_path}")

    # Guardar todo
    full_path = "data/self_play/self_play_data_full.pkl"
    with open(full_path, "wb") as f:
        pickle.dump(all_data, f)
    logger.info(f"✅ Datos combinados guardados en: {full_path}")

    # Guardar estadísticas
    stats_path = "data/self_play/self_play_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(all_stats, f)
    logger.info(f"📊 Estadísticas guardadas en: {stats_path}")

    # Resumen final
    total_time = sum(s["duration"] for s in all_stats)
    avg_moves = np.mean([s["num_moves"] for s in all_stats])
    white_wins = sum(1 for s in all_stats if s["winner"] == "white")
    black_wins = sum(1 for s in all_stats if s["winner"] == "black")
    draws = sum(1 for s in all_stats if s["winner"] == "draw")
    errors = sum(1 for s in all_stats if s["winner"] == "error")

    logger.info("="*60)
    logger.info("🏁 SELF-PLAY COMPLETADO")
    logger.info("="*60)
    logger.info(f"📊 Total partidas: {len(all_stats)} ({errors} con error)")
    logger.info(f"📁 Posiciones generadas: {len(all_data)}")
    logger.info(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"♟️  Promedio de movimientos: {avg_moves:.1f}")
    logger.info(f"🏆 Victorias: Blancas {white_wins} - Negras {black_wins} - Tablas {draws}")
    logger.info(f"❌ Errores: {errors}")
    logger.info(f"💾 Último lote: {full_path}")
    logger.info("="*60)

    return all_data


# --- Ejecución ---
if __name__ == "__main__":
    logger.info("🎮 Iniciando script de self-play")
    run_self_play_parallel(num_games=NUM_GAMES)
    logger.info("🔚 Script de self-play finalizado")