# src/data_processor.py (versión robusta)
import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess
import chess.pgn
from pathlib import Path
from tqdm import tqdm
from board_representation import fen_to_8x8x22

def process_all_games(
    input_dir="notebooks/data/raw",
    output_file="data/processed/training_data.npz",
    max_positions_per_game=50,
    batch_size=10000  # Guardar en lotes para no llenar la RAM
):
    X_batch = []
    y_batch = []

    input_path = Path(input_dir)
    pgn_files = list(input_path.glob("*.pgn"))

    if not pgn_files:
        print("❌ No se encontraron archivos .pgn en la carpeta.")
        return

    print(f"🔍 {len(pgn_files)} archivos PGN encontrados. Procesando...")

    total_positions = 0
    total_games = 0

    for pgn_file in pgn_files:
        print(f"📁 Procesando: {pgn_file.name}")
        try:
            with open(pgn_file, encoding='utf-8') as f:
                game_count = 0
                while True:
                    try:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break

                        board = game.board()
                        move_history = []

                        for move in game.mainline_moves():
                            if len(move_history) >= max_positions_per_game:
                                break

                            try:
                                fen = board.fen()
                                uci_move = move.uci()

                                # Validación básica de FEN
                                if '[]' in fen or len(fen) < 10 or fen.count(' ') < 5:
                                    board.push(move)
                                    move_history.append(uci_move)
                                    continue

                                # Convertir FEN a array
                                board_array = fen_to_8x8x22(fen, last_moves=move_history[-2:])
                                X_batch.append(board_array)
                                y_batch.append(uci_move)

                                total_positions += 1

                            except Exception as e:
                                # print(f"⚠️ Error en posición: {e}")
                                pass

                            board.push(move)
                            move_history.append(uci_move)

                        game_count += 1
                        total_games += 1

                        # Guardar en lotes
                        if len(X_batch) >= batch_size:
                            _save_batch(X_batch, y_batch, output_file, append=True)
                            X_batch.clear()
                            y_batch.clear()

                    except Exception as e:
                        # print(f"🚨 Error grave en partida: {e}")
                        continue

                print(f"✅ {game_count} partidas procesadas de {pgn_file.name}")

        except Exception as e:
            print(f"❌ Error al abrir {pgn_file}: {e}")

    # Guardar lo que queda
    if X_batch:
        _save_batch(X_batch, y_batch, output_file, append=True)

    print(f"\n🎉 Procesamiento completado.")
    print(f"📊 Total de partidas procesadas: {total_games}")
    print(f"📌 Total de posiciones guardadas: {total_positions}")
    print(f"💾 Dataset guardado en: {output_file}")

def _save_batch(X, y, output_file, append=False):
    """Guarda un lote de datos, creando o extendiendo el archivo .npz"""
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        data = np.load(output_path)
        X_prev, y_prev = data['X'], data['y']
        X = np.concatenate([X_prev, X], axis=0)
        y = np.concatenate([y_prev, y], axis=0)

    np.savez(output_path, X=X, y=y)

if __name__ == "__main__":
    process_all_games(max_positions_per_game=50, batch_size=20000)