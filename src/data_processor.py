# src/data_processor.py (versión corregida)

import os
import numpy as np
import chess
import chess.pgn
from pathlib import Path
from board_representation import fen_to_8x8x22

def process_all_games(input_dir="data/raw", output_file="data/processed/training_data.npz", max_positions_per_game=50):
    X = []
    y = []

    input_path = Path(input_dir)
    pgn_files = list(input_path.glob("*.pgn"))

    print(f"Encontrados {len(pgn_files)} archivos PGN. Procesando...")

    for pgn_file in pgn_files:
        print(f"Procesando: {pgn_file.name}")
        with open(pgn_file, encoding='utf-8') as f:
            game_count = 0
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    board = game.board()
                    move_count = 0
                    move_history = []

                    for move in game.mainline_moves():
                        if move_count >= max_positions_per_game:
                            break

                        try:
                            fen = board.fen()
                            uci_move = move.uci()

                            # Validar que el FEN es correcto
                            if '[]' in fen or len(fen) < 10 or ' ' not in fen:
                                print(f"FEN inválido saltado: {fen[:30]}...")
                                board.push(move)
                                move_history.append(uci_move)
                                continue

                            #Intentar crear el array
                            board_array = fen_to_8x8x22(fen, last_moves=move_history[-2:])
                            X.append(board_array)
                            y.append(uci_move)

                        except Exception as e:
                            print(f"Error procesando posición: {e}")
                            pass  # Ignora esta posición

                        # Aplica movimiento
                        board.push(move)
                        move_history.append(uci_move)
                        move_count += 1

                    game_count += 1
                    if game_count % 100 == 0:
                        print(f"  → {game_count} partidas procesadas...")

                except Exception as e:
                    print(f"Error grave en una partida: {e}")
                    continue  # Sigue con la siguiente partida

    # Guardar solo si hay datos
    if len(X) == 0:
        print("No se procesó ningún estado. Revisa los archivos PGN.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    np.savez(output_file, X=X, y=y)
    print(f"Dataset guardado: {output_file}")
    print(f"Forma de X: {X.shape}, tamaño de y: {len(y)}")

if __name__ == "__main__":
    process_all_games()