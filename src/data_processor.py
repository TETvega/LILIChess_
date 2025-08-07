# src/data_processor.py (versión con checkpoint)
import numpy as np
import chess.pgn
from pathlib import Path
from board_representation import fen_to_8x8x29

PROCESSED_LOG = "data/processed/processed_files.txt"

def load_processed_files():
    """Carga la lista de archivos ya procesados."""
    log_path = Path(PROCESSED_LOG)
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_processed_file(filename):
    """Guarda un archivo como procesado."""
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def process_all_games(
    input_dir="data/raw",
    output_file="data/processed/training_data.npz",
    max_positions_per_game=80,
    batch_size=20000
):
    X_batch = []
    y_batch = []
    input_path = Path(input_dir)
    pgn_files = list(input_path.glob("*.pgn"))

    # Cargar archivos ya procesados
    processed_files = load_processed_files()
    remaining_files = [
        f for f in pgn_files
        if f.name not in processed_files
    ]

    print(f"🔍 Total de archivos: {len(pgn_files)}")
    print(f"📌 Archivos ya procesados: {len(processed_files)}")
    print(f"🚀 Archivos por procesar: {len(remaining_files)}")

    if not remaining_files:
        print("✅ ¡Todos los archivos ya fueron procesados!")
        return

    for pgn_file in remaining_files:
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

                                board_array = fen_to_8x8x29(fen, last_moves=move_history[-2:])
                                X_batch.append(board_array)
                                y_batch.append(uci_move)

                            except Exception as e:
                                # print(f"⚠️ Error en posición: {e}")
                                pass

                            board.push(move)
                            move_history.append(uci_move)

                        game_count += 1

                        # Guardar en lotes
                        if len(X_batch) >= batch_size:
                            _save_batch(X_batch, y_batch, output_file, append=True)
                            X_batch.clear()
                            y_batch.clear()

                    except Exception as e:
                        # print(f"🚨 Error grave en partida: {e}")
                        continue

                print(f"✅ {game_count} partidas procesadas de {pgn_file.name}")

                # ✅ Marcar como procesado SOLO si se leyó completamente
                save_processed_file(pgn_file.name)

        except Exception as e:
            print(f"❌ Error al abrir {pgn_file}: {e}")

    # Guardar lo que queda
    if X_batch:
        _save_batch(X_batch, y_batch, output_file, append=True)

    print(f"\n🎉 Procesamiento completado. Dataset guardado en: {output_file}")


def _save_batch(X, y, output_file, append=False):
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        data = np.load(output_path)
        X = np.concatenate([data['X'], X], axis=0)
        y = np.concatenate([data['y'], y], axis=0)

    np.savez(output_path, X=X, y=y)

