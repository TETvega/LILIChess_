# src/test_model.py

import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import numpy as np
import pickle
import chess
from src.neuronal_network import create_chess_network
from src.board_representation import fen_to_8x8x22



def load_model_and_vocab(model_path="models/current/best_model.keras", vocab_path="data/processed/move_vocab.pkl"):
    # Primero carga el vocabulario para saber cuántos movimientos hay
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    num_policies = len(vocab)
    print(f"Vocabulario cargado: {num_policies} movimientos únicos.")

    # Crea el modelo con el tamaño exacto
    model = create_chess_network(num_policies=num_policies)
    
    # Carga los pesos entrenados
    model.load_weights(model_path)
    print("Pesos cargados correctamente.")

    # Invertir vocabulario: índice → movimiento
    idx_to_move = {idx: move for move, idx in vocab.items()}
    return model, idx_to_move

def predict_move(model, idx_to_move, fen, move_history=[]):
    """
    Dado un FEN, predice el movimiento más probable.
    """
    # 1. Convertir FEN a input (8,8,22)
    x = fen_to_8x8x22(fen, last_moves=move_history)
    x = np.expand_dims(x, axis=0)  # (1, 8, 8, 22)

    # 2. Predicción
    policy_pred, value_pred = model.predict(x, verbose=0)
    policy = policy_pred[0]  # distribución de probabilidad
    value = value_pred[0][0]  # valor estimado (-1 a 1)

    # 3. Obtener movimiento más probable
    top_move_idx = np.argmax(policy)
    best_move_uci = idx_to_move.get(top_move_idx, "??")

    # 4. Mostrar los 5 mejores movimientos
    top_5_indices = np.argsort(policy)[::-1][:5]
    print("Top 5 movimientos predichos:")
    for i, idx in enumerate(top_5_indices):
        move = idx_to_move.get(idx, "??")
        prob = policy[idx]
        print(f"  {i+1}. {move} → {prob:.3f}")

    print(f"\nMejor movimiento: {best_move_uci}")
    print(f"Valor del estado: {value:.3f} (>{0}: favorable, <{0}: desfavorable)")

    return best_move_uci, value

# --- Prueba ---
if __name__ == "__main__":
    model, idx_to_move = load_model_and_vocab()

    # Ejemplo: apertura Siciliana
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    history = ["e2e4", "e7e5"]  # últimos movimientos

    print("FEN:", fen)
    print("Historial:", history)
    print("\nPredicción de la IA:")
    predict_move(model, idx_to_move, fen, history)