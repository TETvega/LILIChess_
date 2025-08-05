# src/move_encoding.py

import numpy as np
from collections import Counter
"""
El vocabulario es necesario para convertir los movimientos UCI a índices y viceversa.
La red neuronal espera que la política sea una distribución de probabilidad sobre los movimientos.
El vocabulario se crea a partir de los movimientos únicos en el dataset.

"""
def create_move_vocab(moves: np.ndarray, min_freq=1) -> dict:
    """
    Crea un vocabulario de movimientos UCI.
    Devuelve: { "e2e4": 0, "e7e5": 1, ... }
    Parámetros:
    moves (np.ndarray): Array de movimientos UCI.
    min_freq (int): Frecuencia mínima para incluir un movimiento en el vocabulario.
    """
    counter = Counter(moves) # Cuenta la frecuencia de cada movimiento
    filtered_moves = [
        move for move, cnt in counter.items() if cnt >= min_freq
        ]  # Filtra por frecuencia mínima
    vocab = {move: idx for idx, move in enumerate(sorted(filtered_moves))} # Crea un diccionario con el movimiento como clave y su índice como valor
    return vocab

# codifica los movimientos a índices según el vocabulario
def encode_moves(moves: np.ndarray, vocab: dict) -> np.ndarray:
    """Convierte movimientos a índices."""
    encoded = [vocab.get(m, -1) for m in moves]
    return np.array(encoded)
# decodifica un índice a movimiento
def decode_move(index: int, vocab: dict) -> str:
    """Convierte índice a movimiento."""
    for move, idx in vocab.items():
        if idx == index:
            return move
    return "??" # Si no se encuentra, devuelve un marcador de error