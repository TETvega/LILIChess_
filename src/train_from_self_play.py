# src/train_from_self_play.py

import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle
import numpy as np
import tensorflow as tf
from src.neuronal_network import create_chess_network
from src.board_representation import fen_to_8x8x22

def load_self_play_data(data_path="data/self_play/self_play_data_full.pkl"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def train_from_self_play(
    model_save_path="models/current/best_model_v1.keras",
    vocab_path="data/processed/move_vocab.pkl"
):
    print("Cargando datos de self-play...")
    data = load_self_play_data()

    # Cargar vocabulario
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    num_policies = len(vocab)

    X, y_policy, y_value = [], [], []

    print("Procesando datos...")
    for item in data:
        fen = item["fen"].split()[0]  # solo piezas
        board_array = fen_to_8x8x22(fen, last_moves=[])
        X.append(board_array)

        # Política: vector one-hot ponderado por MCTS
        policy_vec = np.zeros(num_policies)
        for move_uci, prob in item["policy"].items():
            if move_uci in vocab:
                policy_vec[vocab[move_uci]] = prob
        y_policy.append(policy_vec)

        # Valor
        y_value.append(item["value"])

    X = np.array(X)
    y_policy = np.array(y_policy)
    y_value = np.array(y_value)

    print(f"Datos listos: X.shape = {X.shape}")

    # Crear modelo
    model = create_chess_network(input_shape=(8, 8, 22), num_policies=num_policies)

    # Callbacks
    os.makedirs("models/current", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Entrenar
    print("Entrenando con datos de self-play...")
    model.fit(
        X, [y_policy, y_value],
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print(f"✅ Modelo entrenado y guardado en {model_save_path}")


# --- Ejecución ---
if __name__ == "__main__":
    train_from_self_play()