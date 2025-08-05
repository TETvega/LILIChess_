# src/training.py

import sys
import os
import pickle
import numpy as np
import tensorflow as tf

# Añade el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa tus módulos
from src.neuronal_network import create_chess_network
from src.move_encoding import create_move_vocab, encode_moves

def train_supervised(
    data_path="data/processed/training_data.npz",
    model_save_path="models/current/best_model.keras"  # ✅ Formato moderno
):
    """
    Entrena el modelo de ajedrez en modo supervisado.
    """
    print("🚀 Iniciando entrenamiento supervisado...")

    # --- 1. Cargar datos ---
    print("📂 Cargando datos...")
    try:
        data = np.load(data_path)
        X = data["X"]  # (N, 8, 8, 22)
        y_str = data["y"]  # (N,) strings UCI
        print(f"✅ Datos cargados: X.shape = {X.shape}, y.size = {y_str.size}")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return

    # --- 2. Crear vocabulario de movimientos ---
    print("🔤 Creando vocabulario de movimientos...")
    try:
        vocab = create_move_vocab(y_str)
        y_idx = encode_moves(y_str, vocab)
        num_policies = len(vocab)
        
        # Guardar vocabulario
        vocab_dir = "data/processed"
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "move_vocab.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"✅ Vocabulario guardado: {num_policies} movimientos únicos en {vocab_path}")
    except Exception as e:
        print(f"❌ Error al crear vocabulario: {e}")
        return

    # --- 3. Preparar etiquetas ---
    print("🎯 Preparando etiquetas...")
    y_policy = tf.keras.utils.to_categorical(y_idx, num_classes=num_policies)
    y_value = np.zeros(len(X), dtype=np.float32)  # Temporal: valor del estado

    # --- 4. Crear modelo ---
    print("🧠 Creando modelo de red neuronal...")
    try:
        model = create_chess_network(input_shape=(8, 8, 22), num_policies=num_policies)
        print("✅ Modelo creado.")
    except Exception as e:
        print(f"❌ Error al crear el modelo: {e}")
        return

    # --- 5. Callbacks ---
    print("⚙️  Configurando callbacks...")
    os.makedirs("models/current", exist_ok=True)

    callbacks = [
        # Guarda el mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            verbose=1
        ),
        # Reduce el learning rate si no mejora
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Detiene si no hay mejora
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --- 6. Entrenar ---
    print("🔥 Entrenando modelo...")
    try:
        history = model.fit(
            X, [y_policy, y_value],
            epochs=80,           # ✅ 80 épocas
            batch_size=128,      # ✅ 128 (aprovecha la GPU)
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        print(f"Entrenamiento completado. Mejor modelo guardado en {model_save_path}")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")


# --- Ejecución ---
if __name__ == "__main__":
    train_supervised()