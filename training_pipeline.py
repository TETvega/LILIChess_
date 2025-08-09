# training_pipeline.py
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime
import csv
import psutil
from src.move_encoding import uci_to_flat_index
from models.chess_policy_model import create_policy_model
# === CONFIGURACIÓN ===
PROCESSED_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/chess_policy.keras"
CHECKPOINT_DIR = "models/checkpoints"
PROCESSED_LOG_FILE = "logs/processed_files.txt"
METRICS_CSV = "logs/training_metrics.csv"
LOG_FILE = "logs/training.log"
FILTER_PERF_TYPES = ["blitz", "bullet", "rapid", "classical"]
BATCH_SIZE = 128                    # Aumentado: aprovecha VRAM
EPOCHS = 1                          # Por archivo
GLOBAL_EPOCHS = 2                   # Pasar 2 veces por todos los archivos
LEARNING_RATE = 3e-4
SHUFFLE_BUFFER = 16384              # Mayor gracias a 32 GB RAM
NUM_WORKERS = max(1, psutil.cpu_count() - 2)  # Deja al menos 2 núcleos libres
print(f"🧠 Usando {NUM_WORKERS} hilos (dejando 2 libres)")

ROOT_DIR = Path(__file__).parent
PROCESSED_PATH = (ROOT_DIR / PROCESSED_DATA_DIR).resolve()
MODEL_SAVE_PATH = (ROOT_DIR / MODEL_SAVE_PATH).resolve()
CHECKPOINT_DIR = (ROOT_DIR / CHECKPOINT_DIR).resolve()
for path in [MODEL_SAVE_PATH.parent, CHECKPOINT_DIR, ROOT_DIR / "logs"]:
    path.mkdir(parents=True, exist_ok=True)

# === CONFIGURAR LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TrainingPipeline")


# === 1. Descubrir archivos nuevos ===
def discover_files(data_dir, perf_types_filter=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")
    all_files = list(data_dir.glob("*.npz"))
    logger.info(f"🔍 Descubiertos {len(all_files)} archivos .npz")

    processed_files = set()
    if Path(PROCESSED_LOG_FILE).exists():
        with open(PROCESSED_LOG_FILE, "r", encoding="utf-8") as f:
            processed_files = {line.strip() for line in f if line.strip()}

    filtered_files = []
    perf_types_lower = [pt.lower() for pt in perf_types_filter] if perf_types_filter else None

    for file_path in all_files:
        stem = file_path.stem.lower()
        if "_" not in stem:
            continue
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        perf_type = parts[1]
        if (perf_types_lower is None or perf_type in perf_types_lower) and str(file_path) not in processed_files:
            filtered_files.append(file_path)

    logger.info(f"✅ {len(filtered_files)} archivos nuevos después del filtro")
    return filtered_files


# === 2. Crear dataset seguro (sin from_generator frágil) ===
def create_dataset_from_file(file_path, batch_size):
    """
    Carga un archivo .npz y crea un dataset eficiente.
    No sobrecarga RAM: solo carga este archivo.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        X, moves = data["X"], data["y"]

        # Codificar movimientos válidos
        X_list, y_list = [], []
        for board, move in zip(X, moves):
            idx = uci_to_flat_index(str(move))
            if idx == -1:
                continue
            X_list.append(board)
            y_list.append(idx)

        if len(y_list) == 0:
            logger.warning(f"⚠️  Sin movimientos válidos en {file_path}")
            return None

        X_array = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((X_array, y_array))
        return (
            dataset
            .shuffle(min(SHUFFLE_BUFFER, len(y_list)))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    except Exception as e:
        logger.error(f"❌ Error cargando {file_path}: {e}")
        return None


def perf_type_from_filename(file_path):
    stem = file_path.stem.lower()
    return stem.rsplit("_", 1)[1] if "_" in stem else "desconocido"


# === 3. Guardar como procesado ===
def log_processed_file(file_path, pos_count):
    with open(PROCESSED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{file_path}\n")
    logger.info(f"✅ Archivo registrado como procesado: {file_path.name}")


# === 4. Métricas CSV ===
def init_metrics_csv():
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "global_step", "epoch", "global_epoch", "file_index",
            "loss", "accuracy", "top_5_accuracy", "file_name", "perf_type",
            "positions", "file_size_mb"
        ])


def log_metrics_to_csv(global_step, epoch, global_epoch, file_index, logs, file_path, pos_count):
    file_size = file_path.stat().st_size / (1024 * 1024)
    perf_type = perf_type_from_filename(file_path)
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            global_step, epoch, global_epoch, file_index,
            f"{logs.get('loss', 0):.4f}",
            f"{logs.get('sparse_categorical_accuracy', 0):.4f}",
            f"{logs.get('top_5_accuracy', 0):.4f}",
            file_path.name,
            perf_type,
            pos_count,
            f"{file_size:.2f}"
        ])


# === 5. Callback con métricas mejoradas ===
class FileCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path, file_index, pos_count, global_epoch):
        super().__init__()
        self.file_path = file_path
        self.file_index = file_index
        self.pos_count = pos_count
        self.global_epoch = global_epoch
        self.global_step = 0

    def on_epoch_end(self, epoch, logs=None):
        n_samples = self.pos_count
        steps_this_epoch = max(1, n_samples // BATCH_SIZE)
        self.global_step += steps_this_epoch

        log_metrics_to_csv(
            global_step=self.global_step,
            epoch=epoch,
            global_epoch=self.global_epoch,
            file_index=self.file_index,
            logs=logs,
            file_path=self.file_path,
            pos_count=self.pos_count
        )

        safe_name = self.file_path.stem.replace(" ", "_")
        self.model.save(CHECKPOINT_DIR / f"model_after_{safe_name}_epoch{epoch}_global{self.global_epoch}.keras")
        self.model.save(CHECKPOINT_DIR / "model_checkpoint_latest.keras")

        logger.info(f"💾 Checkpoint guardado | loss: {logs.get('loss'):.4f}, "
                    f"acc: {logs.get('sparse_categorical_accuracy'):.4f}")


# === 6. Métrica personalizada que maneja mixed precision ===
@tf.function
def top_5_accuracy_fixed(y_true, y_pred):
    y_pred_float32 = tf.cast(y_pred, tf.float32)
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred_float32, k=5)


# === 7. Entrenamiento incremental con múltiples épocas globales ===
def main():
    logger.info("🚀 Iniciando pipeline de entrenamiento incremental")

    # === Mixed Precision ===
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info(f"🎯 Política de precisión: {tf.keras.mixed_precision.global_policy()}")

    if not Path(METRICS_CSV).exists():
        init_metrics_csv()

    file_paths = discover_files(PROCESSED_PATH, FILTER_PERF_TYPES)
    if not file_paths:
        logger.info("✅ No hay nuevos archivos para procesar.")
        return

    # === Cargar o crear modelo ===
    model_path = CHECKPOINT_DIR / "model_checkpoint_latest.keras"
    if model_path.exists():
        logger.info(f"🔁 Cargando modelo desde: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"top_5_accuracy_fixed": top_5_accuracy_fixed}
        )
    else:
        logger.info("🆕 Creando nuevo modelo...")
        model = create_policy_model(input_shape=(8, 8, 29))

    # === Optimizador con escalado ===
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # === Compilar modelo ===
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'sparse_categorical_accuracy',
            top_5_accuracy_fixed
        ]
    )

    # === Entrenar con múltiples épocas globales ===
    total_positions = 0
    for global_epoch in range(GLOBAL_EPOCHS):
        logger.info(f"🔄 Iniciando época global {global_epoch + 1}/{GLOBAL_EPOCHS}")

        for idx, file_path in enumerate(file_paths):
            logger.info(f"📦 [{idx+1}/{len(file_paths)}] Procesando: {file_path.name}")

            try:
                data = np.load(file_path, allow_pickle=True)
                if "X" not in data or "y" not in data:
                    logger.error(f"❌ {file_path}: faltan 'X' o 'y'")
                    continue
                n_samples = len(data["X"])
                if n_samples == 0:
                    logger.error(f"❌ {file_path}: Sin muestras")
                    continue
                steps_per_epoch = max(1, n_samples // BATCH_SIZE)
                logger.info(f"📁 {n_samples} posiciones | steps_per_epoch: {steps_per_epoch}")
            except Exception as e:
                logger.error(f"❌ Error al cargar {file_path}: {e}")
                continue

            # Crear dataset
            dataset = create_dataset_from_file(file_path, BATCH_SIZE)
            if dataset is None:
                logger.error(f"❌ Dataset vacío para {file_path}")
                continue

            # Diagnóstico
            try:
                for batch in dataset.take(1):
                    x, y = batch
                    logger.info(f"🔧 Batch OK: entrada {x.shape}, etiqueta {y.shape}, ej: {y[0].numpy()}")
                    break
                else:
                    logger.error(f"❌ Dataset vacío tras creación: {file_path}")
                    continue
            except Exception as e:
                logger.error(f"❌ Error al leer dataset {file_path}: {e}")
                continue

            # Entrenar
            try:
                callback = FileCheckpointCallback(file_path, idx, n_samples, global_epoch)
                history = model.fit(
                    dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[callback, tf.keras.callbacks.TerminateOnNaN()],
                    verbose=1
                )
                total_positions += n_samples
                log_processed_file(file_path, n_samples)
                model.save(model_path)
                logger.info(f"✅ Finalizado con {file_path.name} | Total entrenado: {total_positions}")

            except Exception as e:
                logger.error(f"💥 Error entrenando con {file_path}: {str(e)}", exc_info=True)
                continue

    # === Guardar modelo final ===
    model.save(MODEL_SAVE_PATH)
    logger.info(f"🎉 Entrenamiento completado. Modelo guardado en {MODEL_SAVE_PATH}")
    logger.info(f"📊 Total de posiciones entrenadas: {total_positions} (x{GLOBAL_EPOCHS} épocas)")


if __name__ == "__main__":
    main()