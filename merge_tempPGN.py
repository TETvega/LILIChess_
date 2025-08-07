# merge_processed_data.py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

# === CONFIGURACIÓN ===
PROCESSED_DIR = "data/processed"          # Carpeta con los .npz
OUTPUT_FILE = "data/evaluation/training_data_final.npz"
LOGS_DIR = "logs"
BATCH_SIZE = 50000                        # Procesar en lotes de 50k posiciones
TEMP_MERGE_LOG = "logs/.merge_progress.txt"  # Archivo de checkpoint

# Crear carpetas
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# === CONFIGURACIÓN DE LOGS ===
log_file = Path(LOGS_DIR) / "merge_progress.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# === FUNCIONES ===
def load_in_batches(npz_path, batch_size=50000):
    """Carga un archivo .npz en lotes pequeños usando memoria mapeada."""
    try:
        data = np.load(npz_path, mmap_mode='r')
        X = data['X']
        y = data['y']
        num_samples = len(X)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield X[start:end].copy(), y[start:end]  # .copy() para cargar en RAM
    except Exception as e:
        logger.error(f"Error al cargar {npz_path}: {e}")
        yield None, None

def save_checkpoint(processed_files):
    """Guarda la lista de archivos ya procesados."""
    with open(TEMP_MERGE_LOG, 'w', encoding='utf-8') as f:
        for f_path in processed_files:
            f.write(f"{f_path}\n")

def load_checkpoint():
    """Carga la lista de archivos ya procesados."""
    if os.path.exists(TEMP_MERGE_LOG):
        with open(TEMP_MERGE_LOG, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

# === FUSIÓN PRINCIPAL ===
def merge_all_npz():
    processed_path = Path(PROCESSED_DIR)
    npz_files = sorted(list(processed_path.glob("temp_*.npz")))  # Orden alfabético

    if not npz_files:
        logger.warning("❌ No se encontraron archivos .npz en data/processed")
        return

    logger.info(f"🔍 Encontrados {len(npz_files)} archivos .npz. Iniciando fusión...")

    # Cargar progreso anterior
    processed_log = load_checkpoint()
    remaining_files = [f for f in npz_files if str(f) not in processed_log]

    logger.info(f"📌 {len(processed_log)} archivos ya procesados. Quedan {len(remaining_files)} por procesar.")

    X_batch = []
    y_batch = []
    total_positions = 0

    # Barra de progreso para los archivos
    for npz_file in tqdm(remaining_files, desc="Fusionando archivos", unit="archivo"):
        try:
            logger.info(f"📦 Procesando: {npz_file.name}")
            for X_chunk, y_chunk in load_in_batches(npz_file, BATCH_SIZE):
                if X_chunk is None or len(X_chunk) == 0:
                    continue

                X_batch.append(X_chunk)
                y_batch.append(y_chunk)
                total_positions += len(X_chunk)

                # Guardar lote si se alcanza el tamaño
                if sum(len(x) for x in X_batch) >= BATCH_SIZE * 2:
                    X_save = np.concatenate(X_batch, axis=0)
                    y_save = np.concatenate(y_batch, axis=0)
                    temp_batch_file = f"{PROCESSED_DIR}/batch_temp_merge_{total_positions // 100000}.npz"
                    np.savez_compressed(temp_batch_file, X=X_save, y=y_save)
                    X_batch.clear()
                    y_batch.clear()
                    logger.debug(f"💾 Lote temporal guardado: {temp_batch_file}")

            # Marcar como procesado
            save_checkpoint(processed_log.union({str(npz_file)}))
            processed_log.add(str(npz_file))

        except Exception as e:
            logger.error(f"❌ Error crítico con {npz_file}: {e}")
            continue

    # Guardar lo que queda
    if X_batch:
        X_save = np.concatenate(X_batch, axis=0)
        y_save = np.concatenate(y_batch, axis=0)
        np.savez_compressed(OUTPUT_FILE, X=X_save, y=y_save)
        logger.info(f"💾 Dataset final guardado: {OUTPUT_FILE}")
    else:
        logger.warning("⚠️ No se generaron datos. Verifica los archivos de entrada.")

    # Limpiar checkpoint
    if os.path.exists(TEMP_MERGE_LOG):
        os.remove(TEMP_MERGE_LOG)

    logger.info(f"🎉 ¡Fusión completada con éxito!")
    logger.info(f"📌 Total de posiciones guardadas: {total_positions:,}")
    logger.info(f"💾 Archivo final: {OUTPUT_FILE}")

# === EJECUCIÓN ===
if __name__ == "__main__":
    try:
        merge_all_npz()
    except KeyboardInterrupt:
        logger.info("🛑 Proceso interrumpido por el usuario. Se guardó el progreso.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"💥 Error fatal: {e}")
        sys.exit(1)