# merge_final_batches.py
import numpy as np
from pathlib import Path
import logging
import os
import sys
# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === CONFIGURACIÓN ===
BATCH_DIR = "data/processed"                    # Carpeta con los lotes intermedios
OUTPUT_FILE = "data/evaluation/training_data_final.npz"  # Ruta corregida
LOGS_DIR = "logs"
TEMP_MERGE_LOG = "logs/.final_merge_done.txt"   # Checkpoint

# Crear carpetas
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# Configurar logs
log_file = Path(LOGS_DIR) / "final_merge.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FUNCIONES ===
def load_safe(npz_path):
    """Carga un .npz de forma segura."""
    try:
        data = np.load(npz_path)
        X = data['X']
        y = data['y']
        logger.info(f"✅ Cargado: {npz_path.name} | X: {X.shape}, y: {len(y)}")
        return X, y
    except Exception as e:
        logger.error(f"❌ Error con {npz_path}: {e}")
        return None, None

def save_checkpoint():
    """Marca que la fusión final se completó."""
    with open(TEMP_MERGE_LOG, 'w', encoding='utf-8') as f:
        f.write("completed\n")

def has_completed():
    """Verifica si ya se hizo la fusión final."""
    return os.path.exists(TEMP_MERGE_LOG)

# === FUSIÓN FINAL ===
def merge_final():
    if has_completed():
        logger.info("🎉 La fusión final ya se completó. Nada que hacer.")
        return

    batch_files = sorted(Path(BATCH_DIR).glob("batch_temp_merge_*.npz"))
    if not batch_files:
        logger.error("❌ No se encontraron archivos 'batch_temp_merge_*.npz'")
        return

    logger.info(f"🔍 Fusionando {len(batch_files)} lotes intermedios...")

    X_list = []
    y_list = []

    for batch_file in batch_files:
        X, y = load_safe(batch_file)
        if X is not None and len(X) > 0:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        logger.error("❌ No se cargó ningún dato. Verifica los archivos.")
        return

    # Concatenar todo
    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)

    # Guardar con compresión
    np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final)
    save_checkpoint()

    logger.info(f"🎉 ¡Fusión final completada con éxito!")
    logger.info(f"📌 Total de posiciones: {len(X_final):,}")
    logger.info(f"💾 Archivo final guardado en: {OUTPUT_FILE}")
    logger.info(f"📦 Tamaño estimado: ~{len(X_final) * 7.4 / 1e6:.1f} MB")

# === EJECUCIÓN ===
if __name__ == "__main__":
    try:
        merge_final()
    except KeyboardInterrupt:
        logger.info("🛑 Proceso interrumpido por el usuario.")
    except Exception as e:
        logger.critical(f"💥 Error fatal: {e}")