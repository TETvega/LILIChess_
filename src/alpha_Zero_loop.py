# src/alpha_zero_loop.py

import os
import sys
import shutil
import pickle
from datetime import datetime
import logging

# Añade el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/alpha_zero.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Módulos
from src.selft_play import run_self_play_parallel
from src.train_from_self_play import train_from_self_play
from notebooks.evaluate_models import evaluate_models

# Directorios
os.makedirs("logs", exist_ok=True)
os.makedirs("models/best", exist_ok=True)
os.makedirs("models/backup", exist_ok=True)

# Rutas
CURRENT_MODEL = "models/current/best_model.keras"
TEMP_NEW_MODEL = "models/current/best_model_new.keras"
BACKUP_MODEL = f"models/backup/best_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"


def backup_current_model():
    """Guarda una copia del modelo actual antes de sobrescribirlo."""
    if os.path.exists(CURRENT_MODEL):
        shutil.copy(CURRENT_MODEL, BACKUP_MODEL)
        logger.info(f"Modelo actual respaldado: {BACKUP_MODEL}")
    else:
        logger.warning("No se encontró modelo actual para respaldar.")


def save_evaluation_results(results, cycle):
    """Guarda los resultados de la evaluación."""
    results_path = "data/evaluation/evaluation_results.pkl"
    os.makedirs("data/evaluation", exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Resultados de evaluación guardados: {results_path}")


def run_alpha_zero_cycle(cycle_num):
    """
    Ejecuta un ciclo completo de auto-aprendizaje.
    """
    start_time = datetime.now()
    logger.info(f"🔁 INICIANDO CICLO {cycle_num} | {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # --- 1. Self-play ---
        logger.info("1️⃣ Generando partidas con self-play...")
        self_play_data = run_self_play_parallel(num_games=30)  # Ajustado para tu GPU

        # --- 2. Entrenar nuevo modelo ---
        logger.info("2️⃣ Entrenando nuevo modelo con datos de self-play...")
        train_from_self_play(model_save_path=TEMP_NEW_MODEL)

        # --- 3. Evaluar: nuevo vs viejo ---
        logger.info("3️⃣ Evaluando nuevo modelo vs el actual...")
        is_better = evaluate_models(
            path_old=CURRENT_MODEL,
            path_new=TEMP_NEW_MODEL,
            num_games=10
        )

        # --- 4. Reemplazar si es mejor ---
        if is_better:
            logger.info("🏆 ¡El nuevo modelo es mejor! Actualizando...")
            backup_current_model()
            shutil.copy(TEMP_NEW_MODEL, CURRENT_MODEL)
            # Guardar como mejor versión histórica
            shutil.copy(CURRENT_MODEL, f"models/best/best_model_cycle_{cycle_num}.keras")
            logger.info(f"✅ Modelo actualizado: {CURRENT_MODEL}")
        else:
            logger.info("❌ El nuevo modelo no es mejor. Manteniendo el anterior.")

        # --- 5. Limpiar modelo temporal ---
        if os.path.exists(TEMP_NEW_MODEL):
            os.remove(TEMP_NEW_MODEL)

        # --- 6. Resumen del ciclo ---
        duration = datetime.now() - start_time
        logger.info(f"✅ Ciclo {cycle_num} completado. Duración: {duration}")

    except Exception as e:
        logger.error(f"❌ Error en el ciclo {cycle_num}: {e}")
        if os.path.exists(TEMP_NEW_MODEL):
            os.remove(TEMP_NEW_MODEL)


# --- Ejecución ---
if __name__ == "__main__":
    logger.info("🚀 Iniciando ciclo AlphaZero completo")
    
    for cycle in range(1, 6):  # 5 ciclos
        run_alpha_zero_cycle(cycle)
    
    logger.info("🎉 ENTRENAMIENTO COMPLETADO")