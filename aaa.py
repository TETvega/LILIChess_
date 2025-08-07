# merge_processed_data_batched.py
import numpy as np
from pathlib import Path
import os

# === CONFIGURACIÓN ===
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = "data/processed/training_data_final.npz"
BATCH_SIZE = 50000  # Procesar en lotes de 50k posiciones

# === CÓDIGO ===
processed_path = Path(PROCESSED_DIR)
npz_files = list(processed_path.glob("temp_*.npz"))  # Solo los temp_*.npz

print(f"🔍 Encontrados {len(npz_files)} archivos .npz. Fusionando por lotes...")

X_batch = []
y_batch = []

for i, npz_file in enumerate(npz_files):
    try:
        print(f"📂 Procesando archivo {i+1}/{len(npz_files)}: {npz_file.name}")
        data = np.load(npz_file)
        X_chunk = data['X']
        y_chunk = data['y']

        # Procesar el chunk en lotes
        for j in range(len(X_chunk)):
            X_batch.append(X_chunk[j])
            y_batch.append(y_chunk[j])

            # Guardar lote cuando se alcanza el tamaño
            if len(X_batch) >= BATCH_SIZE:
                # Guardar el lote
                batch_file = f"{PROCESSED_DIR}/merged_batch_{len(os.listdir(PROCESSED_DIR))}.npz"
                np.savez_compressed(batch_file, X=np.array(X_batch, dtype=np.float32), y=np.array(y_batch))
                print(f"💾 Lote guardado: {batch_file} ({len(X_batch)} posiciones)")
                X_batch.clear()
                y_batch.clear()

    except Exception as e:
        print(f"❌ Error con {npz_file}: {e}")
        continue

# Guardar lo que queda
if X_batch:
    batch_file = f"{PROCESSED_DIR}/merged_batch_final.npz"
    np.savez_compressed(batch_file, X=np.array(X_batch, dtype=np.float32), y=np.array(y_batch))
    print(f"💾 Lote final guardado: {batch_file}")

print("🎉 ¡Fusión por lotes completada! Ahora combina los lotes.")