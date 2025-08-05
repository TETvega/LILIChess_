# combine_self_play.py
import pickle

import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rutas
data_dir = "data/self_play"
output_path = "data/self_play/self_play_data_combined.pkl"

# Archivos a combinar
files = [
    "self_play_data_batch_0.pkl",
    "self_play_data_batch_1.pkl",
    "self_play_data_batch_2.pkl"
]

all_data = []
for file in files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        with open(path, "rb") as f:
            batch_data = pickle.load(f)
            all_data.extend(batch_data)
        print(f"✅ Cargado: {file} ({len(batch_data)} posiciones)")

# Guardar combinado
with open(output_path, "wb") as f:
    pickle.dump(all_data, f)

print(f"✅ Datos combinados guardados en {output_path}")
print(f"📊 Total de posiciones: {len(all_data)}")