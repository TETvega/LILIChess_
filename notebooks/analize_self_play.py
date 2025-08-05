# analyze_self_play.py
import pickle
import os

def analyze_data():
    data_dir = "data/self_play"
    files = [f for f in os.listdir(data_dir) if f.startswith("self_play_data_batch_")]

    total_positions = 0
    game_lengths = []
    outcomes = {"white": 0, "black": 0, "draw": 0}

    for file in files:
        path = os.path.join(data_dir, file)
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        total_positions += len(data)
        # Las partidas están mezcladas, pero podemos inferir por el valor
        # O podrías modificar self_play.py para guardar estadísticas

    print(f"📊 Total posiciones generadas: {total_positions}")
    print(f"📁 Archivos procesados: {len(files)}")
    # Aquí puedes añadir más análisis

if __name__ == "__main__":
    analyze_data()