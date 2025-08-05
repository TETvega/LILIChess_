chess-alpha-zero/
│
├── data/                          # Datos de partidas y entrenamiento
│   ├── raw/                       # PGNs descargados (partidas reales)
│   │   └── lichess_db_standard_rated_2023.pgn
│   │
│   ├── processed/                 # Datos procesados (estados, movimientos, etc.)
│   │   └── training_data_v1.pkl
│   │
│   └── self_play/                 # Partidas generadas por la IA (auto-aprendizaje)
│       └── games_20250405.pkl
│
├── models/                        # Modelos guardados
│   ├── checkpoints/               # Checkpoints durante entrenamiento
│   │   └── model_epoch_10.h5
│   │
│   ├── best/                      # Mejor modelo según evaluación
│   │   └── best_model.h5
│   │
│   └── current/                   # Modelo actual en uso
│       └── current_model.h5
│
├── src/                           # Código fuente principal
│   ├── board_representation.py    # Codificación del tablero (8x8xplanes)
│   ├── data_loader.py             # Carga y procesamiento de PGNs
│   ├── neural_network.py          # Definición del modelo (TensorFlow)
│   ├── mcts.py                    # Implementación de Monte Carlo Tree Search
│   ├── agent.py                   # La IA: toma decisiones usando MCTS + modelo
│   ├── training.py                # Entrenamiento (supervisado y por refuerzo)
│   ├── self_play.py               # Generación de partidas IA vs IA
│   ├── evaluation.py              # Evaluar IA vs versión anterior o Stockfish
│   └── utils.py                   # Funciones auxiliares (FEN a estado, etc.)
│
├── gameplay/                      # Interfaz y juego contra humano
│   ├── human_vs_ai.py             # Juego interactivo (Pygame o Tkinter)
│   ├── gui/                       # Assets de interfaz (opcional)
│   │   └── chess_pieces/
│   └── move_validator.py          # Validar movimientos del humano
│
├── config/                        # Archivos de configuración
│   ├── config.json                # Hiperparámetros (simulaciones, épocas, etc.)
│   └── hyperparameters.py         # O como módulo Python
│
├── logs/                          # Registros de entrenamiento, errores, etc.
│   ├── training.log
│   └── self_play.log
│
├── tests/                         # Pruebas unitarias
│   ├── test_board.py
│   ├── test_mcts.py
│   └── test_model.py
│
├── notebooks/                     # Notebooks para exploración (Jupyter)
│   ├── exploratory_data_analysis.ipynb
│   └── model_training_debug.ipynb
│
├── requirements.txt               # Dependencias de Python
│
└── main.py                        # Punto de entrada principal (entrenar, jugar, etc.)

