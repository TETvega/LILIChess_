# src/agent.py

import sys
import os
import chess  # Asegúrate de tenerlo

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from tensorflow.keras.models import load_model 
from src.board_representation import fen_to_8x8x22

class ChessAgent:
    """
    Agente de ajedrez basado en un modelo tipo AlphaZero.
    Usa el modelo para predecir política (movimiento) y valor (evaluación).
    """
    
    def __init__(self, model_path="models/current/best_model.keras", vocab_path="data/processed/move_vocab.pkl"):
        # Cargar modelo
        print(f"Cargando modelo desde {model_path}...")
        self.model = load_model(model_path)
        print("Modelo cargado.")
        
        # Cargar vocabulario de movimientos
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.idx_to_move = {idx: move for move, idx in self.vocab.items()}
        self.move_to_idx = self.vocab  # para búsquedas rápidas
        print(f"Vocabulario cargado: {len(self.vocab)} movimientos únicos.")
    
    def predict(self, fen, move_history=[]):
        """
        Predice la política (distribución de movimientos) y el valor del estado.
        
        Args:
            fen (str): Estado actual del tablero.
            move_history (list): Últimos movimientos (para contexto).
            
        Returns:
            tuple: (policy_dict, value) 
                   policy_dict = { "e2e4": 0.42, "g1f3": 0.35, ... }
                   value = float entre -1 y 1
        """
        # Convertir FEN a array (8,8,22)
        board_array = fen_to_8x8x22(fen, last_moves=move_history)
        x = np.expand_dims(board_array, axis=0)  # (1, 8, 8, 22)
        
        # Predicción
        policy_pred, value_pred = self.model.predict(x, verbose=0)
        policy = policy_pred[0]  # distribución de probabilidad
        value = value_pred[0][0]  # valor estimado
        
        # Convertir a diccionario: movimiento → probabilidad
        policy_dict = {}
        for idx, prob in enumerate(policy):
            move = self.idx_to_move.get(idx, None)
            if move:
                policy_dict[move] = float(prob)
        
        return policy_dict, value
    
    def get_best_move(self, fen, move_history=[], top_k=1):
        """
        Devuelve el(s) movimiento(s) más probable(s).
        Aplica penalización a movimientos que repiten posiciones.
        """
        # Crear tablero desde el FEN
        board = chess.Board(fen)
        
        # Obtener política del modelo
        policy_dict, value = self.predict(fen, move_history)
        
        # Penalizar movimientos que repiten posiciones
        for move_uci in list(policy_dict.keys()):
            move = chess.Move.from_uci(move_uci)
            if board.is_legal(move):
                board.push(move)
                # Si esta posición ya se ha repetido antes
                if board.is_repetition(count=2):
                    policy_dict[move_uci] *= 0.1  # Penalización fuerte
                elif board.is_repetition(count=1):
                    policy_dict[move_uci] *= 0.5  # Penalización leve
                board.pop()
        
        # Ordenar por probabilidad
        sorted_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_moves[:top_k], value


# --- Prueba ---
if __name__ == "__main__":
    agent = ChessAgent()
    
    # Ejemplo: Siciliana abierta
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    history = ["e2e4", "e7e5"]
    
    print("FEN:", fen)
    print("Historial:", history)
    
    best_moves, value = agent.get_best_move(fen, history, top_k=5)
    
    print("\nTop 5 movimientos predichos:")
    for i, (move, prob) in enumerate(best_moves):
        print(f"  {i+1}. {move} → {prob:.3f}")
    
    print(f"\nMejor movimiento: {best_moves[0][0]}")
    print(f"Valor del estado: {value:.3f}")