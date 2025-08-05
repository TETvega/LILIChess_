# tests/test_mcts.py
import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.agent import ChessAgent
from src.mcts import MCTS
import chess

agent = ChessAgent()
mcts = MCTS(agent, num_simulations=400)  # menos para pruebas rápidas

board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
print("FEN:", board.fen())

print("\nEjecutando MCTS (400 simulaciones)...")
policy = mcts.run(board)

print("\nTop 5 movimientos (MCTS):")
for i, (prob, move) in enumerate(policy[:5]):
    print(f"  {i+1}. {move.uci()} → {prob:.3f}")

print(f"\n💡 Mejor movimiento (MCTS): {policy[0][1].uci()}")