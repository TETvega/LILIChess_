# src/evaluate_model.py

import sys
import os

# Añade el directorio padre (LILICHEST) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.agent import ChessAgent
from src.mcts import MCTS

def evaluate_models(
    path_old="models/current/best_model.keras",
    path_new="models/current/best_model_v1.keras",
    num_games=10,               #  Ahora son 10 partidas
    num_simulations=150,
    show_moves=False           #  Cambia a True si quieres ver cada movimiento
):
    """
    Evalúa dos modelos: nuevo vs viejo.
    Juega partidas alternando colores.
    Muestra progreso y resultados detallados.
    """
    print("🔍 Iniciando evaluación de modelos...")
    print(f"  Modelo viejo: {path_old}")
    print(f"  Modelo nuevo: {path_new}")
    print(f"  Partidas: {num_games} | Simulaciones: {num_simulations}")
    print("-" * 60)

    # Cargar agentes
    print("Cargando modelo viejo...")
    agent_old = ChessAgent(path_old)
    mcts_old = MCTS(agent_old, num_simulations=num_simulations)

    print("Cargando modelo nuevo...")
    agent_new = ChessAgent(path_new)
    mcts_new = MCTS(agent_new, num_simulations=num_simulations)

    results = {"old": 0, "new": 0, "draw": 0}
    game_details = []  # Guarda detalles de cada partida

    for game_idx in range(num_games):
        print(f"\n🎮 === Partida {game_idx + 1} de {num_games} ===")
        board = chess.Board()
        move_count = 0
        game_moves = []  # Almacena los movimientos UCI

        # Alternar colores: nuevo modelo empieza en partidas pares
        current_mcts = mcts_new if game_idx % 2 == 0 else mcts_old
        opponent_mcts = mcts_old if game_idx % 2 == 0 else mcts_new

        player_names = {
            id(mcts_new): "Nuevo (N)",
            id(mcts_old): "Viejo (V)"
        }

        while not board.is_game_over() and move_count < 150:
            # Obtener nombre del jugador actual
            current_name = player_names.get(id(current_mcts), "Desconocido")

            # Ejecutar MCTS
            policy = current_mcts.run(board)
            best_move = max(policy, key=lambda x: x[0])[1]
            move_uci = best_move.uci()

            # Registrar movimiento
            game_moves.append(move_uci)

            if show_moves:
                print(f"  {move_count + 1:2d}. {current_name} → {move_uci}")

            # Aplicar movimiento
            board.push(best_move)
            move_count += 1

            # Cambiar jugador
            current_mcts, opponent_mcts = opponent_mcts, current_mcts

        # Resultado final
        result = board.result()
        white_wins = result == "1-0"
        black_wins = result == "0-1"

        if (white_wins and game_idx % 2 == 0) or (black_wins and game_idx % 2 == 1):
            winner = "new"
        elif (black_wins and game_idx % 2 == 0) or (white_wins and game_idx % 2 == 1):
            winner = "old"
        else:
            winner = "draw"

        results[winner] += 1

        # Guardar detalles
        game_details.append({
            "game": game_idx + 1,
            "moves": " ".join(game_moves),
            "result": result,
            "winner": winner
        })

        # Mostrar resultado
        print(f"✅ Final: {result}")
        print(f"🏆 Ganador: {'Nuevo' if winner == 'new' else 'Viejo' if winner == 'old' else 'Empate'}")
        print(f"📊 Progreso: {game_idx + 1}/{num_games} partidas jugadas")

    # Resumen final
    print("\n" + "="*60)
    print("🏆 RESULTADO FINAL DE LA EVALUACIÓN")
    print("="*60)
    print(f"Partidas jugadas: {num_games}")
    print(f"Viejo (v1): {results['old']} victorias")
    print(f"Nuevo (v2): {results['new']} victorias")
    print(f"Tablas: {results['draw']}")
    new_is_better = results["new"] > results["old"]
    print("")
    print("🎉 ¡EL NUEVO MODELO ES MEJOR!" if new_is_better else "❌ El nuevo modelo NO es mejor.")
    print("="*60)

    # Opcional: guardar detalles
    import pickle
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/evaluation_details.pkl", "wb") as f:
        pickle.dump(game_details, f)
    print("💾 Detalles de la evaluación guardados en data/evaluation/evaluation_details.pkl")

    return new_is_better


# --- Ejecución ---
if __name__ == "__main__":
    evaluate_models(num_games=5, show_moves=True)  