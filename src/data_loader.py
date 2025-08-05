# src/data_loader.py
import chess.pgn
import os

def load_games_from_pgn(pgn_path, max_games=1000):
    """
    Lee partidas de un archivo PGN.
    Devuelve una lista de partidas (cada una es una secuencia de movimientos).
    """
    games = []
    with open(pgn_path) as f:
        for i in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
            if i % 100 == 0:
                print(f"Partida {i} cargada...")
    return games



    