# src/board_representation.py

import chess
import numpy as np

"""
8X8X22 represetacion de tablero:
- 8x8: tablero
- 22 planos:
    - 12 planos de piezas (6 blancas, 6 negras)
    - 1 plano de turno (blancas)
    - 4 planos de enroque (2 blancas, 2 negras)
    - 1 plano de al paso (captura )
    - 1 plano de jaque
    - 1 plano de movimientos sin progreso (50 movimientos)
    - 2 planos de últimos movimientos (como "fuego")
    
"""



def fen_to_8x8x22(fen: str, last_moves: list = None):
    """
    Convierte FEN a un array 8x8x22.
    
    Parámetros:
    - fen: estado actual
    - last_moves: lista de movimientos UCI recientes (opcional, últimos 2)
    """
    board = chess.Board(fen)
    if last_moves is None:
        last_moves = []

    planes = np.zeros((8, 8, 22), dtype=np.float32)

    # Mapeo de piezas a índices
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

    # Llenar planos de piezas (blancas: 0-5, negras: 6-11)
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7 - i)  # FEN: fila 0 = 8ª del tablero
            piece = board.piece_at(square)
            if piece:
                idx = piece_to_index[piece.symbol().upper()]
                if piece.color == chess.WHITE:
                    planes[i, j, idx] = 1
                else:
                    planes[i, j, idx + 6] = 1

    # Plano 12: turno de blancas
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1

    # Planos 13-16: enroques
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, 16] = 1

    # Plano 17: al paso
    if board.ep_square:
        file = chess.square_file(board.ep_square)
        rank = 7 - chess.square_rank(board.ep_square)  # ajuste a índice 0-7
        planes[rank, file, 17] = 1

    # Plano 18: en jaque
    if board.is_check():
        planes[:, :, 18] = 1

    # Plano 19: movimientos sin progreso (50 movimientos)
    halfmove_clock = board.halfmove_clock
    planes[:, :, 19] = 1 if halfmove_clock >= 4 else 0  # binarizado

    # Planos 20-21: últimos 2 movimientos 
    recent_moves = last_moves[-2:]  # últimos 2
    for k, move_uci in enumerate(recent_moves):
        try:
            move = chess.Move.from_uci(move_uci)
            from_sq = move.from_square
            to_sq = move.to_square

            i_from = 7 - chess.square_rank(from_sq)
            j_from = chess.square_file(from_sq)
            i_to = 7 - chess.square_rank(to_sq)
            j_to = chess.square_file(to_sq)

            planes[i_from, j_from, 20 + k] = 1
            planes[i_to, j_to, 20 + k] = 1
        except:
            pass  # si el movimiento es inválido, ignora

    return planes