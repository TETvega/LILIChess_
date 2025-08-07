# src/board_representation.py
import chess
import numpy as np

def fen_to_8x8x29(fen: str, last_moves: list = None):
    """
    Convierte un FEN a un array 8x8x29.
    Cada plano representa una característica del tablero.
    """
    if last_moves is None:
        last_moves = []
    
    board = chess.Board(fen)
    planes = np.zeros((8, 8, 29), dtype=np.float32)
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    
    # 0-11: Piezas (blancas 0-5, negras 6-11)
    for i in range(8):
        for j in range(8):
            sq = chess.square(j, 7 - i)
            piece = board.piece_at(sq)
            if piece:
                idx = piece_to_index[piece.symbol().upper()]
                if piece.color == chess.WHITE:
                    planes[i, j, idx] = 1
                else:
                    planes[i, j, idx + 6] = 1
    
    # 12: Turno (1 si es blanco)
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1
    
    # 13-16: Enroque
    if board.has_kingside_castling_rights(chess.WHITE): planes[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): planes[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): planes[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): planes[:, :, 16] = 1
    
    # 17: Al paso
    if board.ep_square:
        file = chess.square_file(board.ep_square)
        rank = 7 - chess.square_rank(board.ep_square)
        planes[rank, file, 17] = 1
    
    # 18: Jaque
    if board.is_check():
        planes[:, :, 18] = 1
    
    # 19: 50 movimientos sin progreso
    planes[:, :, 19] = 1 if board.halfmove_clock >= 50 else 0
    
    # 20-21: Últimos 2 movimientos
    recent_moves = last_moves[-2:]
    for k, move_uci in enumerate(recent_moves):
        try:
            move = chess.Move.from_uci(move_uci)
            fr = move.from_square
            to = move.to_square
            i_fr, j_fr = 7 - chess.square_rank(fr), chess.square_file(fr)
            i_to, j_to = 7 - chess.square_rank(to), chess.square_file(to)
            planes[i_fr, j_fr, 20 + k] = 1
            planes[i_to, j_to, 20 + k] = 1
        except:
            pass
    
    # 22: Distancia del rey blanco al centro
    w_king = board.king(chess.WHITE)
    if w_king:
        dist = ((chess.square_file(w_king) - 3.5)**2 + (7 - chess.square_rank(w_king) - 3.5)**2)**0.5
        planes[:, :, 22] = dist / 6
    
    # 23: Distancia del rey negro al centro
    b_king = board.king(chess.BLACK)
    if b_king:
        dist = ((chess.square_file(b_king) - 3.5)**2 + (7 - chess.square_rank(b_king) - 3.5)**2)**0.5
        planes[:, :, 23] = dist / 6
    
    # 24: Control del centro (d4, d5, e4, e5)
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    for sq in center:
        i, j = 7 - chess.square_rank(sq), chess.square_file(sq)
        attackers = len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq))
        planes[i, j, 24] = np.tanh(attackers / 3)
    
    # 25: Estructura de peones (dobles, pasados)
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        # Peón doblado
        if any(chess.square_file(p) == file and chess.square_rank(p) < rank for p in board.pieces(chess.PAWN, chess.WHITE)):
            planes[7 - rank, file, 25] = 0.5
        # Peón pasado
        is_passed = True
        for f in [file-1, file, file+1]:
            if 0 <= f <= 7:
                for r in range(rank+1, 8):
                    if board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color == chess.BLACK:
                        is_passed = False
        if is_passed:
            planes[7 - rank, file, 25] = 1.0
    
    # 26: Piezas desarrolladas (N y B fuera de la fila inicial)
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        if chess.square_rank(sq) > 1:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        if chess.square_rank(sq) > 1:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        if chess.square_rank(sq) < 6:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        if chess.square_rank(sq) < 6:
            planes[7 - chess.square_rank(sq), chess.square_file(sq), 26] = 1
    
    # 27: Mapa de ataques
    for i in range(8):
        for j in range(8):
            sq = chess.square(j, 7 - i)
            diff = len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq))
            planes[i, j, 27] = np.tanh(diff / 5)
    
    # 28: Movilidad (número de movimientos legales por casilla)
    mobility = np.zeros((8, 8))
    for move in board.legal_moves:
        to_sq = move.to_square
        i_to, j_to = 7 - chess.square_rank(to_sq), chess.square_file(to_sq)
        mobility[i_to, j_to] += 1
    planes[:, :, 28] = np.tanh(mobility / 10)
    
    return planes