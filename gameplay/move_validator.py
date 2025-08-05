# gameplay/move_validator.py

import chess
"""
Clase para validar movimientos de ajedrez.
Usa python-chess para verificar movimientos UCI en un tablero dado por FEN.
Incluye validación de movimientos, verificación de jaque/jaque mate, y obtención de movimientos legales.
"""
class MoveValidator:
    """
    Validador robusto de movimientos de ajedrez.
    Usa python-chess para verificar movimientos UCI en un tablero dado por FEN.
    """
    @staticmethod
    def is_valid_move(fen: str, uci_move: str) -> bool:
        """
        Verifica si un movimiento UCI es válido en el estado del tablero dado por FEN.

        Args:
            fen (str): Estado del tablero en formato FEN.
            uci_move (str): Movimiento en formato UCI
        Returns:
            bool: True si el movimiento es legal, False en caso contrario.
        """
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci_move.strip())

            return move in board.legal_moves

        except ValueError as e:
            # Movimiento UCI mal formado (ejemplo longitud incorrecta)
            return False
        except Exception:
            # Cualquier otro error
            return False

    @staticmethod
    def validate_move_with_reason(fen: str, uci_move: str) -> tuple[bool, str]:
        """
        Verifica el movimiento y devuelve una razón detallada si es inválido.
        Args:
            fen (str): Estado del tablero en formato FEN.
            uci_move (str): Movimiento en formato UCI.

        Returns:
            tuple[bool, str]: (es_válido, razón)
        """
        try:
            board = chess.Board(fen)
        except Exception:
            return False, "FEN invalido"

        try:
            uci_move = uci_move.strip()
            if not uci_move:
                return False, "Movimiento vacio"
            move = chess.Move.from_uci(uci_move)
        except ValueError:
            return False, "Movimiento UCI mal formado (ej: debe ser 'e2e4' o 'e7e8q')"

        if move not in board.legal_moves:
            # Verificar si el movimiento es legal
            if board.is_check() and board.is_checkmated():
                return False, "Jaque mate: no hay movimientos legales"
            if board.is_stalemate():
                return False, "Ahogado: no hay movimientos legales"
            if board.is_repetition():
                return False, "Posición repetida (3 veces)"
            if board.is_insufficient_material():
                return False, "Material insuficiente para dar mate"
            return False, "Movimiento ilegal en esta posición"

        # Verificar si es jaque o jaque mate después del movimiento
        board.push(move)
        if board.is_check():
            if board.is_checkmate():
                return True, "[ MATE ]  Movimiento valido -> jaque mate"
            return True, "[ JAQUE ] Movimiento valido -> da jaque"
        if board.is_stalemate():
            return True, "[ TABLAS -> AHOGADO ] Movimiento valido -> provoca ahogado"
        return True, "Movimiento valido"

    @staticmethod
    def get_legal_moves(fen: str) -> list[str]:
        """
        Devuelve la lista de movimientos legales en formato UCI.
        Args:
            fen (str): Estado del tablero en formato FEN.
        Returns:
            list[str]: Lista de movimientos legales en formato UCI.
        """
        try:
            board = chess.Board(fen)
            return [move.uci() for move in board.legal_moves]
        except Exception:
            return []

    @staticmethod
    def is_game_over(fen: str) -> tuple[bool, str]:
        """
        Verifica si la partida ha terminado.
        Args:
            fen (str): Estado del tablero en formato FEN.

        Returns:
            tuple[bool, str]: (terminó, motivo)
        """
        try:
            board = chess.Board(fen)
            if board.is_checkmate():
                winner = "Blancas" if board.turn == chess.BLACK else "Negras"
                return True, f"[ MATE ] Jaque mate - Ganan {winner}"
            if board.is_stalemate():
                return True, "[ TABLAS -> Ahogado ]Ahogado"
            if board.is_insufficient_material():
                return True, "[ TABLAS -> Material Ins ]Material insuficiente"
            if board.is_seventyfive_moves():
                return True, "[ TABLAS -> 75 movs no jaque mate ]Regla de 75 movimientos"
            if board.is_fivefold_repetition():
                return True, "[ TABLAS -> REPETICION ]Repetición quintuple"
            return False, "La partida continúa"
        except Exception:
            return True, "Estado invalido"


if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"

    # Validación simple
    if MoveValidator.is_valid_move(fen, uci_move):
        print("Movimiento valido")

    # Con razón
    es_valido, razon = MoveValidator.validate_move_with_reason(fen, uci_move)
    print(f"Resultado: {razon}")

    # Movimientos legales
    legales = MoveValidator.get_legal_moves(fen)
    print(f"Hay {len(legales)} movimientos legales")

    # ¿Terminó la partida?
    terminado, motivo = MoveValidator.is_game_over(fen)
    print(f"Termino la partida {terminado} - {motivo}")