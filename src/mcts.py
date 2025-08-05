# src/mcts.py

import numpy as np
import chess
import chess.polyglot 

from typing import Dict, List, Tuple

EPS = 1e-8

class Node:
    __slots__ = ['board', 'parent', 'move', 'children', 'N', 'W', 'P', 'is_expanded']

    def __init__(self, board: chess.Board, parent=None, move: chess.Move = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.N = 0
        self.W = 0.0
        self.P = 0.0
        self.is_expanded = False

    def ucb_score(self, cpuct: float = 1.0):
        if self.parent is None:
            return 0.0
        Q = self.W / (self.N + EPS)
        U = cpuct * self.P * np.sqrt(self.parent.N) / (self.N + 1)
        return Q + U

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb_score())

    def expand(self, policy: Dict[str, float], board_occurrences: Dict[str, int]):
        """
        Expande el nodo, pero penaliza movimientos que repiten posiciones.
        """
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            uci = move.uci()
            prior = policy.get(uci, 1e-10)

            # Copia el tablero y aplica el movimiento
            child_board = self.board.copy(stack=False)
            child_board.push(move)
            zobrist = chess.polyglot.zobrist_hash(child_board)  # ✅
            repetition_count = board_occurrences.get(zobrist, 0)
            if repetition_count >= 1:
                prior *= 0.1  # Penalización fuerte
            elif repetition_count == 1:
                prior *= 0.5  # Penalización leve

            child = Node(child_board, parent=self, move=move)
            child.P = prior
            self.children.append(child)
        self.is_expanded = True

    def backpropagate(self, value: float):
        node = self
        turn = node.board.turn
        while node is not None:
            node.N += 1
            node.W += value if node.board.turn == turn else -value
            node = node.parent


class MCTS:
    def __init__(self, agent, num_simulations=400, cpuct=1.0):
        self.agent = agent
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self._input_buffer = np.zeros((1, 8, 8, 22), dtype=np.float32)

    def _board_to_input(self, board: chess.Board, move_history: List[str]):
        from src.board_representation import fen_to_8x8x22
        fen = board.fen().split()[0]
        return fen_to_8x8x22(fen, last_moves=move_history)

    def _predict(self, board: chess.Board, move_history: List[str]):
        x = self._board_to_input(board, move_history)
        self._input_buffer[0] = x
        policy_pred, value_pred = self.agent.model.predict(self._input_buffer, verbose=0)
        policy = policy_pred[0]
        value = value_pred[0][0]
        return policy, value

    def run(self, root_board: chess.Board) -> List[Tuple[chess.Move, float]]:
        """
        Ejecuta MCTS desde una posición.
        Devuelve: lista de (movimiento, probabilidad visitas)
        """
        root = Node(root_board.copy(stack=False))

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            board_occurrences = {}

            # Registrar posiciones desde la raíz
            # Registrar posiciones desde la raíz
            current = root
            while current is not None:
                zobrist = chess.polyglot.zobrist_hash(current.board)  
                board_occurrences[zobrist] = board_occurrences.get(zobrist, 0) + 1
                current = current.parent

            # 1. Selección
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child()
                search_path.append(node)

            # 2. Expansión
            if not node.board.is_game_over():
                move_history = [m.uci() for m in node.board.move_stack[-2:]]
                raw_policy, value = self._predict(node.board, move_history)

                # Mapea política a movimientos legales
                legal_moves = list(node.board.legal_moves)
                total_prob = 0.0
                policy = {}
                for move in legal_moves:
                    uci = move.uci()
                    idx = self.agent.move_to_idx.get(uci, -1)
                    prob = raw_policy[idx] if idx != -1 else 1e-10
                    policy[uci] = prob
                    total_prob += prob

                # Normaliza
                if total_prob > 1e-8:
                    for uci in policy:
                        policy[uci] /= total_prob

                # Expandir con penalización de repeticiones
                node.expand(policy, board_occurrences)
                value_score = value
            else:
                # Si es final de partida
                result = node.board.result()
                value_score = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

            # 3. Retropropagación
            for node in search_path:
                node.backpropagate(value_score)

        # 4. Política final
        visit_counts = [(child.N, child.move) for child in root.children]
        total_visits = sum(count for count, _ in visit_counts)
        if total_visits == 0:
            return []

        mcts_policy = [(count / total_visits, move) for count, move in visit_counts]
        mcts_policy.sort(key=lambda x: x[0], reverse=True)
        return mcts_policy