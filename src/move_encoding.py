# src/move_encoding.py

import numpy as np
import chess
from typing import List, Dict

# === CONFIGURACIÓN DEL ESPACIO DE MOVIMIENTOS (8x8x73) ===
NUM_PLANES = 73
BOARD_SIZE = 8
TOTAL_MOVES = BOARD_SIZE * BOARD_SIZE * NUM_PLANES  # 4672

# === TIPOS DE MOVIMIENTOS (73 planos) ===
# Basado en Leela Chess Zero: https://github.com/LeelaChessZero/lc0/wiki/Move-encoding-in-NN-weights
MOVE_TYPES = []

# 1. Desplazamientos normales: 7 direcciones × 8 distancias = 56
# Direcciones: N, NE, E, SE, S, SW, W, NW → pero se codifican como (fila, columna)
DIRECTIONS = [
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1), # NW
]

for d in DIRECTIONS:
    for steps in range(1, 8):  # 1 a 7 pasos
        MOVE_TYPES.append(('normal', d[0] * steps, d[1] * steps))

# 2. Coronaciones: 3 direcciones × 4 promociones = 12 movimientos
# (adelante, adelante-izq, adelante-der) × (Q, R, B, N)
for dc in (-1, 0, 1):  # desplazamiento en columna
    for promo in ('queen', 'rook', 'bishop', 'knight'):
        MOVE_TYPES.append(('promo', dc, promo))

# 3. Caballo: 8 movimientos en forma de L
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]
for dr, dc in KNIGHT_MOVES:
    MOVE_TYPES.append(('knight', dr, dc))

# ✅ Ahora debe tener 56 + 12 + 8 = 76? → No, espera...
# En realidad Lc0 usa solo 73: 56 normales + 8 caballo + 9 promociones
# Corrección: son 56 + 8 + 9 = 73

# Vamos a rehacer con el esquema real:
del MOVE_TYPES[:]

# === ESQUEMA OFICIAL Lc0 (73 planos) ===
MOVE_TYPES = []

# 1. 7 direcciones × 7 distancias = 49 (no 56)
# Direcciones: N, NE, E, SE, S, SW, W, NW → pero se usan solo 7 distancias
# Pero en realidad: son 8 direcciones × 7 distancias = 56 planos
for dr, dc in DIRECTIONS:
    for steps in range(1, 8):  # 1 a 7
        MOVE_TYPES.append((dr, dc, steps))

# 2. Movimientos de caballo: 8 tipos
for dr, dc in KNIGHT_MOVES:
    MOVE_TYPES.append(('knight', dr, dc))

# 3. Coronaciones: 3 direcciones × 3 promociones (sin caballo) = 9
# En Lc0: promociones son Q, R, B → solo 3, no 4 (knight se maneja aparte)
for dc in (-1, 0, 1):
    for promo in (0, 1, 2):  # Q=0, R=1, B=2
        MOVE_TYPES.append(('promo', dc, promo))

assert len(MOVE_TYPES) == 73, f"MOVE_TYPES debe tener 73 elementos, tiene {len(MOVE_TYPES)}"

# === Mapeo global: índice → (from_sq, to_sq) ===
INDEX_TO_MOVE = {}  # index → (from_sq, to_sq)
MOVE_TO_INDEX = {}  # (from_sq, to_sq) → index

def _generate_move_mapping():
    """Precalcula todos los movimientos válidos en el espacio 8x8x73."""
    index = 0
    for from_sq in range(64):
        fr, fc = from_sq // 8, from_sq % 8
        for plane in range(73):
            move_type = MOVE_TYPES[plane]
            if len(move_type) == 3 and move_type[0] != 'knight' and move_type[0] != 'promo':
                # Movimiento normal: (dr, dc, steps)
                dr, dc, steps = move_type
                tr, tc = fr + dr * steps, fc + dc * steps
                if 0 <= tr < 8 and 0 <= tc < 8:
                    to_sq = tr * 8 + tc
                    INDEX_TO_MOVE[index] = (from_sq, to_sq)
                    MOVE_TO_INDEX[(from_sq, to_sq)] = index
            elif move_type[0] == 'knight':
                _, dr, dc = move_type
                tr, tc = fr + dr, fc + dc
                if 0 <= tr < 8 and 0 <= tc < 8:
                    to_sq = tr * 8 + tc
                    INDEX_TO_MOVE[index] = (from_sq, to_sq)
                    MOVE_TO_INDEX[(from_sq, to_sq)] = index
            elif move_type[0] == 'promo':
                _, d_col, promo = move_type  # promo: 0=Q,1=R,2=B
                if fr == 6:  # peón en 6ta fila
                    tr, tc = 7, fc + d_col
                    if 0 <= tc < 8:
                        to_sq = tr * 8 + tc
                        INDEX_TO_MOVE[index] = (from_sq, to_sq)
                        MOVE_TO_INDEX[(from_sq, to_sq)] = index
            index += 1

_generate_move_mapping()

# === API pública ===
def uci_to_flat_index(uci: str) -> int:
    """Convierte UCI a índice en [0, 4671]. Devuelve -1 si inválido."""
    try:
        move = chess.Move.from_uci(uci)
        from_sq = move.from_square
        to_sq = move.to_square
        fr, fc = from_sq // 8, from_sq % 8
        tr, tc = to_sq // 8, to_sq % 8
        dr, dc = tr - fr, tc - fc

        if move.promotion:
            if fr != 6 or tr != 7:
                return -1
            d_col = dc
            if d_col not in (-1, 0, 1):
                return -1
            promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2}
            promo_idx = promo_map.get(move.promotion, -1)
            if promo_idx == -1:
                return -1
            base_plane = 56 + 8 + (d_col + 1) * 3 + promo_idx  # 56 normales + 8 caballo + 9 promo
        elif (dr, dc) in KNIGHT_MOVES:
            base_plane = 56 + KNIGHT_MOVES.index((dr, dc))
        else:
            found = False
            for steps in range(1, 8):
                for idx_dir, (d_row, d_col) in enumerate(DIRECTIONS):
                    if dr == d_row * steps and dc == d_col * steps:
                        base_plane = idx_dir * 7 + (steps - 1)
                        found = True
                        break
                if found:
                    break
            if not found:
                return -1

        from_idx = from_sq
        total_index = from_idx * 73 + base_plane
        return total_index if total_index < 4672 else -1
    except Exception:
        return -1

def flat_index_to_uci(index: int) -> str:
    """Convierte índice [0,4671] a UCI."""
    if index < 0 or index >= 4672:
        return ""
    if index not in INDEX_TO_MOVE:
        return ""
    from_sq, to_sq = INDEX_TO_MOVE[index]
    fr, fc = from_sq // 8, from_sq % 8
    tr, tc = to_sq // 8, to_sq % 8
    uci = f"{chr(97 + fc)}{fr + 1}{chr(97 + tc)}{tr + 1}"
    plane = index % 73
    if plane >= 56 + 8:  # coronación
        promo_map = {0: 'q', 1: 'r', 2: 'b'}
        promo_idx = (plane - (56 + 8)) % 3
        uci += promo_map[promo_idx]
    return uci

def encode_moves_4672(moves: List[str]) -> np.ndarray:
    indices = [uci_to_flat_index(m) for m in moves]
    indices = [i if i != -1 else 0 for i in indices]
    return np.eye(4672)[indices]

def decode_move_4672(index: int) -> str:
    return flat_index_to_uci(index)

def create_move_vocab(moves: List[str], min_freq=1) -> Dict[str, int]:
    from collections import Counter
    counter = Counter(moves)
    filtered = [m for m, cnt in counter.items() if cnt >= min_freq]
    return {move: idx for idx, move in enumerate(sorted(filtered))}