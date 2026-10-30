from __future__ import annotations

import chess


PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
POLICY_SIZE = 64 * 64 * len(PROMOTIONS)


def move_to_index(move: chess.Move) -> int:
    try:
        promotion = PROMOTIONS.index(move.promotion)
    except ValueError as exc:
        raise ValueError(f"Unsupported promotion: {move.promotion}") from exc
    return (move.from_square * 64 + move.to_square) * len(PROMOTIONS) + promotion


def index_to_move(index: int) -> chess.Move:
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(f"Move index out of range: {index}")
    pair, promotion = divmod(index, len(PROMOTIONS))
    from_square, to_square = divmod(pair, 64)
    return chess.Move(from_square, to_square, promotion=PROMOTIONS[promotion])
