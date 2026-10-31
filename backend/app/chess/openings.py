from __future__ import annotations

import random

import chess

from app.chess.rules import game_outcome


def _opening_weight(board: chess.Board, move: chess.Move) -> float:
    """Prefer plausible developing moves while keeping openings diverse."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return 1.0
    if piece.piece_type == chess.PAWN:
        file_index = chess.square_file(move.to_square)
        return 4.0 if file_index in {2, 3, 4, 5} else 1.5
    if piece.piece_type == chess.KNIGHT:
        return 3.0
    if piece.piece_type == chess.BISHOP:
        return 2.0
    if piece.piece_type in {chess.KING, chess.QUEEN}:
        return 0.25
    return 1.0


def generate_opening(seed: int, plies: int) -> list[chess.Move]:
    """Create a reproducible, randomly sampled short opening.

    Nothing about the line is hard-coded: moves are drawn from all legal moves
    with mild weights that favour development. Evaluation pairs share one line
    with colours swapped; self-play draws a fresh line per game.
    """
    rng = random.Random(seed)
    board = chess.Board()
    line: list[chess.Move] = []
    for _ in range(plies):
        if game_outcome(board):
            break
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        move = rng.choices(legal, weights=[_opening_weight(board, move) for move in legal], k=1)[0]
        line.append(move)
        board.push(move)
    return line
