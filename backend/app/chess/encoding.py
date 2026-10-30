from __future__ import annotations

import chess
import numpy as np


INPUT_PLANES = 20
PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as 20 absolute-orientation feature planes."""
    encoded = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        encoded[PIECE_PLANES[(piece.color, piece.piece_type)], rank, file] = 1.0

    if board.turn == chess.WHITE:
        encoded[12].fill(1.0)
    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for offset, available in enumerate(castling, start=13):
        if available:
            encoded[offset].fill(1.0)
    if board.ep_square is not None:
        encoded[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    encoded[18].fill(min(board.halfmove_clock / 100.0, 1.0))
    encoded[19].fill(min(board.fullmove_number / 200.0, 1.0))
    return encoded
