from __future__ import annotations

import chess
import numpy as np
import pytest

from app.chess.encoding import INPUT_PLANES, encode_board
from app.chess.mcts import terminal_value
from app.chess.move_index import POLICY_SIZE, index_to_move, move_to_index


def test_python_chess_handles_special_rules() -> None:
    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert chess.Move.from_uci("e1g1") in castle.legal_moves
    en_passant = chess.Board("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    assert chess.Move.from_uci("e5d6") in en_passant.legal_moves
    promotion = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    assert chess.Move.from_uci("a7a8q") in promotion.legal_moves


def test_checkmate_and_draw_outcomes() -> None:
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.is_checkmate()
    assert terminal_value(mate) == -1.0
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.is_stalemate()
    assert terminal_value(stale) == 0.0


def test_board_encoding_shape_and_metadata() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor.shape == (INPUT_PLANES, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor[:12].sum() == 32
    assert tensor[12].sum() == 64
    assert all(tensor[index].sum() == 64 for index in range(13, 17))
    board.push_uci("e2e4")
    encoded = encode_board(board)
    assert encoded[12].sum() == 0
    assert encoded[17].sum() == 1


@pytest.mark.parametrize("uci", ["e2e4", "e1g1", "a7a8q", "a7a8n"])
def test_move_index_round_trip(uci: str) -> None:
    move = chess.Move.from_uci(uci)
    index = move_to_index(move)
    assert 0 <= index < POLICY_SIZE
    assert index_to_move(index) == move
