from __future__ import annotations

import os
import tempfile

import chess


def pytest_configure(config) -> None:
    # app.main builds a repository at import time; never let tests touch the
    # real ./data or ./checkpoints directories.
    root = tempfile.mkdtemp(prefix="caro-tests-")
    os.environ["CARO_DATA_DIR"] = os.path.join(root, "data")
    os.environ["CARO_CHECKPOINT_DIR"] = os.path.join(root, "checkpoints")
    os.environ.pop("CARO_CONFIG", None)


def board_after(*ucis: str) -> chess.Board:
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


KNIGHT_CYCLE = ("g1f3", "g8f6", "f3g1", "f6g8")
