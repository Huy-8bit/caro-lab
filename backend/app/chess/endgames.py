"""Random reduced-material start positions for a self-play curriculum.

A network trained from scratch only sees mates at the end of long, noisy
games, so it learns neither that extra material wins nor how to deliver mate.
In our runs the side a queen ahead routinely shuffled into the move limit or a
repetition, and the value head rated KQ vs K as lost. Starting a fraction of
self-play games from simplified positions gives short games with real results.

Nothing here decides a result or a move: a family only fixes which pieces are
on the board, squares are drawn uniformly at random, and the game is played out
by the search under the normal rules. Promotion matches always start from the
standard initial position.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from app.chess.rules import game_outcome


@dataclass(frozen=True)
class EndgameFamily:
    name: str
    strong: str  # piece letters besides the king, e.g. "QR"
    weak: str
    weight: float


# Basic mates teach mating patterns; imbalances teach that material converts;
# pawn and balanced endings keep drawn and technical positions in the data.
FAMILIES: tuple[EndgameFamily, ...] = (
    EndgameFamily("KQvK", "Q", "", 3.0),
    EndgameFamily("KRvK", "R", "", 3.0),
    EndgameFamily("KRRvK", "RR", "", 1.0),
    EndgameFamily("KQvKR", "Q", "R", 1.5),
    EndgameFamily("KQvKN", "Q", "N", 1.0),
    EndgameFamily("KQvKB", "Q", "B", 1.0),
    EndgameFamily("KRvKN", "R", "N", 0.5),
    EndgameFamily("KPvK", "P", "", 1.5),
    EndgameFamily("KPPvKP", "PP", "P", 1.5),
    EndgameFamily("KRPvKR", "RP", "R", 1.5),
    EndgameFamily("KQPvKQ", "QP", "Q", 0.5),
    EndgameFamily("random", "", "", 3.0),
)

PIECES = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}
# Material each side can draw from in the "random" family.
RANDOM_POOL = "QRRBBNNPPPPPPPP"


def _random_material(rng: random.Random) -> tuple[str, str]:
    """Two subsets of a normal army, 1-6 pieces each, usually unequal."""
    def draw() -> str:
        return "".join(rng.sample(RANDOM_POOL, rng.randint(1, 6)))

    return draw(), draw()


def _place(board: chess.Board, rng: random.Random, letters: str, color: chess.Color) -> bool:
    for letter in letters:
        piece_type = PIECES[letter]
        free = [
            square for square in chess.SQUARES
            if board.piece_at(square) is None and (piece_type != chess.PAWN or 1 <= chess.square_rank(square) <= 6)
        ]
        if not free:
            return False
        board.set_piece_at(rng.choice(free), chess.Piece(piece_type, color))
    return True


def random_endgame(seed: int) -> tuple[chess.Board, str]:
    """A legal, unfinished reduced-material position and its family name."""
    rng = random.Random(seed)
    family = rng.choices(FAMILIES, weights=[item.weight for item in FAMILIES], k=1)[0]
    for _ in range(1_000):
        strong, weak = (family.strong, family.weak) if family.name != "random" else _random_material(rng)
        strong_color = rng.choice((chess.WHITE, chess.BLACK))
        board = chess.Board(None)
        board.turn = rng.choice((chess.WHITE, chess.BLACK))
        kings = rng.sample(chess.SQUARES, 2)
        if chess.square_distance(*kings) < 2:
            continue
        board.set_piece_at(kings[0], chess.Piece(chess.KING, strong_color))
        board.set_piece_at(kings[1], chess.Piece(chess.KING, not strong_color))
        if not (_place(board, rng, strong, strong_color) and _place(board, rng, weak, not strong_color)):
            continue
        if board.is_valid() and game_outcome(board) is None and any(board.legal_moves):
            return board, family.name
    raise RuntimeError(f"Could not place a legal {family.name} position")


def curriculum_start(seed: int, fraction: float) -> dict[str, str] | None:
    """Decide from ``seed`` whether a self-play game starts from an endgame."""
    if fraction <= 0 or random.Random(seed ^ 0x5EED).random() >= fraction:
        return None
    board, family = random_endgame(seed)
    return {"fen": board.fen(), "family": family}
