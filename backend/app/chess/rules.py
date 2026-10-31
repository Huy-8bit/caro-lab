"""Single source of truth for when a game is over.

``python-chess``'s ``board.outcome(claim_draw=True)`` reports a draw as soon as
the side to move *could* claim one, including "a legal move would reach a third
occurrence". Used as an automatic adjudicator that ends self-play at the second
occurrence of a position and lets the side to move "claim" even when it is
winning. Engines and engine tournaments instead adjudicate when the condition
has actually happened, which is what this module implements:

* checkmate, stalemate, insufficient material, fivefold repetition and the
  75-move rule (python-chess automatic outcomes);
* threefold repetition once the current position has occurred three times;
* the fifty-move rule once 100 half-moves have passed without progress.

No draw rule is removed; only the "might claim next move" shortcut is gone.
"""

from __future__ import annotations

import chess

MAX_MOVE_TERMINATION = "max move safety limit"
REPETITION_TERMINATIONS = frozenset({"threefold repetition", "fivefold repetition"})


def game_outcome(board: chess.Board) -> chess.Outcome | None:
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return outcome
    if board.halfmove_clock >= 100:
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return None


def termination_name(outcome: chess.Outcome | None) -> str:
    if outcome is None:
        return MAX_MOVE_TERMINATION
    return outcome.termination.name.lower().replace("_", " ")


def is_repetition_termination(termination: str) -> bool:
    return termination in REPETITION_TERMINATIONS


def search_copy(board: chess.Board) -> chess.Board:
    """Copy a board with exactly the history needed for repetition detection.

    A repetition can only reach back to the last capture or pawn move, which is
    what the half-move clock counts, so older history is never needed.
    """
    return board.copy(stack=min(len(board.move_stack), board.halfmove_clock))
