from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import chess
import chess.pgn

from app.chess.encoding import encode_board
from app.chess.mcts import MCTS
from app.chess.openings import generate_opening
from app.chess.rules import game_outcome, is_repetition_termination, termination_name
from app.training.replay import ADJUDICATED, DECISIVE, DRAW, REPETITION, TrainingSample


class GameAborted(RuntimeError):
    """Raised when a worker is told to stop in the middle of a game."""


@dataclass(slots=True)
class SelfPlayResult:
    record: dict[str, Any]
    samples: list[TrainingSample]


@dataclass(slots=True)
class SampleWeights:
    """Replay weight per outcome class; decisive games are the reference 1.0."""

    draw: float = 0.6
    repetition: float = 0.25
    adjudicated: float = 0.3
    length_reference: int = 80

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "SampleWeights":
        return cls(
            draw=float(settings.get("draw_sample_weight", cls.draw)),
            repetition=float(settings.get("repetition_sample_weight", cls.repetition)),
            adjudicated=float(settings.get("adjudicated_sample_weight", cls.adjudicated)),
            length_reference=int(settings.get("sample_length_reference", cls.length_reference)),
        )

    def for_outcome(self, outcome_class: str) -> float:
        return {DECISIVE: 1.0, DRAW: self.draw, REPETITION: self.repetition, ADJUDICATED: self.adjudicated}[outcome_class]


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_value(board: chess.Board) -> float:
    """A bounded fallback target (White's perspective) for games stopped by the safety move limit."""
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += value * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
    return math.tanh(score / 8.0)


def classify_outcome(board: chess.Board, outcome: chess.Outcome | None) -> tuple[str, float]:
    """Return the outcome class and the final value target from White's perspective."""
    if outcome is None:
        return ADJUDICATED, material_value(board)
    if outcome.winner is None:
        return (REPETITION if is_repetition_termination(termination_name(outcome)) else DRAW), 0.0
    return DECISIVE, 1.0 if outcome.winner == chess.WHITE else -1.0


def value_target(white_value: float, turn: chess.Color) -> float:
    """Final result seen by the player to move at a recorded position."""
    return white_value if turn == chess.WHITE else -white_value


def play_self_play_game(
    search: MCTS,
    iteration: int,
    model_version: int,
    max_moves: int,
    temperature_moves: int,
    temperature: float,
    *,
    temperature_final: float = 0.0,
    opening_plies: int = 0,
    opening_seed: int | None = None,
    sample_weights: SampleWeights | None = None,
    start_fen: str | None = None,
    start_family: str | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    pause_callback: Callable[[], bool] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> SelfPlayResult:
    """Play one self-play game.

    ``start_fen`` starts the game from a curriculum position (see
    :mod:`app.chess.endgames`) instead of the initial position plus a random
    opening; the game is then played and scored under the normal rules.
    """
    weights = sample_weights or SampleWeights()
    game_id = uuid.uuid4().hex[:12]
    board = chess.Board(start_fen) if start_fen else chess.Board()
    started = time.perf_counter()
    positions: list[tuple[Any, Any, bool, int]] = []
    seen_positions: set[Any] = set()
    duplicates = 0
    move_records: list[dict[str, Any]] = []
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "Caro Lab Self-Play",
            "White": f"Model v{model_version}",
            "Black": f"Model v{model_version}",
            "Date": datetime.now(UTC).strftime("%Y.%m.%d"),
        }
    )
    if start_fen:
        game.setup(board)
    pgn_node = game

    def record_move(move: chess.Move, ply: int, root_value: float | None, visits: int, opening: bool) -> None:
        nonlocal pgn_node
        san = board.san(move)
        before_fen = board.fen()
        board.push(move)
        pgn_node = pgn_node.add_variation(move)
        record = {
            "ply": ply + 1,
            "uci": move.uci(),
            "san": san,
            "fen_before": before_fen,
            "fen": board.fen(),
            "root_value": root_value,
            "visits": visits,
            "opening": opening,
        }
        move_records.append(record)
        if event_callback:
            event_callback(
                {"type": "move", "game_id": game_id, "iteration": iteration, "model_version": model_version, **record}
            )

    seed = opening_seed if opening_seed is not None else int(search.rng.integers(2**31))
    opening = [] if start_fen else generate_opening(seed, opening_plies)
    for ply, move in enumerate(opening[: max_moves * 2]):
        seen_positions.add(board._transposition_key())
        record_move(move, ply, None, 0, True)

    for ply in range(len(move_records), max_moves * 2):
        if abort_callback and abort_callback():
            raise GameAborted(game_id)
        if pause_callback:
            while pause_callback():
                time.sleep(0.1)
        if game_outcome(board) is not None:
            break
        policy, root_value, root = search.search(board, add_noise=True)
        key = board._transposition_key()
        if key in seen_positions:
            # A position seen earlier in this game would only duplicate its
            # first sample (same state, same outcome) and over-weight cycles.
            duplicates += 1
        else:
            seen_positions.add(key)
            positions.append((encode_board(board), policy, board.turn, ply))
        move_temperature = temperature if ply < temperature_moves * 2 else temperature_final
        move = search.select_move(root, move_temperature)
        record_move(move, ply, root_value, root.visit_count, False)

    outcome = game_outcome(board)
    outcome_class, white_value = classify_outcome(board, outcome)
    termination = termination_name(outcome)
    if outcome is None:
        result, winner = "1/2-1/2", None
    else:
        result = outcome.result()
        winner = "white" if outcome.winner == chess.WHITE else "black" if outcome.winner == chess.BLACK else None
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    length_factor = min(1.0, weights.length_reference / max(1, len(positions)))
    sample_weight = weights.for_outcome(outcome_class) * length_factor
    samples = [
        TrainingSample.from_dense(
            state=state,
            policy=policy,
            value=value_target(white_value, turn),
            game_id=game_id,
            ply=ply,
            iteration=iteration,
            weight=sample_weight,
            outcome=outcome_class,
        )
        for state, policy, turn, ply in positions
    ]
    record = {
        "id": game_id,
        "iteration": iteration,
        "model_version": model_version,
        "white_agent": f"Model v{model_version}",
        "black_agent": f"Model v{model_version}",
        "result": result,
        "winner": winner,
        "move_count": len(move_records),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(UTC).isoformat(),
        "pgn": str(game),
        "final_fen": board.fen(),
        "termination": termination,
        "moves": move_records,
        "metadata": {
            "mcts_simulations": search.simulations,
            "training_samples": len(samples),
            "duplicate_positions_skipped": duplicates,
            "opening_plies": len(opening),
            "start": "endgame" if start_fen else "standard",
            "start_family": start_family if start_fen else None,
            "start_fen": start_fen,
            "outcome_class": outcome_class,
            "sample_weight": round(sample_weight, 5),
            "value_target": round(white_value, 5),
            "value_target_source": "material_bootstrap" if outcome is None else "game_outcome",
            "draw_contempt": search.draw_contempt,
        },
    }
    if event_callback:
        event_callback({"type": "game_complete", "game_id": game_id, "game": record})
    return SelfPlayResult(record, samples)
