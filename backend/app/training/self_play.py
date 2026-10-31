from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import chess
import chess.pgn
import math

from app.chess.encoding import encode_board
from app.chess.mcts import MCTS
from app.training.replay import TrainingSample


@dataclass(slots=True)
class SelfPlayResult:
    record: dict[str, Any]
    samples: list[TrainingSample]


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_value(board: chess.Board) -> float:
    """A bounded fallback target for games stopped by the safety move limit."""
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += value * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
    return math.tanh(score / 8.0)


def play_self_play_game(
    search: MCTS,
    iteration: int,
    model_version: int,
    max_moves: int,
    temperature_moves: int,
    temperature: float,
    draw_penalty: float = 0.05,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    pause_callback: Callable[[], bool] | None = None,
) -> SelfPlayResult:
    game_id = uuid.uuid4().hex[:12]
    board = chess.Board()
    started = time.perf_counter()
    positions: list[tuple[Any, Any, bool, int]] = []
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
    pgn_node = game

    for ply in range(max_moves * 2):
        if pause_callback:
            while pause_callback():
                time.sleep(0.1)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        policy, root_value, root = search.search(board, add_noise=True)
        positions.append((encode_board(board), policy, board.turn, ply))
        move = search.select_move(root, temperature if ply < temperature_moves * 2 else 0.0, board)
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
            "visits": root.visit_count,
        }
        move_records.append(record)
        if event_callback:
            event_callback(
                {
                    "type": "move",
                    "game_id": game_id,
                    "iteration": iteration,
                    "model_version": model_version,
                    **record,
                }
            )

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result, winner, termination = "1/2-1/2", None, "max move safety limit"
        result_value = material_value(board)
    else:
        result = outcome.result()
        winner = "white" if outcome.winner == chess.WHITE else "black" if outcome.winner == chess.BLACK else None
        termination = outcome.termination.name.lower().replace("_", " ")
        result_value = 1.0 if outcome.winner == chess.WHITE else -1.0 if outcome.winner == chess.BLACK else 0.0
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    samples = [
        TrainingSample(
            state=state,
            policy=policy,
            value=(
                -draw_penalty
                if outcome is not None and outcome.winner is None
                else result_value if turn == chess.WHITE else -result_value
            ),
            game_id=game_id,
            ply=ply,
            iteration=iteration,
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
            "value_target": round(result_value, 5),
            "value_target_source": (
                "material_bootstrap"
                if outcome is None
                else "draw_penalty" if outcome.winner is None else "game_outcome"
            ),
        },
    }
    if event_callback:
        event_callback({"type": "game_complete", "game_id": game_id, "game": record})
    return SelfPlayResult(record, samples)
