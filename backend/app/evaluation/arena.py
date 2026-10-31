from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import chess
import torch

from app.chess.mcts import MCTS
from app.chess.openings import _opening_weight, generate_opening  # noqa: F401 - re-exported
from app.chess.rules import game_outcome, is_repetition_termination, termination_name
from app.evaluation.progress import EvaluationProgress
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint


def heuristic_move(board: chess.Board, rng: random.Random) -> chess.Move:
    moves = list(board.legal_moves)
    captures = [move for move in moves if board.is_capture(move)]
    promotions = [move for move in moves if move.promotion]
    return rng.choice(promotions or captures or moves)


def opening_seed(seed: int, game_index: int) -> int:
    """Games 2k and 2k+1 share an opening; colours are swapped between them."""
    return seed + (game_index // 2) * 104_729


def candidate_is_white(game_index: int) -> bool:
    return game_index % 2 == 0


def play_match_game(
    candidate: torch.nn.Module,
    opponent: torch.nn.Module | str,
    device: torch.device,
    game_index: int,
    *,
    seed: int,
    simulations: int,
    max_plies: int,
    opening_plies: int,
    c_puct: float = 1.5,
    draw_contempt: float = 0.0,
    policy_mode: str = "visits",
    value_scale: float = 0.1,
    move_callback: Callable[[int, chess.Board, chess.Move], None] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Play one deterministic evaluation game (no Dirichlet noise, greedy moves).

    ``opponent`` is a model or one of the fixed baselines ``"random"`` and
    ``"heuristic"``. The result is reported from the candidate's perspective.
    """
    rng = random.Random(seed + game_index)
    is_white = candidate_is_white(game_index)
    opening = generate_opening(opening_seed(seed, game_index), opening_plies)
    board = chess.Board()
    moves: list[str] = []
    for move in opening:
        moves.append(board.san(move))
        board.push(move)
    candidate_search = MCTS(
        candidate, device, simulations=simulations, c_puct=c_puct,
        dirichlet_epsilon=0.0, draw_contempt=draw_contempt, seed=seed + game_index * 2,
        policy_mode=policy_mode, value_scale=value_scale,
    )
    opponent_search = (
        MCTS(
            opponent, device, simulations=simulations, c_puct=c_puct,
            dirichlet_epsilon=0.0, draw_contempt=draw_contempt, seed=seed + game_index * 2 + 1,
            policy_mode=policy_mode, value_scale=value_scale,
        )
        if isinstance(opponent, torch.nn.Module)
        else None
    )
    while len(board.move_stack) < max_plies and game_outcome(board) is None:
        if abort_callback and abort_callback():
            raise InterruptedError(f"evaluation game {game_index} aborted")
        if board.turn == (chess.WHITE if is_white else chess.BLACK):
            _, _, root = candidate_search.search(board, add_noise=False)
            move = candidate_search.select_move(root, 0)
        elif opponent_search is not None:
            _, _, root = opponent_search.search(board, add_noise=False)
            move = opponent_search.select_move(root, 0)
        elif opponent == "random":
            move = rng.choice(list(board.legal_moves))
        else:
            move = heuristic_move(board, rng)
        moves.append(board.san(move))
        board.push(move)
        if move_callback:
            move_callback(game_index, board, move)
    outcome = game_outcome(board)
    if outcome is None or outcome.winner is None:
        result = "draw"
    else:
        result = "win" if outcome.winner == (chess.WHITE if is_white else chess.BLACK) else "loss"
    return {
        "game": game_index + 1,
        "game_index": game_index,
        "model_color": "white" if is_white else "black",
        "result": result,
        "termination": termination_name(outcome),
        "plies": len(board.move_stack),
        "opening_plies": len(opening),
        "moves": moves,
        "final_fen": board.fen(),
    }


def summarize_match(details: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(details)
    wins = sum(1 for item in details if item["result"] == "win")
    draws = sum(1 for item in details if item["result"] == "draw")
    losses = games - wins - draws
    score = (wins + 0.5 * draws) / max(1, games)
    terminations = Counter(item["termination"] for item in details)
    repetitions = sum(count for name, count in terminations.items() if is_repetition_termination(name))
    summary = {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "termination_counts": dict(terminations),
        "repetition_rate": repetitions / max(1, games),
        "decisive_rate": (wins + losses) / max(1, games),
        "checkmate_rate": terminations.get("checkmate", 0) / max(1, games),
    }
    summary.update(elo_estimate(details))
    return summary


def elo_estimate(details: list[dict[str, Any]]) -> dict[str, float | None]:
    """Relative Elo difference with a 95% normal-approximation interval."""
    games = len(details)
    if not games:
        return {"elo_diff": None, "elo_low": None, "elo_high": None}
    points = [{"win": 1.0, "draw": 0.5, "loss": 0.0}[item["result"]] for item in details]
    mean = sum(points) / games
    variance = sum((point - mean) ** 2 for point in points) / games
    margin = 1.96 * math.sqrt(variance / games)

    def to_elo(score: float) -> float:
        bounded = min(max(score, 0.5 / games), 1 - 0.5 / games)
        return -400 * math.log10(1 / bounded - 1)

    return {
        "elo_diff": round(to_elo(mean), 1),
        "elo_low": round(to_elo(mean - margin), 1),
        "elo_high": round(to_elo(mean + margin), 1),
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    model_version: int,
    device: torch.device,
    repository: Repository,
    opponent: str,
    games: int,
    simulations: int = 8,
    max_plies: int = 160,
    checkpoint_path: str | None = None,
    opponent_model: torch.nn.Module | None = None,
    opponent_version_label: str | None = None,
    event_callback: Callable[[dict], None] | None = None,
    seed: int = 0,
    opening_plies: int = 4,
    persist: bool = True,
    c_puct: float = 1.5,
    draw_contempt: float = 0.0,
    parallel: int = 1,
    progress: EvaluationProgress | None = None,
    abort_callback: Callable[[], bool] | None = None,
    finish_progress: bool = True,
    policy_mode: str = "visits",
    value_scale: float = 0.1,
) -> dict:
    """Play a colour-balanced, opening-paired match and summarise it.

    Used by Model Battle and by the local promotion gate; distributed
    promotion runs the same :func:`play_match_game` on self-play workers.
    """
    opponent_version = opponent
    if checkpoint_path:
        opponent_model, payload = load_checkpoint(Path(checkpoint_path), device)
        opponent_version = f"v{payload['version']}"
    elif opponent_model is not None:
        opponent_version = opponent_version_label or "champion"
    rival: torch.nn.Module | str = opponent_model if opponent_model is not None else opponent

    def on_move(game_index: int, board: chess.Board, move: chess.Move) -> None:
        if progress:
            progress.game_move(game_index, len(board.move_stack), board.fen())
        if event_callback:
            event_callback(
                {
                    "type": "evaluation_move",
                    "game_index": game_index,
                    "fen": board.fen(),
                    "move": move.uci(),
                    "model_is_white": candidate_is_white(game_index),
                }
            )

    def run(game_index: int) -> dict[str, Any]:
        if progress:
            progress.game_started(game_index, "white" if candidate_is_white(game_index) else "black")
        detail = play_match_game(
            model, rival, device, game_index, seed=seed, simulations=simulations, max_plies=max_plies,
            opening_plies=opening_plies, c_puct=c_puct, draw_contempt=draw_contempt, move_callback=on_move,
            abort_callback=abort_callback, policy_mode=policy_mode, value_scale=value_scale,
        )
        if progress:
            progress.game_finished(detail)
        return detail

    with ThreadPoolExecutor(max_workers=max(1, min(parallel, games)), thread_name_prefix="caro-eval") as pool:
        details = list(pool.map(run, range(games)))
    summary = summarize_match(details)
    evaluation = {
        "id": progress.id if progress else uuid.uuid4().hex[:12],
        "model_a": model_version,
        "model_b": opponent_version,
        "created_at": datetime.now(UTC).isoformat(),
        "details": details,
        "kind": "manual",
        **summary,
    }
    if progress and finish_progress:
        progress.finish("complete", {"status": "complete", **summary})
    if persist:
        repository.save_evaluation(evaluation)
    if event_callback:
        event_callback({"type": "evaluation_complete", "evaluation": evaluation})
    return evaluation
