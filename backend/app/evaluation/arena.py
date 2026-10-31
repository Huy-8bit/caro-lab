from __future__ import annotations

import random
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import chess
import torch

from app.chess.mcts import MCTS
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint


def heuristic_move(board: chess.Board, rng: random.Random) -> chess.Move:
    moves = list(board.legal_moves)
    captures = [move for move in moves if board.is_capture(move)]
    promotions = [move for move in moves if move.promotion]
    return rng.choice(promotions or captures or moves)


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
    """Create a reproducible short opening shared by a color-swapped pair."""
    rng = random.Random(seed)
    board = chess.Board()
    line: list[chess.Move] = []
    for _ in range(plies):
        if board.outcome(claim_draw=True):
            break
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        move = rng.choices(legal, weights=[_opening_weight(board, move) for move in legal], k=1)[0]
        line.append(move)
        board.push(move)
    return line


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
) -> dict:
    rng = random.Random(seed)
    opponent_version = opponent
    if checkpoint_path:
        opponent_model, payload = load_checkpoint(Path(checkpoint_path), device)
        opponent_version = f"v{payload['version']}"
    elif opponent_model is not None:
        opponent_version = opponent_version_label or "champion"
    wins = draws = losses = 0
    details = []
    for game_index in range(games):
        board = chess.Board()
        opening = generate_opening(seed + (game_index // 2) * 104_729, opening_plies)
        moves: list[str] = []
        for move in opening:
            moves.append(board.san(move))
            board.push(move)
        model_is_white = game_index % 2 == 0
        model_search = MCTS(
            model, device, simulations=simulations, seed=seed + game_index * 2,
            dirichlet_epsilon=0,
        )
        opponent_search = (
            MCTS(
                opponent_model, device, simulations=simulations,
                seed=seed + game_index * 2 + 1, dirichlet_epsilon=0,
            )
            if opponent_model
            else None
        )
        for _ in range(max(0, max_plies - len(opening))):
            if board.outcome(claim_draw=True):
                break
            model_turn = board.turn == chess.WHITE if model_is_white else board.turn == chess.BLACK
            if model_turn:
                _, _, root = model_search.search(board, add_noise=False)
                move = model_search.select_move(root, 0, board)
            elif opponent_search:
                _, _, root = opponent_search.search(board, add_noise=False)
                move = opponent_search.select_move(root, 0, board)
            elif opponent == "random":
                move = rng.choice(list(board.legal_moves))
            else:
                move = heuristic_move(board, rng)
            moves.append(board.san(move))
            board.push(move)
            if event_callback:
                event_callback(
                    {
                        "type": "evaluation_move",
                        "game_index": game_index,
                        "fen": board.fen(),
                        "move": move.uci(),
                        "model_is_white": model_is_white,
                    }
                )
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
            result = "draw"
        else:
            model_won = outcome.winner == model_is_white
            if model_won:
                wins += 1
                result = "win"
            else:
                losses += 1
                result = "loss"
        termination = (
            outcome.termination.name.lower().replace("_", " ")
            if outcome is not None
            else "max plies"
        )
        details.append(
            {
                "game": game_index + 1,
                "model_color": "white" if model_is_white else "black",
                "result": result,
                "termination": termination,
                "opening_plies": len(opening),
                "moves": moves,
            }
        )
    termination_counts = dict(Counter(item["termination"] for item in details))
    evaluation = {
        "id": uuid.uuid4().hex[:12],
        "model_a": model_version,
        "model_b": opponent_version,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "created_at": datetime.now(UTC).isoformat(),
        "details": details,
        "termination_counts": termination_counts,
        "score": (wins + 0.5 * draws) / max(1, games),
    }
    if persist:
        repository.save_evaluation(evaluation)
    if event_callback:
        event_callback({"type": "evaluation_complete", "evaluation": evaluation})
    return evaluation
