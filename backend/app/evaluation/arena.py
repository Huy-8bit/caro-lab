from __future__ import annotations

import random
import uuid
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
    event_callback: Callable[[dict], None] | None = None,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    opponent_model = None
    opponent_version = opponent
    if checkpoint_path:
        opponent_model, payload = load_checkpoint(Path(checkpoint_path), device)
        opponent_version = f"v{payload['version']}"
    wins = draws = losses = 0
    details = []
    model_search = MCTS(model, device, simulations=simulations, seed=seed, dirichlet_epsilon=0)
    opponent_search = (
        MCTS(opponent_model, device, simulations=simulations, seed=seed + 1, dirichlet_epsilon=0)
        if opponent_model
        else None
    )
    for game_index in range(games):
        board = chess.Board()
        model_is_white = game_index % 2 == 0
        moves: list[str] = []
        for _ in range(max_plies):
            if board.outcome(claim_draw=True):
                break
            model_turn = board.turn == chess.WHITE if model_is_white else board.turn == chess.BLACK
            if model_turn:
                _, _, root = model_search.search(board, add_noise=False)
                move = model_search.select_move(root, 0)
            elif opponent_search:
                _, _, root = opponent_search.search(board, add_noise=False)
                move = opponent_search.select_move(root, 0)
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
        details.append({"game": game_index + 1, "model_color": "white" if model_is_white else "black", "result": result, "moves": moves})
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
    }
    repository.save_evaluation(evaluation)
    if event_callback:
        event_callback({"type": "evaluation_complete", "evaluation": evaluation})
    return evaluation
