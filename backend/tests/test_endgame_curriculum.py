"""The endgame curriculum only changes start positions, never rules or results."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import torch

from app.chess.endgames import FAMILIES, curriculum_start, random_endgame
from app.chess.mcts import MCTS
from app.chess.rules import game_outcome
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.analytics import quality_report
from app.training.engine import TrainingEngine
from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.self_play import play_self_play_game


def test_curriculum_positions_are_legal_unfinished_and_reproducible() -> None:
    families, strong_white, turns = Counter(), 0, Counter()
    for seed in range(600):
        board, family = random_endgame(seed)
        assert board.is_valid() and game_outcome(board) is None and any(board.legal_moves)
        assert board.fen() == random_endgame(seed)[0].fen()
        assert not board.castling_rights and board.ep_square is None
        for square in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK):
            assert 1 <= chess.square_rank(square) <= 6
        families[family] += 1
        turns[board.turn] += 1
        if family == "KQvK":
            strong_white += bool(board.pieces(chess.QUEEN, chess.WHITE))
    assert set(families) == {family.name for family in FAMILIES}
    assert 0.35 < strong_white / families["KQvK"] < 0.65  # both colours get the extra material
    assert 0.4 < turns[chess.WHITE] / 600 < 0.6


def test_curriculum_fraction() -> None:
    assert all(curriculum_start(seed, 0.0) is None for seed in range(200))
    share = sum(curriculum_start(seed, 0.3) is not None for seed in range(4000)) / 4000
    assert 0.26 < share < 0.34
    start = next(item for item in (curriculum_start(seed, 1.0) for seed in range(5)) if item)
    assert chess.Board(start["fen"]).is_valid()


def test_selfplay_from_a_curriculum_position_is_recorded_and_replayable() -> None:
    torch.manual_seed(0)
    model = PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=0))
    fen = "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"
    result = play_self_play_game(
        MCTS(model, torch.device("cpu"), simulations=4, seed=1), iteration=3, model_version=1,
        max_moves=6, temperature_moves=2, temperature=1.0, opening_plies=4, start_fen=fen, start_family="KQvK",
    )
    record = result.record
    assert record["metadata"]["start"] == "endgame" and record["metadata"]["start_family"] == "KQvK"
    assert record["metadata"]["opening_plies"] == 0  # no random opening on top of the position
    assert record["moves"][0]["fen_before"] == fen and not any(move["opening"] for move in record["moves"])
    game = chess.pgn.read_game(io.StringIO(record["pgn"]))
    assert game.headers["FEN"] == fen
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    assert board.fen() == record["final_fen"]
    assert len(result.samples) == len(record["moves"]) - record["metadata"]["duplicate_positions_skipped"]


def test_trainer_assigns_curriculum_starts_to_a_fraction_of_jobs(tmp_path: Path, monkeypatch) -> None:
    config = TrainingConfig(channels=8, residual_blocks=0, self_play_mode="distributed", games_per_iteration=40,
                            endgame_curriculum_fraction=0.5)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    monkeypatch.setattr(engine, "_active_workers", lambda: [])
    monkeypatch.setattr(engine, "_collect", lambda batch, expected, on_result=None: [])
    engine.iteration = 1
    engine._play_remote_iteration()
    jobs = [lease.job for lease in engine.jobs._jobs.values()]
    starts = [job for job in jobs if job.get("start_fen")]
    assert len(jobs) == 40 and 10 <= len(starts) <= 30
    assert all(chess.Board(job["start_fen"]).is_valid() and job["start_family"] for job in starts)


def _game(game_id: str, iteration: int, termination: str, start: str) -> dict:
    result = "1-0" if termination == "checkmate" else "1/2-1/2"
    return {
        "id": game_id, "iteration": iteration, "model_version": 1, "white_agent": "a", "black_agent": "b",
        "result": result, "winner": "white" if result == "1-0" else None, "move_count": 40, "duration_seconds": 1.0,
        "created_at": f"2026-01-01T00:00:{len(game_id):02d}+00:00", "pgn": "", "final_fen": chess.Board().fen(),
        "termination": termination, "moves": [], "metadata": {"start": start},
    }


def test_quality_rates_ignore_curriculum_games(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    repository.save_game(_game("s1", 1, "threefold repetition", "standard"))
    repository.save_game(_game("s2", 1, "checkmate", "standard"))
    repository.save_game({**_game("legacy", 1, "threefold repetition", "standard"), "metadata": {}})
    for index in range(5):
        repository.save_game(_game(f"e{index}", 1, "checkmate" if index < 4 else "fifty moves", "endgame"))
    row = repository.iteration_quality()[0]
    assert (row["games"], row["draws"], row["repetitions"], row["checkmates"]) == (3, 2, 2, 1)
    assert (row["curriculum_games"], row["curriculum_checkmates"]) == (5, 4)
    rolling = quality_report([row])["rolling"]
    assert rolling["draw_rate"] == 2 / 3 and rolling["curriculum_checkmate_rate"] == 0.8


def test_curriculum_games_use_their_own_temperature_schedule(tmp_path: Path, monkeypatch) -> None:
    from app.training import engine as engine_module

    config = TrainingConfig(channels=8, residual_blocks=0, temperature_moves=40, endgame_temperature_moves=0,
                            endgame_curriculum_fraction=0.5)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    calls: list[dict] = []
    monkeypatch.setattr(engine_module, "play_self_play_game", lambda **kwargs: calls.append(kwargs))
    for index in range(30):
        engine._play_game(index)
    endgame = [call for call in calls if call.get("start_fen")]
    standard = [call for call in calls if not call.get("start_fen")]
    assert endgame and standard
    assert {call["temperature_moves"] for call in endgame} == {0}
    assert {call["temperature_moves"] for call in standard} == {40}
