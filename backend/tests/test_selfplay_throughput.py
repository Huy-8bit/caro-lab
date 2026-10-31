"""Speed-ups must not change what self-play computes or lose any game."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from app.chess.encoding import encode_board
from app.chess.mcts import MCTS
from app.chess.move_index import move_to_index
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.distributed import JobBoard
from app.training.engine import TrainingEngine
from app.training.network import NetworkSpec, PolicyValueNetwork


class FullHeadOnly(torch.nn.Module):
    """Same weights, but without the legal-move-only fast path."""

    def __init__(self, network: PolicyValueNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs):
        return self.network(inputs)


def test_legal_move_logits_match_the_full_policy_head() -> None:
    torch.manual_seed(3)
    model = PolicyValueNetwork(NetworkSpec(channels=16, residual_blocks=1)).eval()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    indices = torch.tensor([move_to_index(move) for move in board.legal_moves])
    inputs = torch.from_numpy(encode_board(board)).unsqueeze(0)
    with torch.inference_mode():
        full_logits, full_value = model(inputs)
        logits, value = model.evaluate_moves(inputs, indices)
    torch.testing.assert_close(logits[0], full_logits[0, indices])
    torch.testing.assert_close(value, full_value)


def test_search_is_unchanged_by_the_fast_path() -> None:
    torch.manual_seed(5)
    model = PolicyValueNetwork(NetworkSpec(channels=16, residual_blocks=1))
    board = chess.Board()
    fast = MCTS(model, torch.device("cpu"), simulations=24, seed=11, draw_contempt=0.15)
    slow = MCTS(FullHeadOnly(model), torch.device("cpu"), simulations=24, seed=11, draw_contempt=0.15)
    fast_policy, fast_value, fast_root = fast.search(board)
    slow_policy, slow_value, slow_root = slow.search(board)
    np.testing.assert_allclose(fast_policy, slow_policy, atol=1e-6)
    assert abs(fast_value - slow_value) < 1e-5
    assert {m: c.visit_count for m, c in fast_root.children.items()} == {m: c.visit_count for m, c in slow_root.children.items()}
    assert model.training  # search restores the caller's mode


def test_queued_jobs_can_be_retargeted_or_cancelled_without_touching_running_ones() -> None:
    board = JobBoard()
    board.open_batch("s")
    first, second, third = (board.submit("s", {"model_key": "old", "n": n}) for n in range(3))
    assert board.claim("w1")["job_id"] == first
    assert board.outstanding("s") == 3
    assert board.update_queued("s", {"model_key": "new"}) == 2
    assert board.get(first).job["model_key"] == "old"
    assert board.get(second).job["model_key"] == board.get(third).job["model_key"] == "new"
    assert board.cancel_queued("s") == 2
    assert board.outstanding("s") == 1 and board.claim("w2") is None
    assert board.complete("w1", first, "done")
    assert board.next_result("s", timeout=0.1) == (first, "done")


def test_selfplay_stream_keeps_workers_busy_and_carries_running_games_over(tmp_path: Path, monkeypatch) -> None:
    config = TrainingConfig(channels=8, residual_blocks=0, self_play_mode="distributed", games_per_iteration=4)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    monkeypatch.setattr(engine, "_active_workers", lambda: [{}, {}, {}])
    collect = engine._collect
    script: list = []

    def scripted_collect(batch, expected, on_result=None):
        script.pop(0)(batch)
        return collect(batch, expected, on_result)

    monkeypatch.setattr(engine, "_collect", scripted_collect)
    slow: dict = {}

    def iteration_one(batch):
        # 4 games + 1 look-ahead job per worker are outstanding.
        assert engine.jobs.outstanding(batch) == 7
        slow.update(engine.jobs.claim("slow"))  # a long game, still running at the boundary
        for index in range(4):
            job = engine.jobs.claim(f"fast-{index}")
            engine.jobs.complete(f"fast-{index}", job["job_id"], job)

    def iteration_two(batch):
        queued = [lease.job for lease in engine.jobs._jobs.values() if lease.status == "queued"]
        assert len(queued) == 6 and engine.jobs.outstanding(batch) == 7
        assert all(job["model_key"] == engine._selfplay_model_key and job["iteration"] == 2 for job in queued)
        # The running game keeps its model, which is still downloadable.
        assert slow["model_key"] != engine._selfplay_model_key and slow["iteration"] == 1
        assert engine.worker_model_path(slow["model_key"]).exists()
        engine.jobs.complete("slow", slow["job_id"], slow)
        for index in range(3):
            job = engine.jobs.claim(f"next-{index}")
            engine.jobs.complete(f"next-{index}", job["job_id"], job)

    script[:] = [iteration_one, iteration_two]
    engine.iteration = 1
    first = engine._play_remote_iteration()
    engine.iteration = 2
    engine._worker_model_revision = -1  # force a new model publish
    second = engine._play_remote_iteration()

    assert len(first) == len(second) == 4
    # Consumed results are forgotten, so a long run does not accumulate leases.
    ids = [job["job_id"] for job in first + second]
    assert len(set(ids)) == 8 and slow["job_id"] in ids
    assert len({job["seed"] for job in first + second}) == 8
    assert sorted(job["iteration"] for job in second) == [1, 2, 2, 2]
    assert all(engine.jobs.get(job_id) is None for job_id in (job["job_id"] for job in first + second))


def test_selfplay_stream_backlog_is_bounded(tmp_path: Path, monkeypatch) -> None:
    config = TrainingConfig(channels=8, residual_blocks=0, self_play_mode="distributed", games_per_iteration=4)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    monkeypatch.setattr(engine, "_active_workers", lambda: [{}, {}, {}])
    collect = engine._collect

    def fast_workers(batch, expected, on_result=None):
        # Workers finish every outstanding job before the backend consumes any.
        index = 0
        while (job := engine.jobs.claim(f"w{index}")) is not None:
            engine.jobs.complete(f"w{index}", job["job_id"], job)
            index += 1
        return collect(batch, expected, on_result)

    monkeypatch.setattr(engine, "_collect", fast_workers)
    for iteration in range(1, 6):
        engine.iteration = iteration
        assert len(engine._play_remote_iteration()) == 4
        stream = engine._selfplay_stream
        assert engine.jobs.pending_results(stream) + engine.jobs.outstanding(stream) <= 3
    # Only results waiting to be consumed are still held.
    assert engine.jobs.counts() == {"queued": 0, "leased": 0, "complete": engine.jobs.pending_results(stream)}
