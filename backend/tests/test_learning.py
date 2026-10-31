from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from app.chess.mcts import MCTS
from app.chess.move_index import POLICY_SIZE
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.evaluation.arena import evaluate_checkpoint, generate_opening
from app.training.checkpoints import load_checkpoint, save_checkpoint, save_training_state
from app.training.distributed import decode_result, encode_samples
from app.training.engine import TrainingEngine
from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.replay import ReplayBuffer, TrainingSample
from app.training.self_play import material_value


def tiny_model() -> PolicyValueNetwork:
    return PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=0))


def test_network_output_shapes() -> None:
    model = tiny_model()
    policy, value = model(torch.zeros(3, 20, 8, 8))
    assert policy.shape == (3, POLICY_SIZE)
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1)


def test_mcts_policy_is_legal_and_normalized() -> None:
    board = chess.Board()
    search = MCTS(tiny_model(), torch.device("cpu"), simulations=2, seed=2)
    policy, _, root = search.search(board, add_noise=False)
    assert root.children
    assert set(root.children).issubset(set(board.legal_moves))
    assert np.isclose(policy.sum(), 1.0)
    assert search.select_move(root, 0) in board.legal_moves


def test_replay_buffer_sampling() -> None:
    buffer = ReplayBuffer(3, seed=1)
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[7] = 1.0
    samples = [TrainingSample.from_dense(np.zeros((20, 8, 8), np.float32), policy, 0, "g", i, 1) for i in range(5)]
    buffer.extend(samples)
    assert len(buffer) == 3
    assert len(buffer.sample(10)) == 3


def test_distributed_sample_payload_round_trip() -> None:
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[3] = 1.0
    samples = [
        TrainingSample.from_dense(np.zeros((20, 8, 8), np.float32), policy, -0.05, "game", 2, 4, 0.25, "repetition"),
        TrainingSample.from_dense(np.ones((20, 8, 8), np.float32), policy, 1.0, "game", 3, 4),
    ]
    record = {"id": "game", "moves": [], "result": "1-0"}
    restored = decode_result(record, encode_samples(samples), iteration=4)
    assert len(restored.samples) == 2
    assert restored.samples[0].value == pytest.approx(-0.05)
    assert np.array_equal(restored.samples[1].state, samples[1].state)
    assert np.array_equal(restored.samples[0].dense_policy(), policy)
    assert restored.samples[0].outcome == "repetition"
    assert restored.samples[0].weight == pytest.approx(0.25)
    assert restored.samples[1].outcome == "decisive"


def test_optimizer_changes_real_weights(tmp_path: Path) -> None:
    config = TrainingConfig(channels=8, residual_blocks=0, batch_size=2, replay_capacity=10)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    policy = np.zeros(POLICY_SIZE, np.float32)
    policy[0] = 1
    engine.replay.extend([
        TrainingSample.from_dense(np.zeros((20, 8, 8), np.float32), policy, 1.0, "a", 0, 1),
        TrainingSample.from_dense(np.ones((20, 8, 8), np.float32), policy, -1.0, "a", 1, 1),
    ])
    before = next(engine.model.parameters()).detach().clone()
    losses = engine._optimize()
    after = next(engine.model.parameters()).detach()
    assert losses and losses["total_loss"] > 0
    assert not torch.equal(before, after)


def test_checkpoint_save_and_load_consistency(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(tmp_path, repository, model, optimizer, 2, 3, 4, 5, {"seed": 1})
    restored, payload = load_checkpoint(path, torch.device("cpu"))
    assert payload["version"] == 2
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)
    assert repository.get_checkpoint(2)["file_size"] == path.stat().st_size


def test_candidate_autosave_does_not_create_model_versions(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters())
    first = save_training_state(tmp_path, model, optimizer, 3, 10, 20, 30, 8, {"seed": 1})
    second = save_training_state(tmp_path, model, optimizer, 3, 11, 24, 31, 8, {"seed": 1})
    assert first == second == tmp_path / "training-latest.pt"
    assert repository.list_checkpoints() == []
    _, payload = load_checkpoint(second, torch.device("cpu"))
    assert payload["iteration"] == 11
    assert payload["champion_version"] == 3


def test_evaluation_uses_paired_diverse_openings_and_reports_termination(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    evaluation = evaluate_checkpoint(
        model,
        1,
        torch.device("cpu"),
        repository,
        "random",
        games=4,
        simulations=1,
        max_plies=8,
        opening_plies=2,
        seed=17,
    )
    details = evaluation["details"]
    assert details[0]["moves"][:2] == details[1]["moves"][:2]
    assert details[2]["moves"][:2] == details[3]["moves"][:2]
    assert details[0]["moves"][:2] != details[2]["moves"][:2]
    assert all(item["termination"] for item in details)
    assert 0 <= evaluation["score"] <= 1


def test_opening_and_material_bootstrap_are_deterministic() -> None:
    assert [move.uci() for move in generate_opening(5, 4)] == [move.uci() for move in generate_opening(5, 4)]
    white_up_queen = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black_up_rook = chess.Board("3rk3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert material_value(white_up_queen) > 0
    assert material_value(black_up_rook) < 0


