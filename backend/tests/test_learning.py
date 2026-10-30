from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from app.chess.mcts import MCTS
from app.chess.move_index import POLICY_SIZE
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint, save_checkpoint
from app.training.engine import TrainingEngine
from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.replay import ReplayBuffer, TrainingSample


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
    samples = [TrainingSample(np.zeros((20, 8, 8), np.float32), policy, 0, "g", i, 1) for i in range(5)]
    buffer.extend(samples)
    assert len(buffer) == 3
    assert len(buffer.sample(10)) == 3


def test_optimizer_changes_real_weights(tmp_path: Path) -> None:
    config = TrainingConfig(channels=8, residual_blocks=0, batch_size=2, replay_capacity=10)
    engine = TrainingEngine(Repository(tmp_path / "db.sqlite"), EventBroker(), tmp_path / "checkpoints", config)
    policy = np.zeros(POLICY_SIZE, np.float32)
    policy[0] = 1
    engine.replay.extend([
        TrainingSample(np.zeros((20, 8, 8), np.float32), policy, 1.0, "a", 0, 1),
        TrainingSample(np.ones((20, 8, 8), np.float32), policy, -1.0, "a", 1, 1),
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
