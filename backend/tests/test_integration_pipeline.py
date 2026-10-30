from __future__ import annotations

from pathlib import Path

import torch

from app.chess.mcts import MCTS
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint, save_checkpoint
from app.training.engine import TrainingEngine
from app.training.self_play import play_self_play_game


def test_complete_self_play_train_persist_checkpoint_pipeline(tmp_path: Path) -> None:
    config = TrainingConfig(
        games_per_iteration=1, mcts_simulations=1, max_game_moves=2, batch_size=2,
        training_updates=1, channels=8, residual_blocks=0, replay_capacity=32,
    )
    repository = Repository(tmp_path / "caro.db")
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", config)
    search = MCTS(engine.model, engine.device, simulations=1, seed=4)
    result = play_self_play_game(search, 1, 0, 2, 1, 1.0)
    assert result.record["moves"]
    assert all(move["uci"] for move in result.record["moves"])
    repository.save_game(result.record)
    engine.replay.extend(result.samples)
    before = {name: value.detach().clone() for name, value in engine.model.state_dict().items()}
    metric = engine._optimize()
    assert metric and metric["policy_loss"] > 0
    assert any(not torch.equal(before[name], value) for name, value in engine.model.state_dict().items())
    checkpoint = save_checkpoint(tmp_path / "checkpoints", repository, engine.model, engine.optimizer, 1, 1, 1, engine.step, config.to_dict())
    loaded, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert payload["games"] == 1
    assert repository.get_game(result.record["id"])["pgn"] == result.record["pgn"]
    assert next(loaded.parameters()).shape == next(engine.model.parameters()).shape
