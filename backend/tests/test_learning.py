from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from app.chess.mcts import MCTS, SearchNode
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


def test_move_selection_avoids_claiming_threefold_when_progress_is_available() -> None:
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    repeating = chess.Move.from_uci("f6g8")
    progressive = chess.Move.from_uci("h8g8")
    root = SearchNode(
        children={
            repeating: SearchNode(visit_count=100),
            progressive: SearchNode(visit_count=1),
        }
    )
    search = MCTS(tiny_model(), torch.device("cpu"), simulations=1, seed=2)
    assert search.select_move(root, 0, board) == progressive


def test_replay_buffer_sampling() -> None:
    buffer = ReplayBuffer(3, seed=1)
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    samples = [TrainingSample(np.zeros((20, 8, 8), np.float32), policy, 0, "g", i, 1) for i in range(5)]
    buffer.extend(samples)
    assert len(buffer) == 3
    assert len(buffer.sample(10)) == 3


def test_distributed_sample_payload_round_trip() -> None:
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[3] = 1.0
    samples = [
        TrainingSample(np.zeros((20, 8, 8), np.float32), policy, -0.05, "game", 2, 4),
        TrainingSample(np.ones((20, 8, 8), np.float32), policy, 1.0, "game", 3, 4),
    ]
    record = {"id": "game", "moves": [], "result": "1-0"}
    restored = decode_result(record, encode_samples(samples), iteration=4)
    assert len(restored.samples) == 2
    assert restored.samples[0].value == pytest.approx(-0.05)
    assert np.array_equal(restored.samples[1].state, samples[1].state)
    assert np.array_equal(restored.samples[0].policy, policy)


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


def test_rejected_candidate_restores_champion_without_new_version(tmp_path: Path, monkeypatch) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    champion_path = save_checkpoint(tmp_path / "checkpoints", repository, model, optimizer, 1, 0, 0, 0, {})
    config = TrainingConfig(
        channels=8,
        residual_blocks=0,
        promotion_games=2,
        promotion_threshold=0.55,
        evaluation_simulations=1,
    )
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", config)
    with torch.no_grad():
        next(engine.model.parameters()).add_(1)
    engine.iteration = 10

    monkeypatch.setattr(
        "app.training.engine.evaluate_checkpoint",
        lambda *args, **kwargs: {
            "model_a": 2,
            "model_b": "v1",
            "games": 2,
            "wins": 0,
            "draws": 2,
            "losses": 0,
            "score": 0.5,
            "details": [],
        },
    )
    engine._promote_candidate()
    restored, _ = load_checkpoint(champion_path, torch.device("cpu"))
    assert engine.model_version == 1
    assert repository.get_checkpoint(2) is None
    assert engine.last_promotion and engine.last_promotion["status"] == "rejected"
    assert not engine._stop.is_set()
    for expected, actual in zip(restored.parameters(), engine.model.parameters(), strict=True):
        assert torch.equal(expected.to(actual.device), actual)


def test_successful_promotion_requests_configured_stop(tmp_path: Path, monkeypatch) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(tmp_path / "checkpoints", repository, model, optimizer, 1, 0, 0, 0, {})
    config = TrainingConfig(
        channels=8,
        residual_blocks=0,
        promotion_games=2,
        promotion_threshold=0.55,
        evaluation_simulations=1,
        stop_after_promotion=True,
    )
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", config)
    engine.iteration = 10

    monkeypatch.setattr(
        "app.training.engine.evaluate_checkpoint",
        lambda *args, **kwargs: {
            "id": "promotion-test",
            "model_a": 2,
            "model_b": "v1",
            "games": 2,
            "wins": 1,
            "draws": 1,
            "losses": 0,
            "score": 0.75,
            "created_at": "2026-10-30T00:00:00+00:00",
            "details": [],
        },
    )

    engine._promote_candidate()

    assert engine.model_version == 2
    assert repository.get_checkpoint(2) is not None
    assert engine.last_promotion and engine.last_promotion["status"] == "promoted"
    assert engine.status == "stopping"
    assert engine._stop.is_set()
    assert any("stopping training as configured" in item["message"] for item in engine.logs)
