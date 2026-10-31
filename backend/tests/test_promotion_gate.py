from __future__ import annotations

from pathlib import Path

import torch

from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint, save_checkpoint
from app.training.engine import EVALUATION_STATE_KEY, TrainingEngine
from app.training.network import NetworkSpec, PolicyValueNetwork


def tiny_model() -> PolicyValueNetwork:
    return PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=0))


def details(results: list[tuple[str, str]]) -> list[dict]:
    return [
        {"game": index + 1, "game_index": index, "model_color": "white" if index % 2 == 0 else "black",
         "result": result, "termination": termination, "plies": 80, "opening_plies": 4, "moves": []}
        for index, (result, termination) in enumerate(results)
    ]


def engine_with_champion(tmp_path: Path, **overrides) -> tuple[TrainingEngine, Repository, Path]:
    repository = Repository(tmp_path / "db.sqlite")
    model = tiny_model()
    champion_path = save_checkpoint(
        tmp_path / "checkpoints", repository, model, torch.optim.AdamW(model.parameters()), 1, 0, 0, 0, {},
        {"status": "bootstrap", "score": 1.0},
    )
    config = TrainingConfig(channels=8, residual_blocks=0, promotion_games=4, evaluation_simulations=1, **overrides)
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", config)
    with torch.no_grad():
        next(engine.model.parameters()).add_(1)
    engine.iteration = 10
    return engine, repository, champion_path


def run_gate(engine: TrainingEngine, match: list[dict], monkeypatch) -> None:
    def fake_match(candidate, champion, progress):
        for item in match:
            progress.game_finished(item)
        return [dict(item) for item in match]

    monkeypatch.setattr(engine, "_play_promotion_match", fake_match)
    engine._promote_candidate()


def test_winning_candidate_is_promoted_and_stops_training(tmp_path: Path, monkeypatch) -> None:
    engine, repository, _ = engine_with_champion(tmp_path)
    run_gate(engine, details([("win", "checkmate"), ("win", "checkmate"), ("draw", "fifty moves"), ("loss", "checkmate")]), monkeypatch)
    assert engine.model_version == 2
    checkpoint = repository.get_checkpoint(2)
    assert checkpoint["evaluation"]["status"] == "promoted"
    assert checkpoint["evaluation"]["opponent"] == "v1"
    assert repository.get_checkpoint(1) is not None  # previous champion kept for v2-vs-v1 replays
    assert engine._stop.is_set() and engine.status == "stopping"
    evaluation = repository.list_evaluations()[0]
    assert evaluation["kind"] == "promotion" and evaluation["status"] == "promoted"
    assert evaluation["title"] == "Candidate v2 vs Champion v1"


def test_high_score_with_too_many_repetitions_is_rejected(tmp_path: Path, monkeypatch) -> None:
    engine, repository, champion_path = engine_with_champion(tmp_path, gate_max_repetition_rate=0.5)
    match = details([("win", "checkmate"), ("draw", "threefold repetition"), ("draw", "threefold repetition"), ("draw", "threefold repetition")])
    run_gate(engine, match, monkeypatch)  # score 62.5% >= 55%, but 75% repetitions
    assert engine.model_version == 1
    assert repository.get_checkpoint(2) is None
    assert engine.last_promotion["status"] == "rejected"
    assert any("repetition rate" in reason for reason in engine.last_promotion["reasons"])
    assert not engine._stop.is_set()
    stored = repository.list_evaluations()[0]
    assert stored["status"] == "rejected" and stored["reasons"]
    # Score was above reject_reset_score, so the trainer keeps learning from its weights.
    champion, _ = load_checkpoint(champion_path, torch.device("cpu"))
    assert not torch.equal(next(champion.parameters()), next(engine.model.parameters()).cpu())


def test_low_decisive_rate_is_rejected(tmp_path: Path, monkeypatch) -> None:
    engine, _, _ = engine_with_champion(tmp_path, gate_min_decisive_rate=0.3, promotion_threshold=0.5)
    run_gate(engine, details([("win", "checkmate")] + [("draw", "max move safety limit")] * 5), monkeypatch)
    assert engine.model_version == 1
    assert any("decisive-game rate" in reason for reason in engine.last_promotion["reasons"])


def test_badly_losing_candidate_resets_to_champion_weights(tmp_path: Path, monkeypatch) -> None:
    engine, repository, champion_path = engine_with_champion(tmp_path, reject_reset_score=0.35)
    run_gate(engine, details([("loss", "checkmate")] * 4), monkeypatch)
    assert engine.model_version == 1
    champion, _ = load_checkpoint(champion_path, torch.device("cpu"))
    for expected, actual in zip(champion.parameters(), engine.model.parameters(), strict=True):
        assert torch.equal(expected, actual.cpu())
    assert "restarted from champion" in repository.list_evaluations()[0]["action"]


def test_progress_is_durable_and_reports_counts(tmp_path: Path, monkeypatch) -> None:
    engine, repository, _ = engine_with_champion(tmp_path)
    run_gate(engine, details([("win", "checkmate"), ("loss", "checkmate"), ("draw", "stalemate"), ("loss", "checkmate")]), monkeypatch)
    stored = repository.get_state(EVALUATION_STATE_KEY)
    assert stored["status"] == "rejected"
    assert stored["title"] == "Candidate v2 vs Champion v1"
    assert (stored["games_completed"], stored["games_total"]) == (4, 4)
    assert (stored["wins"], stored["draws"], stored["losses"]) == (1, 1, 2)
    assert stored["terminations"] == {"checkmate": 3, "stalemate": 1}
    assert stored["decision"]["reasons"]
    # A restarted backend still serves the last evaluation state.
    restarted = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", engine.config)
    assert restarted.evaluation_state["id"] == stored["id"]


def test_bootstrap_is_labelled_and_never_claims_a_score(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "db.sqlite")
    config = TrainingConfig(channels=8, residual_blocks=0)
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", config)
    engine.iteration = 10
    engine._promote_candidate()
    checkpoint = repository.get_checkpoint(1)
    assert checkpoint["evaluation"]["status"] == "bootstrap"
    assert "score" not in checkpoint["evaluation"]
    assert engine.last_promotion == {"status": "bootstrap", "version": 1, "iteration": 10}
    assert repository.list_evaluations() == []
    assert not engine._stop.is_set()


def test_legacy_bootstrap_rows_are_shown_without_their_fake_score(tmp_path: Path) -> None:
    _, repository, _ = engine_with_champion(tmp_path)  # stored as {"status": "bootstrap", "score": 1.0}
    evaluation = repository.get_checkpoint(1)["evaluation"]
    assert evaluation["status"] == "bootstrap" and "score" not in evaluation


def test_promotion_games_are_even_for_colour_pairs(tmp_path: Path) -> None:
    engine, _, _ = engine_with_champion(tmp_path)
    engine.config.promotion_games = 41
    assert engine.promotion_games() == 42


def test_mismatched_network_never_silently_resets_the_run(tmp_path: Path) -> None:
    import pytest

    engine, repository, _ = engine_with_champion(tmp_path)
    with pytest.raises(ValueError, match="separate data/checkpoint directory"):
        engine.start(TrainingConfig(channels=16, residual_blocks=0))
    assert engine.model_version == 1 and engine.status == "idle"
    assert repository.get_checkpoint(1) is not None
