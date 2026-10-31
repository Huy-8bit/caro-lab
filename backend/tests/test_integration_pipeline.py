from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from app.chess.mcts import MCTS
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint, save_checkpoint
from app.training.engine import EVALUATION_STATE_KEY, TrainingEngine
from app.training.self_play import play_self_play_game


def tiny_config(**overrides) -> TrainingConfig:
    values = dict(
        games_per_iteration=2, parallel_workers=2, mcts_simulations=2, max_game_moves=10, batch_size=8,
        training_updates=2, checkpoint_interval=1, promotion_games=2, evaluation_simulations=2,
        evaluation_opening_plies=2, selfplay_opening_plies=2, channels=8, residual_blocks=0,
        replay_capacity=500, promotion_threshold=0.5, gate_min_decisive_rate=0.0,
        gate_max_repetition_rate=1.0, stop_after_promotion=True, autosave_interval=1, worker_job_timeout=2,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def wait_until_idle(engine: TrainingEngine, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while engine._thread and engine._thread.is_alive():
        if time.monotonic() > deadline:
            engine.stop(wait=True)
            raise AssertionError("training did not stop in time")
        time.sleep(0.05)


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


def test_local_self_play_replay_optimize_promotion_loop(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "caro.db")
    engine = TrainingEngine(repository, EventBroker(), tmp_path / "checkpoints", tiny_config())
    engine.start()
    wait_until_idle(engine)
    assert engine.status == "idle" and engine.last_error is None
    checkpoints = {item["version"]: item for item in repository.list_checkpoints()}
    assert checkpoints[1]["evaluation"]["status"] == "bootstrap"
    evaluations = repository.list_evaluations()
    assert evaluations and evaluations[-1]["kind"] == "promotion"
    first = evaluations[-1]
    assert first["title"] == "Candidate v2 vs Champion v1" and len(first["details"]) == 2
    assert {item["model_color"] for item in first["details"]} == {"white", "black"}
    assert first["details"][0]["moves"][:2] == first["details"][1]["moves"][:2]  # paired opening
    if any(item["status"] == "promoted" for item in evaluations):
        assert 2 in checkpoints and engine.model_version == 2  # stopped right after promotion
    metrics = repository.list_metrics()
    assert metrics and all(row["selfplay_seconds"] is not None and row["optimize_seconds"] is not None for row in metrics)
    games_per_iteration = Counter(game["iteration"] for game in repository.list_games(500))
    assert set(games_per_iteration.values()) == {2}
    progress = repository.get_state(EVALUATION_STATE_KEY)
    assert progress["games_completed"] == progress["games_total"] == 2
    analytics = engine.analytics()
    assert analytics["iterations"] and "rolling" in analytics["iterations"][-1]


def test_distributed_workers_with_a_crash_complete_every_job_once(tmp_path: Path, monkeypatch) -> None:
    from app.main import app
    from app.training import worker as worker_module

    repository = Repository(tmp_path / "caro.db")
    engine = TrainingEngine(
        repository, EventBroker(), tmp_path / "checkpoints",
        tiny_config(self_play_mode="distributed", games_per_iteration=4, promotion_games=4, checkpoint_interval=1),
    )
    original = (app.state.engine, app.state.repository)
    app.state.engine, app.state.repository = engine, repository
    try:
        _run_distributed(app, engine, worker_module, monkeypatch, tmp_path)
    finally:
        app.state.engine, app.state.repository = original
    _assert_distributed_results(engine, repository)


def _run_distributed(app, engine, worker_module, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(worker_module, "HEARTBEAT_SECONDS", 0.2)
    monkeypatch.setattr(worker_module, "ALIVE_FILE", tmp_path / "alive")
    headers = {"X-Caro-Worker-Token": "local-scale-token"}
    stop = threading.Event()
    with TestClient(app, headers=headers) as client:
        engine.start()

        # A worker that leases one job and dies without a heartbeat.
        crashed = None
        while crashed is None and not stop.is_set():
            response = client.post("/api/workers/claim", json={"worker_id": "crashed"})
            if response.status_code == 200:
                crashed = response.json()["job_id"]
            else:
                time.sleep(0.05)

        def run(name: str) -> None:
            worker = worker_module.Worker(client, torch.device("cpu"), worker_id=name)
            while not stop.is_set():
                if not worker.run_once():
                    time.sleep(0.05)

        threads = [threading.Thread(target=run, args=(f"live-{index}",), daemon=True) for index in range(3)]
        for thread in threads:
            thread.start()
        wait_until_idle(engine, timeout=180)
        stop.set()
        for thread in threads:
            thread.join(timeout=30)


def _assert_distributed_results(engine, repository) -> None:
    assert engine.last_error is None
    games = repository.list_games(500)
    # Every iteration consumes exactly games_per_iteration results. A game still
    # running at an iteration boundary is counted in the next iteration but keeps
    # the label of the iteration (model) it was played with.
    assert games and len(games) == 4 * engine.iteration, (len(games), engine.iteration)
    assert all(1 <= game["iteration"] <= engine.iteration for game in games)
    assert len({game["id"] for game in games}) == len(games)
    evaluation = repository.list_evaluations()[-1]
    assert sorted(item["game_index"] for item in evaluation["details"]) == [0, 1, 2, 3]
    # The crashed worker's lease either expired and was requeued, or the
    # stream's look-ahead jobs finished the run first; the counts above show it
    # neither blocked the run nor lost or duplicated a game. Lease expiry
    # itself is covered in test_distributed_jobs.
