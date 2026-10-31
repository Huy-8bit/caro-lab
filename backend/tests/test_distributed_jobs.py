"""Scaling workers up or down must not duplicate or lose jobs."""

from __future__ import annotations

import random
import threading
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.training.distributed import JobBoard, encode_samples
from app.training.replay import TrainingSample


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def fill(board: JobBoard, count: int, batch: str = "b") -> list[str]:
    board.open_batch(batch)
    return [board.submit(batch, {"kind": "selfplay", "n": index}) for index in range(count)]


def drain(board: JobBoard, batch: str = "b") -> list[str]:
    done = []
    while (item := board.next_result(batch, timeout=0.01)) is not None:
        done.append(item[0])
    return done


def test_each_job_is_leased_to_one_worker_at_a_time() -> None:
    board = JobBoard(lease_timeout=60)
    ids = fill(board, 3)
    leased = [board.claim(f"w{index}")["job_id"] for index in range(3)]
    assert sorted(leased) == sorted(ids)
    assert board.claim("w9") is None


def test_dead_worker_lease_expires_and_is_requeued_once() -> None:
    clock = FakeClock()
    board = JobBoard(lease_timeout=30, clock=clock)
    (job_id,) = fill(board, 1)
    assert board.claim("dead")["job_id"] == job_id
    clock.now = 20
    assert board.expire() == []
    clock.now = 31
    assert board.expire() == [job_id]
    assert board.expire() == []
    assert board.claim("alive")["job_id"] == job_id
    assert board.complete("alive", job_id, "result")
    assert drain(board) == [job_id]


def test_heartbeat_keeps_a_long_game_leased() -> None:
    clock = FakeClock()
    board = JobBoard(lease_timeout=30, clock=clock)
    (job_id,) = fill(board, 1)
    board.claim("slow")
    for second in range(0, 300, 10):
        clock.now = second
        assert board.heartbeat("slow", job_id)
        assert board.expire() == []
    assert not board.heartbeat("someone-else", job_id)


def test_late_result_from_presumed_dead_worker_is_accepted_once() -> None:
    clock = FakeClock()
    board = JobBoard(lease_timeout=10, clock=clock)
    (job_id,) = fill(board, 1)
    board.claim("a")
    clock.now = 11
    board.expire()
    board.claim("b")
    assert board.complete("a", job_id, "from-a")
    assert not board.complete("b", job_id, "from-b")
    assert not board.heartbeat("b", job_id)  # b is told to abandon
    assert drain(board) == [job_id]


def test_restarted_worker_with_same_id_requeues_its_old_job() -> None:
    board = JobBoard(lease_timeout=60)
    first, second = fill(board, 2)
    assert board.claim("container-1")["job_id"] == first
    # Container restarted (same hostname) and claims again: the abandoned job
    # goes back to the front of the queue and is handed out immediately.
    assert board.claim("container-1")["job_id"] == first
    assert board.claim("container-2")["job_id"] == second


def test_graceful_release_requeues_immediately() -> None:
    board = JobBoard(lease_timeout=600)
    (job_id,) = fill(board, 1)
    board.claim("scaled-down")
    assert board.release("scaled-down", job_id)
    assert board.claim("survivor")["job_id"] == job_id


def test_closed_batch_rejects_stale_results() -> None:
    board = JobBoard()
    (job_id,) = fill(board, 1)
    board.claim("w")
    board.close_batch("b")
    assert not board.complete("w", job_id, "late")


def test_concurrent_scaling_never_duplicates_or_loses_jobs() -> None:
    clock = FakeClock()
    board = JobBoard(lease_timeout=5, clock=clock)
    ids = fill(board, 200)
    rng = random.Random(7)
    lock = threading.Lock()
    accepted: list[str] = []

    def worker(name: str, crash_rate: float) -> None:
        local = random.Random(name)
        while True:
            job = board.claim(name)
            if job is None:
                with lock:
                    if len(accepted) == len(ids):
                        return
                board.expire()
                continue
            fate = local.random()
            if fate < crash_rate:
                continue  # worker vanished mid-game: lease left to expire
            if fate < crash_rate * 2:
                board.release(name, job["job_id"])  # scaled down gracefully
                continue
            if board.complete(name, job["job_id"], name):
                with lock:
                    accepted.append(job["job_id"])
            # duplicate delivery attempt must be rejected
            assert not board.complete(name, job["job_id"], name)

    def ticker(stop: threading.Event) -> None:
        while not stop.is_set():
            clock.now += 1
            stop.wait(0.001)

    stop = threading.Event()
    clock_thread = threading.Thread(target=ticker, args=(stop,))
    clock_thread.start()
    threads = [threading.Thread(target=worker, args=(f"w{index}", rng.choice([0.0, 0.1, 0.2]))) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    stop.set()
    clock_thread.join()
    results = drain(board)
    assert sorted(results) == sorted(ids)
    assert sorted(accepted) == sorted(ids)


def _sample() -> TrainingSample:
    policy = np.zeros(20_480, np.float32)
    policy[11] = 1.0
    return TrainingSample.from_dense(np.zeros((20, 8, 8), np.float32), policy, 1.0, "g", 0, 1)


def test_worker_http_protocol_end_to_end(tmp_path: Path, monkeypatch) -> None:
    from app.main import app

    engine = app.state.engine
    headers = {"X-Caro-Worker-Token": "local-scale-token"}
    with TestClient(app) as client:
        assert client.post("/api/workers/claim", json={"worker_id": "w1"}).status_code == 401
        monkeypatch.setattr(engine.config, "self_play_mode", "distributed")
        monkeypatch.setattr(engine, "status", "running")
        engine.jobs.open_batch("b")
        job_id = engine.jobs.submit("b", {"kind": "selfplay", "iteration": 3, "model_version": 1, "revision": 3})
        job = client.post("/api/workers/claim", json={"worker_id": "w1"}, headers=headers).json()
        assert job["job_id"] == job_id
        assert client.post("/api/workers/heartbeat", json={"worker_id": "w1", "job_id": job_id}, headers=headers).json() == {"active": True}
        assert client.post("/api/workers/heartbeat", json={"worker_id": "w2", "job_id": job_id}, headers=headers).json() == {"active": False}
        record = {"id": "remote-game", "result": "1-0", "moves": []}
        body = {"worker_id": "w1", "job_id": job_id, "record": record, "samples_blob": encode_samples([_sample()])}
        assert client.post("/api/workers/results", json=body, headers=headers).status_code == 200
        assert client.post("/api/workers/results", json=body, headers=headers).status_code == 409
        _, result = engine.jobs.next_result("b", timeout=1)
        assert result.record["iteration"] == 3 and len(result.samples) == 1
        engine.jobs.close_batch("b")
