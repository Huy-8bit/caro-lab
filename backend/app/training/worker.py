"""Stateless self-play / evaluation worker.

A worker owns nothing durable: it leases one job at a time from the backend,
downloads the model snapshots the job names, plays the game, and posts the
result. While playing it heartbeats so the lease stays alive; if it is told
the job is no longer active it abandons the game. On SIGTERM/SIGINT (for
example ``docker compose up --scale selfplay-worker=4`` shrinking from 6) it
releases the job so the backend requeues it immediately instead of waiting
for the lease to expire.
"""

from __future__ import annotations

import os
import signal
import socket
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import chess
import httpx
import torch

from app.chess.mcts import MCTS
from app.evaluation.arena import play_match_game
from app.training.distributed import encode_samples, load_worker_model
from app.training.network import choose_device
from app.training.self_play import GameAborted, SampleWeights, play_self_play_game


COORDINATOR_URL = os.getenv("CARO_COORDINATOR_URL", "http://backend:8000")
WORKER_TOKEN = os.getenv("CARO_WORKER_TOKEN", "local-scale-token")
WORKER_ID = os.getenv("CARO_WORKER_ID", socket.gethostname())
WORKER_DEVICE = os.getenv("CARO_WORKER_DEVICE", "cpu")
POLL_SECONDS = float(os.getenv("CARO_WORKER_POLL_SECONDS", "0.5"))
HEARTBEAT_SECONDS = float(os.getenv("CARO_WORKER_HEARTBEAT_SECONDS", "5"))
ALIVE_FILE = Path(os.getenv("CARO_WORKER_ALIVE_FILE", "/tmp/caro-worker-alive"))
MODEL_CACHE_SIZE = 3


class Worker:
    def __init__(self, client: httpx.Client, device: torch.device, worker_id: str = WORKER_ID) -> None:
        self.client = client
        self.worker_id = worker_id
        self.device = device
        self.models: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self.stopping = False
        self.job: dict[str, Any] | None = None
        self._last_heartbeat = 0.0
        self._job_active = True

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self, *_: Any) -> None:
        self.stopping = True

    def alive(self) -> None:
        try:
            ALIVE_FILE.touch()
        except OSError:
            pass

    def model(self, key: str) -> torch.nn.Module:
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]
        response = self.client.get(f"/api/workers/models/{key}")
        response.raise_for_status()
        model, _ = load_worker_model(response.content, self.device)
        self.models[key] = model
        while len(self.models) > MODEL_CACHE_SIZE:
            self.models.popitem(last=False)
        print(f"worker={self.worker_id} loaded model={key}", flush=True)
        return model

    def should_abort(self) -> bool:
        """Called once per ply: heartbeat the lease and honour shutdown."""
        self.alive()
        if self.stopping:
            return True
        now = time.monotonic()
        if self.job is not None and now - self._last_heartbeat >= HEARTBEAT_SECONDS:
            self._last_heartbeat = now
            try:
                response = self.client.post(
                    "/api/workers/heartbeat",
                    json={"worker_id": self.worker_id, "job_id": self.job["job_id"]},
                    timeout=5.0,
                )
                self._job_active = bool(response.json().get("active", False))
            except (httpx.HTTPError, ValueError):
                pass  # transient: keep playing, the lease timeout is the backstop
        return not self._job_active

    def post_event(self, event: dict[str, Any]) -> None:
        assert self.job is not None
        try:
            self.client.post(
                "/api/workers/events",
                json={"worker_id": self.worker_id, "job_id": self.job["job_id"], "event": event},
                timeout=3.0,
            )
        except httpx.HTTPError:
            pass

    def release(self) -> None:
        if self.job is None:
            return
        try:
            self.client.post(
                "/api/workers/release",
                json={"worker_id": self.worker_id, "job_id": self.job["job_id"]},
                timeout=5.0,
            )
        except httpx.HTTPError:
            pass

    def submit(self, payload: dict[str, Any]) -> None:
        assert self.job is not None
        body = {"worker_id": self.worker_id, "job_id": self.job["job_id"], **payload}
        for attempt in range(10):
            try:
                response = self.client.post("/api/workers/results", json=body, timeout=300.0)
            except httpx.HTTPError:
                time.sleep(min(30, 2**attempt))
                continue
            if response.status_code in {200, 409}:
                return
            response.raise_for_status()

    # -- jobs ----------------------------------------------------------------

    def play_selfplay(self, job: dict[str, Any]) -> dict[str, Any]:
        config = job["config"]
        model = self.model(job["model_key"])
        search = MCTS(
            model,
            self.device,
            simulations=int(config["mcts_simulations"]),
            c_puct=float(config["c_puct"]),
            dirichlet_alpha=float(config["dirichlet_alpha"]),
            dirichlet_epsilon=float(config["dirichlet_epsilon"]),
            draw_contempt=float(config["draw_contempt"]),
            seed=int(job["seed"]),
            policy_mode=str(config.get("policy_target", "visits")),
            value_scale=float(config.get("completed_q_scale", 0.1)),
        )
        sample_rate = max(1, int(config["event_sample_rate"]))

        def report(event: dict[str, Any]) -> None:
            if event.get("type") == "move" and int(event["ply"]) % sample_rate == 0:
                self.post_event(event)

        result = play_self_play_game(
            search=search,
            iteration=int(job["iteration"]),
            model_version=int(job["model_version"]),
            max_moves=int(config["max_game_moves"]),
            temperature_moves=int(
                config.get("endgame_temperature_moves", 0) if job.get("start_fen") else config["temperature_moves"]
            ),
            temperature=float(config["temperature"]),
            temperature_final=float(config.get("temperature_final", 0.0)),
            opening_plies=int(config.get("selfplay_opening_plies", 0)),
            opening_seed=int(job["seed"]),
            sample_weights=SampleWeights.from_settings(config),
            start_fen=job.get("start_fen"),
            start_family=job.get("start_family"),
            event_callback=report,
            abort_callback=self.should_abort,
        )
        return {"record": result.record, "samples_blob": encode_samples(result.samples), "summary": result.record["result"]}

    def play_evaluation(self, job: dict[str, Any]) -> dict[str, Any]:
        config = job["config"]
        candidate = self.model(job["model_keys"]["candidate"])
        opponent = self.model(job["model_keys"]["opponent"])
        sample_rate = max(1, int(config.get("event_sample_rate", 1)))

        def on_move(game_index: int, board: chess.Board, move: chess.Move) -> None:
            plies = len(board.move_stack)
            if plies % sample_rate == 0:
                self.post_event({"type": "evaluation_move", "plies": plies, "fen": board.fen(), "uci": move.uci()})

        detail = play_match_game(
            candidate,
            opponent,
            self.device,
            int(job["game_index"]),
            seed=int(job["seed"]),
            simulations=int(config["simulations"]),
            max_plies=int(config["max_plies"]),
            opening_plies=int(job["opening_plies"]),
            c_puct=float(config["c_puct"]),
            draw_contempt=float(config["draw_contempt"]),
            policy_mode=str(config.get("policy_target", "visits")),
            value_scale=float(config.get("completed_q_scale", 0.1)),
            move_callback=on_move,
            abort_callback=self.should_abort,
        )
        return {"detail": detail, "summary": f"{detail['result']} ({detail['termination']})"}

    def run_once(self) -> bool:
        self.alive()
        response = self.client.post("/api/workers/claim", json={"worker_id": self.worker_id})
        if response.status_code == 204:
            return False
        response.raise_for_status()
        job = self.job = response.json()
        self._job_active = True
        self._last_heartbeat = time.monotonic()
        try:
            if job["kind"] == "evaluation":
                payload = self.play_evaluation(job)
            else:
                payload = self.play_selfplay(job)
        except (GameAborted, InterruptedError):
            if self.stopping:
                self.release()
            print(f"worker={self.worker_id} abandoned job={job['job_id'][:8]}", flush=True)
            self.job = None
            return True
        except BaseException:
            self.release()
            raise
        summary = payload.pop("summary")
        self.submit(payload)
        print(f"worker={self.worker_id} completed {job['kind']} job={job['job_id'][:8]} result={summary}", flush=True)
        self.job = None
        return True


def run() -> None:
    torch.set_num_threads(max(1, int(os.getenv("CARO_TORCH_THREADS", "1"))))
    device = choose_device(WORKER_DEVICE)
    headers = {"X-Caro-Worker-Token": WORKER_TOKEN}
    with httpx.Client(base_url=COORDINATOR_URL, headers=headers, timeout=30.0) as client:
        worker = Worker(client, device)
        signal.signal(signal.SIGTERM, worker.request_stop)
        signal.signal(signal.SIGINT, worker.request_stop)
        print(f"worker={WORKER_ID} coordinator={COORDINATOR_URL} device={device}", flush=True)
        while not worker.stopping:
            try:
                if not worker.run_once():
                    time.sleep(POLL_SECONDS)
            except Exception as exc:
                print(f"worker={WORKER_ID} error={type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                worker.job = None
                time.sleep(2)
        print(f"worker={WORKER_ID} stopped", flush=True)


if __name__ == "__main__":
    run()
