from __future__ import annotations

import copy
import platform
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from app.chess.mcts import MCTS
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.evaluation.arena import evaluate_checkpoint
from app.storage.repository import Repository
from app.training.checkpoints import load_checkpoint, save_checkpoint, save_training_state
from app.training.distributed import decode_result, save_worker_model
from app.training.network import NetworkSpec, PolicyValueNetwork, choose_device
from app.training.replay import ReplayBuffer
from app.training.self_play import play_self_play_game


class TrainingEngine:
    def __init__(
        self,
        repository: Repository,
        broker: EventBroker,
        checkpoint_dir: str | Path,
        config: TrainingConfig | None = None,
    ) -> None:
        self.repository = repository
        self.broker = broker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or TrainingConfig()
        self.device = choose_device(self.config.device)
        self.model = PolicyValueNetwork(NetworkSpec(self.config.channels, self.config.residual_blocks)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.replay = ReplayBuffer(self.config.replay_capacity, self.config.random_seed)
        persisted = self.repository.get_state("training") or {}
        self.iteration = int(persisted.get("iteration", 0))
        self.step = int(persisted.get("step", 0))
        self.model_version = int(persisted.get("model_version", 0))
        self._cycle_start_iteration = int(persisted.get("cycle_start_iteration", self.iteration))
        self.games_completed = int(self.repository.game_summary()["total"])
        self._session_start_games = self.games_completed
        self.status = "idle"
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.phase = "idle"
        self.last_promotion: dict[str, Any] | None = None
        self.active_games: dict[str, dict[str, Any]] = {}
        self.logs: deque[dict[str, str]] = deque(maxlen=300)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()
        self._remote_jobs: queue.Queue[dict[str, Any]] = queue.Queue()
        self._remote_results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._pending_remote: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self._worker_model_revision = -1
        self._worker_model_path = self.checkpoint_dir / "worker-current.pt"
        self._restore_latest()

    def _restore_latest(self) -> None:
        checkpoints = self.repository.list_checkpoints()
        if checkpoints:
            latest = checkpoints[0]
            path = Path(latest["path"])
            if path.exists():
                try:
                    model, payload = load_checkpoint(path, self.device)
                    self.model = model
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
                    )
                    if "optimizer_state" in payload:
                        self.optimizer.load_state_dict(payload["optimizer_state"])
                    self.model_version = int(payload["version"])
                    self.iteration = max(self.iteration, int(payload["iteration"]))
                    self.step = max(self.step, int(payload.get("step", 0)))
                    self._cycle_start_iteration = max(self._cycle_start_iteration, int(payload["iteration"]))
                    self._log("info", f"Restored champion model v{self.model_version}")
                except Exception as exc:  # pragma: no cover - defensive startup path
                    self._log("warning", f"Could not restore checkpoint: {exc}")

        candidate_path = self.checkpoint_dir / "training-latest.pt"
        if not candidate_path.exists():
            return
        try:
            candidate, payload = load_checkpoint(candidate_path, self.device)
            if int(payload.get("champion_version", payload["version"])) != self.model_version:
                return
            if int(payload["iteration"]) < self.iteration:
                return
            self.model = candidate
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            if "optimizer_state" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            self.iteration = int(payload["iteration"])
            self.step = max(self.step, int(payload.get("step", 0)))
            self._cycle_start_iteration = int(payload.get("cycle_start_iteration", self._cycle_start_iteration))
            self._log("info", f"Restored candidate progress after model v{self.model_version}")
        except Exception as exc:  # pragma: no cover - defensive startup path
            self._log("warning", f"Could not restore candidate state: {exc}")

    def _log(self, level: str, message: str) -> None:
        item = {"level": level, "message": message, "time": datetime.now(UTC).isoformat()}
        self.logs.appendleft(item)
        self.broker.publish("system", {"type": "log", **item})

    def start(self, config: TrainingConfig | None = None, resume_version: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Training is already running")
            if config:
                self.config = config
            self.device = choose_device(self.config.device)
            replaced_model = False
            if resume_version is not None:
                metadata = self.repository.get_checkpoint(resume_version)
                if not metadata:
                    raise ValueError(f"Checkpoint v{resume_version} does not exist")
                self.model, payload = load_checkpoint(metadata["path"], self.device)
                self.model_version = resume_version
                self.iteration = int(payload["iteration"])
                self.step = int(payload.get("step", 0))
                self._cycle_start_iteration = self.iteration
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
                )
                if "optimizer_state" in payload:
                    self.optimizer.load_state_dict(payload["optimizer_state"])
                replaced_model = True
            elif self.model.spec.channels != self.config.channels or self.model.spec.residual_blocks != self.config.residual_blocks:
                self.model = PolicyValueNetwork(NetworkSpec(self.config.channels, self.config.residual_blocks)).to(self.device)
                self.model_version = 0
                self.iteration = 0
                self.step = 0
                self._cycle_start_iteration = 0
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
                )
                replaced_model = True
            if not replaced_model:
                self.model = self.model.to(self.device)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.learning_rate
                    group["weight_decay"] = self.config.weight_decay
            self.replay = ReplayBuffer(self.config.replay_capacity, self.config.random_seed)
            self._reset_remote_jobs()
            self._worker_model_revision = -1
            self._stop.clear()
            self._pause.clear()
            self.last_error = None
            self.status = "running"
            self.phase = "self_play"
            self.started_at = time.monotonic()
            self._session_start_games = self.games_completed
            self._thread = threading.Thread(target=self._run, name="caro-training", daemon=True)
            self._thread.start()
            self._log("info", f"Training started on {self.device.type}")
            return self.snapshot()

    def _reset_remote_jobs(self) -> None:
        with self._lock:
            self._pending_remote.clear()
            while True:
                try:
                    self._remote_jobs.get_nowait()
                except queue.Empty:
                    break
            while True:
                try:
                    self._remote_results.get_nowait()
                except queue.Empty:
                    break

    def pause(self) -> dict[str, Any]:
        if self.status != "running":
            raise RuntimeError("Training is not running")
        self._pause.set()
        self.status = "paused"
        self._log("info", "Training paused at a safe move boundary")
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        if self.status != "paused":
            raise RuntimeError("Training is not paused")
        self._pause.clear()
        self.status = "running"
        self._log("info", "Training resumed")
        return self.snapshot()

    def stop(self, wait: bool = False) -> dict[str, Any]:
        self._stop.set()
        self._pause.clear()
        if self.status in {"running", "paused"}:
            self.status = "stopping"
        if wait and self._thread:
            self._thread.join(timeout=30)
        return self.snapshot()

    def _wait_if_paused(self) -> bool:
        return self._pause.is_set() and not self._stop.is_set()

    def _game_event(self, event: dict[str, Any]) -> None:
        game_id = event["game_id"]
        with self._lock:
            if event["type"] == "move":
                current = self.active_games.setdefault(
                    game_id,
                    {
                        "id": game_id,
                        "iteration": event["iteration"],
                        "model_version": event["model_version"],
                        "status": "in_progress",
                        "moves": [],
                    },
                )
                current["moves"].append(event)
                current["fen"] = event["fen"]
                current["last_move"] = event["uci"]
                current["updated_at"] = datetime.now(UTC).isoformat()
                if event["ply"] % max(1, self.config.event_sample_rate) == 0:
                    self.broker.publish(f"game:{game_id}", event)
                    self.broker.publish("live", event)
            else:
                current = self.active_games.setdefault(game_id, {"id": game_id})
                current.update({"status": "complete", "game": event["game"]})
                self.broker.publish(f"game:{game_id}", event)
                self.broker.publish("live", event)

    def _play_game(self, game_index: int):
        search = MCTS(
            self.model,
            self.device,
            simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            repetition_penalty=self.config.repetition_penalty,
            seed=self.config.random_seed + self.games_completed + game_index,
        )
        return play_self_play_game(
            search=search,
            iteration=self.iteration,
            model_version=self.model_version,
            max_moves=self.config.max_game_moves,
            temperature_moves=self.config.temperature_moves,
            temperature=self.config.temperature,
            draw_penalty=self.config.draw_penalty,
            event_callback=self._game_event,
            pause_callback=self._wait_if_paused,
        )

    def _worker_config(self) -> dict[str, Any]:
        return {
            "mcts_simulations": self.config.mcts_simulations,
            "max_game_moves": self.config.max_game_moves,
            "temperature_moves": self.config.temperature_moves,
            "temperature": self.config.temperature,
            "dirichlet_alpha": self.config.dirichlet_alpha,
            "dirichlet_epsilon": self.config.dirichlet_epsilon,
            "c_puct": self.config.c_puct,
            "repetition_penalty": self.config.repetition_penalty,
            "draw_penalty": self.config.draw_penalty,
            "event_sample_rate": self.config.event_sample_rate,
        }

    def _publish_worker_model(self) -> int:
        if (
            self._worker_model_revision < 0
            or self.iteration - self._worker_model_revision >= self.config.worker_sync_interval
            or not self._worker_model_path.exists()
        ):
            save_worker_model(
                str(self._worker_model_path),
                self.model,
                self.iteration,
                self.model_version,
            )
            self._worker_model_revision = self.iteration
            self._log("info", f"Published candidate revision {self._worker_model_revision} to scale workers")
        return self._worker_model_revision

    def _queue_remote_iteration(self) -> None:
        revision = self._publish_worker_model()
        for game_index in range(self.config.games_per_iteration):
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "revision": revision,
                "iteration": self.iteration,
                "model_version": self.model_version,
                "seed": self.config.random_seed + self.iteration * 1_000_003 + game_index,
                "config": self._worker_config(),
            }
            self._pending_remote[job_id] = {
                "job": job,
                "status": "queued",
                "lease_at": None,
                "worker_id": None,
            }
            self._remote_jobs.put(job)

    def _requeue_expired_jobs(self) -> None:
        now = time.monotonic()
        with self._lock:
            for pending in self._pending_remote.values():
                if pending["status"] != "leased" or pending["lease_at"] is None:
                    continue
                if now - pending["lease_at"] < self.config.worker_job_timeout:
                    continue
                pending["status"] = "queued"
                pending["worker_id"] = None
                pending["lease_at"] = None
                self._remote_jobs.put(pending["job"])
                self._log("warning", f"Requeued expired scale job {pending['job']['job_id']}")

    def _play_remote_iteration(self) -> list[Any]:
        self._queue_remote_iteration()
        completed: list[Any] = []
        last_wait_log = 0.0
        while len(completed) < self.config.games_per_iteration and not self._stop.is_set():
            try:
                job_id, result = self._remote_results.get(timeout=0.5)
            except queue.Empty:
                self._requeue_expired_jobs()
                if not self._active_workers() and time.monotonic() - last_wait_log > 15:
                    self.phase = "waiting_for_workers"
                    self._log("warning", "Waiting for distributed self-play workers")
                    last_wait_log = time.monotonic()
                continue
            with self._lock:
                self._pending_remote.pop(job_id, None)
            self.phase = "self_play"
            completed.append(result)
        return completed

    def _active_workers(self) -> list[dict[str, Any]]:
        cutoff = time.monotonic() - 30
        with self._lock:
            return [worker for worker in self._workers.values() if worker["seen_at"] >= cutoff]

    def claim_worker_job(self, worker_id: str) -> dict[str, Any] | None:
        self._touch_worker(worker_id, status="idle")
        if self.config.self_play_mode != "distributed" or self.status != "running" or self._pause.is_set():
            return None
        while True:
            try:
                job = self._remote_jobs.get_nowait()
            except queue.Empty:
                return None
            with self._lock:
                pending = self._pending_remote.get(job["job_id"])
                if not pending or pending["status"] != "queued":
                    continue
                pending["status"] = "leased"
                pending["worker_id"] = worker_id
                pending["lease_at"] = time.monotonic()
            self._touch_worker(worker_id, status="busy", revision=job["revision"], job_id=job["job_id"])
            self.phase = "self_play"
            return job

    def submit_worker_result(
        self,
        worker_id: str,
        job_id: str,
        revision: int,
        record: dict[str, Any],
        samples_blob: str,
    ) -> bool:
        with self._lock:
            pending = self._pending_remote.get(job_id)
            if not pending or pending["status"] == "complete":
                return False
            job = pending["job"]
            if revision != job["revision"]:
                raise ValueError("Worker result uses a stale model revision")
            record = {
                **record,
                "iteration": job["iteration"],
                "model_version": job["model_version"],
            }
            result = decode_result(record, samples_blob, job["iteration"])
            pending["status"] = "complete"
            self._remote_results.put((job_id, result))
        worker = self._touch_worker(worker_id, status="idle", revision=revision, job_id=None)
        with self._lock:
            worker["games_completed"] += 1
        return True

    def ingest_worker_event(
        self,
        worker_id: str,
        job_id: str,
        revision: int,
        event: dict[str, Any],
    ) -> bool:
        with self._lock:
            pending = self._pending_remote.get(job_id)
            if not pending or pending["job"]["revision"] != revision:
                return False
            job = pending["job"]
        event = {**event, "iteration": job["iteration"], "model_version": job["model_version"]}
        self._touch_worker(worker_id, status="busy", revision=revision, job_id=job_id)
        self._game_event(event)
        return True

    def _touch_worker(
        self,
        worker_id: str,
        status: str,
        revision: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            worker = self._workers.setdefault(
                worker_id,
                {
                    "id": worker_id,
                    "games_completed": 0,
                    "revision": None,
                    "job_id": None,
                },
            )
            worker.update(
                {
                    "status": status,
                    "revision": revision if revision is not None else worker["revision"],
                    "job_id": job_id,
                    "last_seen": now,
                    "seen_at": time.monotonic(),
                }
            )
            return worker

    def worker_model_path(self, revision: int) -> Path:
        if revision != self._worker_model_revision or not self._worker_model_path.exists():
            raise ValueError("Requested worker model revision is no longer available")
        return self._worker_model_path

    def _optimize(self) -> dict[str, float] | None:
        batch = self.replay.sample(self.config.batch_size)
        if not batch:
            return None
        states = torch.from_numpy(np.stack([sample.state for sample in batch])).to(self.device)
        policies = torch.from_numpy(np.stack([sample.policy for sample in batch])).to(self.device)
        values = torch.tensor([sample.value for sample in batch], dtype=torch.float32, device=self.device)
        self.model.train()
        logits, predicted_values = self.model(states)
        policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_values, values)
        total_loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.step += 1
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }

    def _autosave_candidate(self) -> None:
        save_training_state(
            self.checkpoint_dir,
            self.model,
            self.optimizer,
            self.model_version,
            self.iteration,
            self.games_completed,
            self.step,
            self._cycle_start_iteration,
            self.config.to_dict(),
        )

    def _promote_candidate(self) -> None:
        """Promote only candidates that beat the current champion."""
        next_version = self.model_version + 1
        if self.model_version == 0:
            path = save_checkpoint(
                self.checkpoint_dir,
                self.repository,
                self.model,
                self.optimizer,
                next_version,
                self.iteration,
                self.games_completed,
                self.step,
                self.config.to_dict(),
                {"status": "bootstrap", "score": 1.0},
            )
            self.model_version = next_version
            self._cycle_start_iteration = self.iteration
            self.last_promotion = {"status": "promoted", "score": 1.0, "version": next_version}
            self._log("info", f"Promoted bootstrap model v{next_version} ({path.name})")
            self._worker_model_revision = -1
            self._request_stop_after_promotion(next_version)
            return

        champion_metadata = self.repository.get_checkpoint(self.model_version)
        if not champion_metadata or not Path(champion_metadata["path"]).exists():
            raise RuntimeError(f"Champion checkpoint v{self.model_version} is unavailable")

        self.phase = "evaluating"
        candidate = copy.deepcopy(self.model).to(self.device)
        champion, _ = load_checkpoint(champion_metadata["path"], self.device)
        evaluation = evaluate_checkpoint(
            candidate,
            next_version,
            self.device,
            self.repository,
            "checkpoint",
            self.config.promotion_games,
            simulations=self.config.evaluation_simulations,
            max_plies=self.config.max_game_moves * 2,
            opponent_model=champion,
            opponent_version_label=f"v{self.model_version}",
            event_callback=lambda event: self.broker.publish("evaluation", event),
            seed=self.config.random_seed + self.iteration,
            opening_plies=self.config.evaluation_opening_plies,
            persist=False,
        )
        score = float(evaluation["score"])
        summary = {
            "score": score,
            "threshold": self.config.promotion_threshold,
            "wins": evaluation["wins"],
            "draws": evaluation["draws"],
            "losses": evaluation["losses"],
            "games": evaluation["games"],
        }
        evaluation.setdefault("id", uuid.uuid4().hex[:12])
        evaluation.setdefault("created_at", datetime.now(UTC).isoformat())
        if score >= self.config.promotion_threshold:
            path = save_checkpoint(
                self.checkpoint_dir,
                self.repository,
                self.model,
                self.optimizer,
                next_version,
                self.iteration,
                self.games_completed,
                self.step,
                self.config.to_dict(),
                {"status": "promoted", **summary},
            )
            self.model_version = next_version
            self.last_promotion = {"status": "promoted", "version": next_version, **summary}
            self._log(
                "info",
                f"Promoted model v{next_version}: score {score:.1%} over champion ({path.name})",
            )
            self._request_stop_after_promotion(next_version)
        else:
            self.model, payload = load_checkpoint(champion_metadata["path"], self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            if "optimizer_state" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            self.last_promotion = {"status": "rejected", "version": self.model_version, **summary}
            self._log(
                "warning",
                f"Rejected candidate: score {score:.1%}, requires {self.config.promotion_threshold:.1%}; restored v{self.model_version} and retained replay data",
            )
        evaluation["model_a"] = next_version
        self.repository.save_evaluation(evaluation)
        self._cycle_start_iteration = self.iteration
        self._worker_model_revision = -1
        self.phase = "self_play"

    def _request_stop_after_promotion(self, version: int) -> None:
        if not self.config.stop_after_promotion:
            return
        self._stop.set()
        self._pause.clear()
        self.status = "stopping"
        self._log("info", f"Model v{version} saved; stopping training as configured")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.iteration += 1
                iteration_start = time.monotonic()
                self.phase = "self_play"
                self._log("info", f"Iteration {self.iteration}: generating self-play data")
                self.model.eval()
                if self.config.self_play_mode == "distributed":
                    results = self._play_remote_iteration()
                else:
                    workers = min(self.config.parallel_workers, self.config.games_per_iteration)
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="caro-selfplay") as pool:
                        futures = [
                            pool.submit(self._play_game, game_index)
                            for game_index in range(self.config.games_per_iteration)
                        ]
                        results = [future.result() for future in as_completed(futures)]
                for result in results:
                    self.repository.save_game(result.record)
                    self.repository.trim_games(self.config.history_retention)
                    self.replay.extend(result.samples)
                    self.games_completed += 1
                    self._game_event({"type": "game_complete", "game_id": result.record["id"], "game": result.record})
                    with self._lock:
                        self.active_games.pop(result.record["id"], None)
                    self._log("info", f"Saved game {result.record['id']} ({result.record['result']})")

                losses = []
                if not self._stop.is_set():
                    self.phase = "optimizing"
                    for _ in range(self.config.training_updates):
                        while self._pause.is_set() and not self._stop.is_set():
                            time.sleep(0.1)
                        metric = self._optimize()
                        if metric:
                            losses.append(metric)
                if losses:
                    elapsed = max(0.001, time.monotonic() - iteration_start)
                    averaged = {key: sum(row[key] for row in losses) / len(losses) for key in losses[0]}
                    saved = {
                        "iteration": self.iteration,
                        "step": self.step,
                        **averaged,
                        "games_completed": self.games_completed,
                        "throughput": len(results) * 60 / elapsed,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                    self.repository.save_metric(saved)
                    self.broker.publish("metrics", {"type": "metric", **saved})
                candidate_iterations = self.iteration - self._cycle_start_iteration
                if candidate_iterations >= self.config.checkpoint_interval and not self._stop.is_set():
                    self._promote_candidate()
                if self.iteration % self.config.autosave_interval == 0 and not self._stop.is_set():
                    self._autosave_candidate()
                self._persist_state()
        except Exception as exc:  # pragma: no cover - surfaced through API
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self._log("error", self.last_error)
            self._log("debug", traceback.format_exc())
        finally:
            if self.status != "error":
                try:
                    self._autosave_candidate()
                except Exception as exc:  # pragma: no cover - shutdown recovery path
                    self._log("warning", f"Could not autosave candidate: {exc}")
                self.status = "idle"
            self.phase = "idle"
            self.active_games.clear()
            self._persist_state()
            self._log("info", "Training worker stopped cleanly")

    def _persist_state(self) -> None:
        self.repository.set_state(
            "training",
            {
                "iteration": self.iteration,
                "step": self.step,
                "model_version": self.model_version,
                "cycle_start_iteration": self._cycle_start_iteration,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        summary = self.repository.game_summary()
        metrics = self.repository.list_metrics(limit=1)
        latest = metrics[-1] if metrics else {}
        total = max(1, int(summary["total"]))
        runtime = time.monotonic() - self.started_at if self.started_at else 0.0
        process = psutil.Process()
        workers = [
            {key: value for key, value in worker.items() if key != "seen_at"}
            for worker in self._active_workers()
        ]
        return {
            "status": self.status,
            "phase": self.phase,
            "iteration": self.iteration,
            "step": self.step,
            "model_version": self.model_version,
            "candidate_progress": max(0, self.iteration - self._cycle_start_iteration),
            "candidate_target": self.config.checkpoint_interval,
            "last_promotion": self.last_promotion,
            "games_completed": int(summary["total"]),
            "worker_count": len(workers),
            "workers": workers,
            "active_games": list(self.active_games.values()),
            "runtime_seconds": runtime,
            "games_per_minute": ((int(summary["total"]) - self._session_start_games) / runtime * 60) if runtime else 0,
            "average_game_length": float(summary["average_moves"]),
            "white_win_rate": float(summary["white_wins"]) / total,
            "black_win_rate": float(summary["black_wins"]) / total,
            "draw_rate": float(summary["draws"]) / total,
            "policy_loss": latest.get("policy_loss"),
            "value_loss": latest.get("value_loss"),
            "total_loss": latest.get("total_loss"),
            "replay_size": len(self.replay),
            "device": str(self.device),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": psutil.virtual_memory().percent,
                "process_mb": process.memory_info().rss / 1024 / 1024,
                "platform": platform.platform(),
            },
            "last_error": self.last_error,
            "config": self.config.to_dict(),
        }
