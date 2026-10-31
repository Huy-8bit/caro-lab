from __future__ import annotations

import copy
import platform
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

from app.chess.endgames import curriculum_start
from app.chess.mcts import MCTS
from app.chess.move_index import POLICY_SIZE
from app.config import TrainingConfig
from app.core.events import EventBroker
from app.evaluation.arena import candidate_is_white, evaluate_checkpoint, summarize_match
from app.evaluation.progress import EvaluationProgress
from app.storage.repository import Repository
from app.training.analytics import quality_report
from app.training.checkpoints import load_checkpoint, save_checkpoint, save_training_state
from app.training.distributed import JobBoard, Lease, decode_result, save_worker_model
from app.training.network import NetworkSpec, PolicyValueNetwork, choose_device
from app.training.replay import NON_DECISIVE, ReplayBuffer
from app.training.self_play import SampleWeights, play_self_play_game


EVALUATION_STATE_KEY = "evaluation_progress"


class TrainingEngine:
    """The single authoritative trainer.

    Only this object (inside the backend process) owns the optimizer, the
    replay buffer, the SQLite repository and checkpoint files. Self-play and
    evaluation games may run on stateless workers, which lease jobs from
    :class:`JobBoard` and return results over HTTP.
    """

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
        self.optimizer = self._new_optimizer()
        self.replay = self._new_replay()
        persisted = self.repository.get_state("training") or {}
        self.iteration = int(persisted.get("iteration", 0))
        self.step = int(persisted.get("step", 0))
        self.model_version = int(persisted.get("model_version", 0))
        self._cycle_start_iteration = int(persisted.get("cycle_start_iteration", self.iteration))
        self.last_promotion: dict[str, Any] | None = persisted.get("last_promotion")
        self.games_completed = int(self.repository.game_summary()["total"])
        self._session_start_games = self.games_completed
        self.status = "idle"
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.phase = "idle"
        self.active_games: dict[str, dict[str, Any]] = {}
        self.logs: deque[dict[str, str]] = deque(maxlen=300)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()
        self.jobs = JobBoard(lease_timeout=self.config.worker_job_timeout)
        self.jobs.on_requeue = self._job_requeued
        self._workers: dict[str, dict[str, Any]] = {}
        self._worker_cache = self.checkpoint_dir / "worker-cache"
        self._worker_models: dict[str, Path] = {}
        self._worker_model_revision = -1
        self._selfplay_model_key: str | None = None
        # Superseded self-play model, kept until the next publish so games that
        # started on it before the switch can still download it.
        self._retiring_model_key: str | None = None
        # Distributed self-play is one continuous job stream (see
        # _play_remote_iteration) rather than a batch per iteration.
        self._selfplay_stream: str | None = None
        self._selfplay_jobs_submitted = 0
        self._evaluation: EvaluationProgress | None = None
        self.evaluation_state: dict[str, Any] | None = self.repository.get_state(EVALUATION_STATE_KEY)
        if self.evaluation_state and self.evaluation_state.get("status") == "running":
            self.evaluation_state = {**self.evaluation_state, "status": "interrupted", "current_games": []}
            self.repository.set_state(EVALUATION_STATE_KEY, self.evaluation_state)
        if self.repository.backup_path:
            self._log("info", f"Database schema upgraded; backup written to {self.repository.backup_path.name}")
        self._restore_latest()

    # ------------------------------------------------------------------
    # Construction helpers and state restore
    # ------------------------------------------------------------------

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _new_replay(self) -> ReplayBuffer:
        return ReplayBuffer(self.config.replay_capacity, self.config.random_seed, self.config.replay_max_draw_fraction)

    def _load_weights(self, path: str | Path) -> dict[str, Any]:
        self.model, payload = load_checkpoint(path, self.device)
        self.optimizer = self._new_optimizer()
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        return payload

    def _restore_latest(self) -> None:
        checkpoints = self.repository.list_checkpoints()
        if checkpoints:
            latest = checkpoints[0]
            path = Path(latest["path"])
            if path.exists():
                try:
                    payload = self._load_weights(path)
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
            _, payload = load_checkpoint(candidate_path, self.device)
            if int(payload.get("champion_version", payload["version"])) != self.model_version:
                return
            if int(payload["iteration"]) < self.iteration:
                return
            self._load_weights(candidate_path)
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

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

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
                payload = self._load_weights(metadata["path"])
                self.model_version = resume_version
                self.iteration = int(payload["iteration"])
                self.step = int(payload.get("step", 0))
                self._cycle_start_iteration = self.iteration
                replaced_model = True
            elif self.model.spec.channels != self.config.channels or self.model.spec.residual_blocks != self.config.residual_blocks:
                if self.model_version > 0:
                    raise ValueError(
                        f"Configured network ({self.config.channels}ch × {self.config.residual_blocks} blocks) does not "
                        f"match champion v{self.model_version} ({self.model.spec.channels}ch × "
                        f"{self.model.spec.residual_blocks} blocks). Use a separate data/checkpoint directory for a new run."
                    )
                self.model = PolicyValueNetwork(NetworkSpec(self.config.channels, self.config.residual_blocks)).to(self.device)
                self.optimizer = self._new_optimizer()
                replaced_model = True
            if not replaced_model:
                self.model = self.model.to(self.device)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.learning_rate
                    group["weight_decay"] = self.config.weight_decay
            self.replay = self._new_replay()
            self.jobs.clear()
            self.jobs.lease_timeout = self.config.worker_job_timeout
            self._worker_model_revision = -1
            self._selfplay_stream = None
            self._stop.clear()
            self._pause.clear()
            self.last_error = None
            self.status = "running"
            self.phase = "self_play"
            self.started_at = time.monotonic()
            self._session_start_games = self.games_completed
            self._thread = threading.Thread(target=self._run, name="caro-training", daemon=True)
            self._thread.start()
            self._log("info", f"Training started on {self.device.type} ({self.config.self_play_mode} self-play)")
            return self.snapshot()

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
            self._thread.join(timeout=60)
        return self.snapshot()

    def _wait_if_paused(self) -> bool:
        return self._pause.is_set() and not self._stop.is_set()

    # ------------------------------------------------------------------
    # Live game events
    # ------------------------------------------------------------------

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
                del current["moves"][:-400]
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

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def _play_game(self, game_index: int):
        settings = self.config.selfplay_settings()
        seed = self.config.random_seed + self.iteration * 1_000_003 + game_index
        start = self._start_position(seed)
        search = MCTS(
            self.model,
            self.device,
            simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            draw_contempt=self.config.draw_contempt,
            seed=seed,
            policy_mode=self.config.policy_target,
            value_scale=self.config.completed_q_scale,
        )
        return play_self_play_game(
            search=search,
            iteration=self.iteration,
            model_version=self.model_version,
            max_moves=self.config.max_game_moves,
            temperature_moves=self.config.endgame_temperature_moves if start else self.config.temperature_moves,
            temperature=self.config.temperature,
            temperature_final=self.config.temperature_final,
            opening_plies=self.config.selfplay_opening_plies,
            opening_seed=seed,
            sample_weights=SampleWeights.from_settings(settings),
            **start,
            event_callback=self._game_event,
            pause_callback=self._wait_if_paused,
            abort_callback=self._stop.is_set,
        )

    def _start_position(self, seed: int) -> dict[str, str]:
        """Curriculum start for a self-play game, decided by the trainer."""
        start = curriculum_start(seed, self.config.endgame_curriculum_fraction)
        return {"start_fen": start["fen"], "start_family": start["family"]} if start else {}

    def _publish_worker_model(self) -> str:
        if (
            self._worker_model_revision < 0
            or self.iteration - self._worker_model_revision >= self.config.worker_sync_interval
            or self._selfplay_model_key not in self._worker_models
        ):
            revision = self.iteration
            key = f"selfplay-v{self.model_version}-r{revision}"
            path = self._worker_cache / f"{key}.pt"
            save_worker_model(path, self.model, revision, self.model_version)
            previous = self._selfplay_model_key
            with self._lock:
                self._worker_models[key] = path
            self._selfplay_model_key = key
            self._worker_model_revision = revision
            if previous and previous != key:
                if self._retiring_model_key and self._retiring_model_key != key:
                    self._retire_worker_model(self._retiring_model_key)
                self._retiring_model_key = previous
            self._log("info", f"Published candidate revision {revision} to scale workers")
        assert self._selfplay_model_key is not None
        return self._selfplay_model_key

    def _retire_worker_model(self, key: str) -> None:
        with self._lock:
            path = self._worker_models.pop(key, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def _play_local_iteration(self) -> list[Any]:
        workers = min(self.config.parallel_workers, self.config.games_per_iteration)
        results = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="caro-selfplay") as pool:
            futures = [pool.submit(self._play_game, index) for index in range(self.config.games_per_iteration)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    if not self._stop.is_set():
                        raise
                    self._log("info", f"Self-play game stopped early ({type(exc).__name__})")
        return results

    def _play_remote_iteration(self) -> list[Any]:
        """Collect ``games_per_iteration`` games from the self-play job stream.

        Game lengths vary several-fold, so a batch that waits for its last game
        leaves most workers idle at the end of every iteration. Instead the
        stream keeps one extra job per active worker in flight: an iteration takes
        the first ``games_per_iteration`` results, and games still being played
        are counted towards the next iteration instead of being cancelled.
        Queued jobs are switched to the latest model before each iteration, so
        only games already in progress were started on the previous one. Each
        game is recorded with the iteration and model version it was played
        with.
        """
        model_key = self._publish_worker_model()
        if self._selfplay_stream is None:
            self._selfplay_stream = f"selfplay-{uuid.uuid4().hex[:8]}"
            self.jobs.open_batch(self._selfplay_stream)
        stream = self._selfplay_stream
        current = {
            "model_key": model_key,
            "revision": self._worker_model_revision,
            "iteration": self.iteration,
            "model_version": self.model_version,
        }
        self.jobs.update_queued(stream, current)
        settings = self.config.selfplay_settings()
        lookahead = min(len(self._active_workers()), self.config.games_per_iteration)
        # Finished-but-unconsumed games count as supply too, so the backlog can
        # never grow beyond one iteration plus the look-ahead.
        supply = self.jobs.outstanding(stream) + self.jobs.pending_results(stream)
        missing = self.config.games_per_iteration + lookahead - supply
        for _ in range(max(0, missing)):
            self._selfplay_jobs_submitted += 1
            seed = self.config.random_seed + self.iteration * 1_000_003 + self._selfplay_jobs_submitted
            self.jobs.submit(
                stream,
                {"kind": "selfplay", **current, "seed": seed, "config": settings, **self._start_position(seed)},
            )
        return self._collect(stream, self.config.games_per_iteration)

    def _pause_selfplay_stream(self) -> None:
        """Let promotion jobs go first; self-play games in progress still finish."""
        if self._selfplay_stream is not None:
            self.jobs.cancel_queued(self._selfplay_stream)

    def _collect(self, batch: str, expected: int, on_result=None) -> list[Any]:
        completed: list[Any] = []
        last_wait_log = 0.0
        while len(completed) < expected and not self._stop.is_set():
            item = self.jobs.next_result(batch, timeout=0.5)
            if item is None:
                self.jobs.expire()
                if not self._active_workers() and time.monotonic() - last_wait_log > 15:
                    self._log("warning", "Waiting for distributed self-play workers")
                    last_wait_log = time.monotonic()
                continue
            _, result = item
            completed.append(result)
            if on_result:
                on_result(result)
        return completed

    def _job_requeued(self, lease: Lease, reason: str) -> None:
        self._log("warning", f"Requeued {lease.job.get('kind', 'job')} job {lease.job['job_id'][:8]} ({reason})")
        if lease.job.get("kind") == "evaluation" and self._evaluation is not None:
            self._evaluation.game_requeued(int(lease.job["game_index"]))

    # ------------------------------------------------------------------
    # Worker protocol
    # ------------------------------------------------------------------

    def _active_workers(self) -> list[dict[str, Any]]:
        cutoff = time.monotonic() - 30
        with self._lock:
            return [worker for worker in self._workers.values() if worker["seen_at"] >= cutoff]

    def claim_worker_job(self, worker_id: str) -> dict[str, Any] | None:
        self._touch_worker(worker_id, status="idle")
        if self.config.self_play_mode != "distributed" or self.status != "running" or self._pause.is_set():
            return None
        job = self.jobs.claim(worker_id)
        if job is None:
            return None
        self._touch_worker(worker_id, status="busy", revision=job.get("revision"), job_id=job["job_id"])
        if job["kind"] == "evaluation" and self._evaluation is not None:
            self._evaluation.game_started(
                int(job["game_index"]), "white" if candidate_is_white(int(job["game_index"])) else "black", worker_id
            )
        return job

    def heartbeat_worker_job(self, worker_id: str, job_id: str) -> bool:
        active = self.jobs.heartbeat(worker_id, job_id) and not self._stop.is_set()
        self._touch_worker(worker_id, status="busy" if active else "idle", job_id=job_id if active else None)
        return active

    def release_worker_job(self, worker_id: str, job_id: str) -> bool:
        released = self.jobs.release(worker_id, job_id)
        self._touch_worker(worker_id, status="idle", job_id=None)
        return released

    def submit_worker_result(self, worker_id: str, job_id: str, payload: dict[str, Any]) -> bool:
        lease = self.jobs.get(job_id)
        if lease is None:
            return False
        job = lease.job
        if job["kind"] == "selfplay":
            record = {**payload["record"], "iteration": job["iteration"], "model_version": job["model_version"]}
            result: Any = decode_result(record, payload["samples_blob"], job["iteration"])
        else:
            detail = payload["detail"]
            if int(detail.get("game_index", -1)) != int(job["game_index"]) or detail.get("result") not in {"win", "draw", "loss"}:
                raise ValueError("Evaluation result does not match its job")
            result = detail
        accepted = self.jobs.complete(worker_id, job_id, result)
        worker = self._touch_worker(worker_id, status="idle", job_id=None)
        if accepted:
            with self._lock:
                worker["games_completed"] += 1
        return accepted

    def ingest_worker_event(self, worker_id: str, job_id: str, event: dict[str, Any]) -> bool:
        lease = self.jobs.get(job_id)
        if lease is None or not self.jobs.heartbeat(worker_id, job_id):
            return False
        job = lease.job
        self._touch_worker(worker_id, status="busy", revision=job.get("revision"), job_id=job_id)
        if job["kind"] == "evaluation":
            if self._evaluation is not None:
                self._evaluation.game_move(int(job["game_index"]), int(event.get("plies", 0)), str(event.get("fen", "")))
            return True
        self._game_event({**event, "iteration": job["iteration"], "model_version": job["model_version"]})
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
                {"id": worker_id, "games_completed": 0, "revision": None, "job_id": None},
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

    def worker_model_path(self, key: str) -> Path:
        with self._lock:
            path = self._worker_models.get(key)
        if path is None or not path.exists():
            raise ValueError("Requested worker model is no longer available")
        return path

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _optimize(self) -> dict[str, float] | None:
        batch = self.replay.sample(self.config.batch_size)
        if not batch:
            return None
        states = torch.from_numpy(np.stack([sample.state for sample in batch]).astype(np.float32)).to(self.device)
        dense = np.zeros((len(batch), POLICY_SIZE), dtype=np.float32)
        for row, sample in enumerate(batch):
            dense[row, sample.policy_index] = sample.policy_prob
        policies = torch.from_numpy(dense).to(self.device)
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
        draw_share = sum(1 for sample in batch if sample.outcome in NON_DECISIVE) / len(batch)
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "replay_draw_share": draw_share,
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

    # ------------------------------------------------------------------
    # Promotion gate
    # ------------------------------------------------------------------

    def _evaluation_changed(self, snapshot: dict[str, Any], persist: bool) -> None:
        self.evaluation_state = snapshot
        if persist:
            self.repository.set_state(EVALUATION_STATE_KEY, snapshot)
        self.broker.publish("evaluation", {"type": "evaluation_progress", "progress": snapshot})

    def new_evaluation_progress(self, kind: str, candidate_label: str, opponent_label: str, games: int, **kwargs) -> EvaluationProgress:
        return EvaluationProgress(kind, candidate_label, opponent_label, games, on_change=self._evaluation_changed, **kwargs)

    def promotion_games(self) -> int:
        games = max(2, int(self.config.promotion_games))
        return games + games % 2

    def _bootstrap_champion(self) -> None:
        evaluation = {
            "status": "bootstrap",
            "opponent": None,
            "note": "First model of the run; registered without a promotion match.",
        }
        path = save_checkpoint(
            self.checkpoint_dir, self.repository, self.model, self.optimizer, 1, self.iteration,
            self.games_completed, self.step, self.config.to_dict(), evaluation,
        )
        self.model_version = 1
        self._cycle_start_iteration = self.iteration
        self.last_promotion = {"status": "bootstrap", "version": 1, "iteration": self.iteration}
        self._worker_model_revision = -1
        self._log("info", f"Registered bootstrap model v1 ({path.name}); it has not played a match. Training continues toward v2.")

    def _play_promotion_match(self, candidate: torch.nn.Module, champion: torch.nn.Module, progress: EvaluationProgress) -> list[dict[str, Any]] | None:
        games = progress.snapshot()["games_total"]
        seed = self.config.random_seed + self.iteration
        settings = self.config.evaluation_settings()
        if self.config.self_play_mode == "distributed":
            stamp = progress.id
            keys = {"candidate": f"eval-{stamp}-candidate", "opponent": f"eval-{stamp}-champion"}
            for role, key in keys.items():
                path = self._worker_cache / f"{key}.pt"
                save_worker_model(path, candidate if role == "candidate" else champion, self.iteration, self.model_version)
                with self._lock:
                    self._worker_models[key] = path
            batch = f"evaluation-{stamp}"
            self.jobs.open_batch(batch)
            try:
                for game_index in range(games):
                    self.jobs.submit(
                        batch,
                        {
                            "kind": "evaluation",
                            "evaluation_id": stamp,
                            "game_index": game_index,
                            "seed": seed,
                            "opening_plies": self.config.evaluation_opening_plies,
                            "model_keys": keys,
                            "config": settings,
                        },
                    )
                details = self._collect(batch, games, on_result=progress.game_finished)
            finally:
                self.jobs.close_batch(batch)
                for key in keys.values():
                    self._retire_worker_model(key)
            return details if len(details) == games else None
        try:
            evaluation = evaluate_checkpoint(
                candidate, self.model_version + 1, self.device, self.repository, "checkpoint", games,
                simulations=settings["simulations"], max_plies=settings["max_plies"], opponent_model=champion,
                opponent_version_label=f"v{self.model_version}", seed=seed,
                opening_plies=self.config.evaluation_opening_plies, persist=False, c_puct=settings["c_puct"],
                draw_contempt=settings["draw_contempt"], parallel=self.config.parallel_workers,
                policy_mode=settings["policy_target"], value_scale=settings["completed_q_scale"],
                progress=progress, abort_callback=self._stop.is_set, finish_progress=False,
            )
        except InterruptedError:
            return None
        return evaluation["details"]

    def gate_decision(self, summary: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons = []
        if summary["score"] < self.config.promotion_threshold:
            reasons.append(f"score {summary['score']:.1%} is below the required {self.config.promotion_threshold:.0%}")
        if summary["repetition_rate"] > self.config.gate_max_repetition_rate:
            reasons.append(
                f"repetition rate {summary['repetition_rate']:.1%} exceeds the {self.config.gate_max_repetition_rate:.0%} limit"
            )
        if summary["decisive_rate"] < self.config.gate_min_decisive_rate:
            reasons.append(
                f"decisive-game rate {summary['decisive_rate']:.1%} is below the {self.config.gate_min_decisive_rate:.0%} minimum"
            )
        return not reasons, reasons

    def _promote_candidate(self) -> None:
        """Promote only candidates that beat the current champion cleanly."""
        if self.model_version == 0:
            self._bootstrap_champion()
            return

        champion_version = self.model_version
        next_version = champion_version + 1
        champion_metadata = self.repository.get_checkpoint(champion_version)
        if not champion_metadata or not Path(champion_metadata["path"]).exists():
            raise RuntimeError(f"Champion checkpoint v{champion_version} is unavailable")

        self.phase = "evaluating"
        started = time.monotonic()
        candidate = copy.deepcopy(self.model).to(self.device).eval()
        champion, _ = load_checkpoint(champion_metadata["path"], self.device)
        champion.eval()
        progress = self.new_evaluation_progress(
            "promotion", f"Candidate v{next_version}", f"Champion v{champion_version}", self.promotion_games(),
            candidate_version=next_version, opponent_version=champion_version,
        )
        self._evaluation = progress
        self._log("info", f"Promotion match started: Candidate v{next_version} vs Champion v{champion_version} ({self.promotion_games()} games)")
        try:
            self._pause_selfplay_stream()
            details = self._play_promotion_match(candidate, champion, progress)
        finally:
            self._evaluation = None
        if details is None:
            progress.finish("interrupted", {"status": "interrupted", "reasons": ["training stopped during the match"]})
            self._log("warning", f"Promotion match interrupted; champion v{champion_version} unchanged")
            self.phase = "self_play"
            return

        details.sort(key=lambda item: item["game_index"])
        summary = summarize_match(details)
        promoted, reasons = self.gate_decision(summary)
        gate = {
            "threshold": self.config.promotion_threshold,
            "max_repetition_rate": self.config.gate_max_repetition_rate,
            "min_decisive_rate": self.config.gate_min_decisive_rate,
        }
        public_summary = {key: value for key, value in summary.items()}
        decision = {
            "status": "promoted" if promoted else "rejected",
            "candidate_version": next_version,
            "champion_version": champion_version,
            "reasons": reasons,
            "gate": gate,
            "iteration": self.iteration,
            "duration_seconds": round(time.monotonic() - started, 1),
            **public_summary,
        }
        evaluation = {
            "id": progress.id,
            "model_a": next_version,
            "model_b": f"v{champion_version}",
            "created_at": datetime.now(UTC).isoformat(),
            "details": details,
            "kind": "promotion",
            "title": f"Candidate v{next_version} vs Champion v{champion_version}",
            **decision,
        }
        if promoted:
            path = save_checkpoint(
                self.checkpoint_dir, self.repository, self.model, self.optimizer, next_version, self.iteration,
                self.games_completed, self.step, self.config.to_dict(),
                {"status": "promoted", "opponent": f"v{champion_version}", "evaluation_id": progress.id, **public_summary, **gate},
            )
            self.model_version = next_version
            self.last_promotion = {"status": "promoted", "version": next_version, **decision}
            self._log("info", f"Promoted model v{next_version}: score {summary['score']:.1%} vs champion v{champion_version} ({path.name})")
        else:
            self.last_promotion = {"status": "rejected", "version": champion_version, **decision}
            if summary["score"] < self.config.reject_reset_score:
                self._load_weights(champion_metadata["path"])
                action = f"score below {self.config.reject_reset_score:.0%}, trainer restarted from champion weights"
            else:
                action = "trainer keeps its weights and continues learning"
            decision["action"] = action
            evaluation["action"] = action
            self._log("warning", f"Rejected Candidate v{next_version}: {'; '.join(reasons)}. Champion v{champion_version} unchanged; {action}.")
        self.repository.save_evaluation(evaluation)
        progress.finish(decision["status"], decision)
        self._cycle_start_iteration = self.iteration
        self._worker_model_revision = -1
        self._persist_state()
        self.phase = "self_play"
        if promoted:
            self._request_stop_after_promotion(next_version)

    def _request_stop_after_promotion(self, version: int) -> None:
        if not self.config.stop_after_promotion:
            return
        self._stop.set()
        self._pause.clear()
        self.status = "stopping"
        self._log("info", f"Model v{version} saved; stopping training as configured")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

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
                    results = self._play_local_iteration()
                selfplay_seconds = time.monotonic() - iteration_start
                for result in results:
                    self.repository.save_game(result.record)
                    self.replay.extend(result.samples)
                    self.games_completed += 1
                    self._game_event({"type": "game_complete", "game_id": result.record["id"], "game": result.record})
                    with self._lock:
                        self.active_games.pop(result.record["id"], None)
                    self._log("info", f"Saved game {result.record['id']} ({result.record['result']}, {result.record['termination']})")
                self.repository.trim_games(self.config.history_retention)

                losses = []
                optimize_start = time.monotonic()
                if not self._stop.is_set() and results:
                    self.phase = "optimizing"
                    for _ in range(self.config.training_updates):
                        while self._pause.is_set() and not self._stop.is_set():
                            time.sleep(0.1)
                        metric = self._optimize()
                        if metric:
                            losses.append(metric)
                optimize_seconds = time.monotonic() - optimize_start
                if losses:
                    elapsed = max(0.001, time.monotonic() - iteration_start)
                    averaged = {key: sum(row[key] for row in losses) / len(losses) for key in losses[0]}
                    saved = {
                        "iteration": self.iteration,
                        "step": self.step,
                        **averaged,
                        "games_completed": self.games_completed,
                        "throughput": len(results) * 60 / elapsed,
                        "selfplay_games_per_minute": len(results) * 60 / max(0.001, selfplay_seconds),
                        "selfplay_seconds": round(selfplay_seconds, 3),
                        "optimize_seconds": round(optimize_seconds, 3),
                        "iteration_seconds": round(elapsed, 3),
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
            self.jobs.clear()
            self._selfplay_stream = None
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
                "last_promotion": self.last_promotion,
            },
        )

    # ------------------------------------------------------------------
    # Read models
    # ------------------------------------------------------------------

    def analytics(self, limit: int = 300, window: int | None = None) -> dict[str, Any]:
        rows = self.repository.iteration_quality(limit)
        return quality_report(rows, window or self.config.analytics_window)

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
        quality = self.analytics(limit=max(3 * self.config.analytics_window, 30))
        return {
            "status": self.status,
            "phase": self.phase,
            "iteration": self.iteration,
            "step": self.step,
            "model_version": self.model_version,
            "candidate_version": self.model_version + 1,
            "candidate_progress": max(0, self.iteration - self._cycle_start_iteration),
            "candidate_target": self.config.checkpoint_interval,
            "last_promotion": self.last_promotion,
            "evaluation": self.evaluation_state,
            "games_completed": int(summary["total"]),
            "worker_count": len(workers),
            "workers": workers,
            "jobs": self.jobs.counts(),
            "active_games": list(self.active_games.values()),
            "runtime_seconds": runtime,
            # Session wall clock: includes self-play, optimizer and evaluation time.
            "games_per_minute": ((int(summary["total"]) - self._session_start_games) / runtime * 60) if runtime else 0,
            # Last iteration, self-play phase only (excludes optimizer and evaluation).
            "selfplay_games_per_minute": latest.get("selfplay_games_per_minute"),
            "selfplay_seconds": latest.get("selfplay_seconds"),
            "optimize_seconds": latest.get("optimize_seconds"),
            "average_game_length": float(summary["average_moves"]),
            "white_win_rate": float(summary["white_wins"]) / total,
            "black_win_rate": float(summary["black_wins"]) / total,
            "draw_rate": float(summary["draws"]) / total,
            "quality": {"rolling": quality["rolling"], "alerts": quality["alerts"], "window": quality["window"]},
            "policy_loss": latest.get("policy_loss"),
            "value_loss": latest.get("value_loss"),
            "total_loss": latest.get("total_loss"),
            "replay_size": len(self.replay),
            "replay": self.replay.composition(),
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
