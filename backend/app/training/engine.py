from __future__ import annotations

import copy
import platform
import threading
import time
import traceback
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
from app.training.checkpoints import load_checkpoint, save_checkpoint
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
        self.games_completed = int(self.repository.game_summary()["total"])
        self._session_start_games = self.games_completed
        self.status = "idle"
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.active_games: dict[str, dict[str, Any]] = {}
        self.logs: deque[dict[str, str]] = deque(maxlen=300)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()
        self._restore_latest()

    def _restore_latest(self) -> None:
        checkpoints = self.repository.list_checkpoints()
        if not checkpoints:
            return
        latest = checkpoints[0]
        path = Path(latest["path"])
        if not path.exists():
            return
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
            self._log("info", f"Restored model v{self.model_version}")
        except Exception as exc:  # pragma: no cover - defensive startup path
            self._log("warning", f"Could not restore checkpoint: {exc}")

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
            if resume_version is not None:
                metadata = self.repository.get_checkpoint(resume_version)
                if not metadata:
                    raise ValueError(f"Checkpoint v{resume_version} does not exist")
                self.model, payload = load_checkpoint(metadata["path"], self.device)
                self.model_version = resume_version
                self.iteration = int(payload["iteration"])
                self.step = int(payload.get("step", 0))
            elif self.model.spec.channels != self.config.channels or self.model.spec.residual_blocks != self.config.residual_blocks:
                self.model = PolicyValueNetwork(NetworkSpec(self.config.channels, self.config.residual_blocks)).to(self.device)
                self.model_version = 0
                self.iteration = 0
                self.step = 0
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            self.replay = ReplayBuffer(self.config.replay_capacity, self.config.random_seed)
            self._stop.clear()
            self._pause.clear()
            self.last_error = None
            self.status = "running"
            self.started_at = time.monotonic()
            self._session_start_games = self.games_completed
            self._thread = threading.Thread(target=self._run, name="caro-training", daemon=True)
            self._thread.start()
            self._log("info", f"Training started on {self.device.type}")
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
            seed=self.config.random_seed + self.games_completed + game_index,
        )
        return play_self_play_game(
            search,
            self.iteration,
            self.model_version,
            self.config.max_game_moves,
            self.config.temperature_moves,
            self.config.temperature,
            self._game_event,
            self._wait_if_paused,
        )

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

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.iteration += 1
                iteration_start = time.monotonic()
                self._log("info", f"Iteration {self.iteration}: generating self-play data")
                workers = min(self.config.parallel_workers, self.config.games_per_iteration)
                self.model.eval()
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="caro-selfplay") as pool:
                    futures = [pool.submit(self._play_game, game_index) for game_index in range(self.config.games_per_iteration)]
                    for future in as_completed(futures):
                        result = future.result()
                        self.repository.save_game(result.record)
                        self.repository.trim_games(self.config.history_retention)
                        self.replay.extend(result.samples)
                        self.games_completed += 1
                        with self._lock:
                            self.active_games.pop(result.record["id"], None)
                        self._log("info", f"Saved game {result.record['id']} ({result.record['result']})")

                losses = []
                if not self._stop.is_set():
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
                        "throughput": self.config.games_per_iteration * 60 / elapsed,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                    self.repository.save_metric(saved)
                    self.broker.publish("metrics", {"type": "metric", **saved})
                if self.iteration % self.config.checkpoint_interval == 0 and not self._stop.is_set():
                    self.model_version += 1
                    path = save_checkpoint(
                        self.checkpoint_dir,
                        self.repository,
                        self.model,
                        self.optimizer,
                        self.model_version,
                        self.iteration,
                        self.games_completed,
                        self.step,
                        self.config.to_dict(),
                    )
                    self._log("info", f"Saved checkpoint {path.name}")
                    if self.iteration % self.config.evaluation_frequency == 0:
                        self._launch_automatic_evaluation()
                self._persist_state()
        except Exception as exc:  # pragma: no cover - surfaced through API
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self._log("error", self.last_error)
            self._log("debug", traceback.format_exc())
        finally:
            if self.status != "error":
                self.status = "idle"
            self.active_games.clear()
            self._persist_state()
            self._log("info", "Training worker stopped cleanly")

    def _persist_state(self) -> None:
        self.repository.set_state(
            "training",
            {"iteration": self.iteration, "step": self.step, "model_version": self.model_version},
        )

    def _launch_automatic_evaluation(self) -> None:
        candidate = copy.deepcopy(self.model).to(self.device)
        version = self.model_version
        previous = self.repository.get_checkpoint(version - 1) if version > 1 else None
        simulations = max(2, min(16, self.config.mcts_simulations // 4))

        def run() -> None:
            try:
                evaluate_checkpoint(
                    candidate, version, self.device, self.repository, "random", 4,
                    simulations=simulations, event_callback=lambda event: self.broker.publish("evaluation", event),
                    seed=self.config.random_seed + version,
                )
                if previous:
                    evaluate_checkpoint(
                        candidate, version, self.device, self.repository, "checkpoint", 4,
                        simulations=simulations, checkpoint_path=previous["path"],
                        event_callback=lambda event: self.broker.publish("evaluation", event),
                        seed=self.config.random_seed + version + 1,
                    )
                self._log("info", f"Automatic evaluation for model v{version} completed")
            except Exception as exc:  # pragma: no cover - background diagnostic path
                self._log("error", f"Automatic evaluation failed: {exc}")

        threading.Thread(target=run, name=f"caro-auto-eval-v{version}", daemon=True).start()

    def snapshot(self) -> dict[str, Any]:
        summary = self.repository.game_summary()
        metrics = self.repository.list_metrics(limit=1)
        latest = metrics[-1] if metrics else {}
        total = max(1, int(summary["total"]))
        runtime = time.monotonic() - self.started_at if self.started_at else 0.0
        process = psutil.Process()
        return {
            "status": self.status,
            "iteration": self.iteration,
            "step": self.step,
            "model_version": self.model_version,
            "games_completed": int(summary["total"]),
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
