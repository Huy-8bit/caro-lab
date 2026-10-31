from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Callable


class EvaluationProgress:
    """Thread-safe, persistable state of one running evaluation match.

    ``on_change(snapshot, persist)`` is invoked after every update; ``persist``
    is True for durable milestones (start, game finished, finish) so the
    backend can store them while live move updates are only broadcast.
    """

    def __init__(
        self,
        kind: str,
        candidate_label: str,
        opponent_label: str,
        games_total: int,
        candidate_version: int | None = None,
        opponent_version: int | None = None,
        on_change: Callable[[dict[str, Any], bool], None] | None = None,
        evaluation_id: str | None = None,
    ) -> None:
        self._lock = RLock()
        self._on_change = on_change
        self._started = time.monotonic()
        self._state: dict[str, Any] = {
            "id": evaluation_id or uuid.uuid4().hex[:12],
            "kind": kind,
            "status": "running",
            "title": f"{candidate_label} vs {opponent_label}",
            "candidate_label": candidate_label,
            "opponent_label": opponent_label,
            "candidate_version": candidate_version,
            "opponent_version": opponent_version,
            "games_total": games_total,
            "games_completed": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "score": None,
            "terminations": {},
            "current_games": {},
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "decision": None,
        }
        self._emit(True)

    @property
    def id(self) -> str:
        return self._state["id"]

    def _emit(self, persist: bool) -> None:
        if self._on_change:
            self._on_change(self.snapshot(), persist)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = {**self._state}
            elapsed = time.monotonic() - self._started if state["status"] == "running" else state["elapsed_seconds"]
            state["elapsed_seconds"] = round(elapsed, 1)
            done = state["games_completed"]
            remaining = state["games_total"] - done
            if state["status"] == "running" and done:
                state["eta_seconds"] = round(elapsed / done * remaining, 1)
            elif state["status"] != "running":
                state["eta_seconds"] = 0.0
            state["current_games"] = sorted(state["current_games"].values(), key=lambda item: item["game_index"])
            state["terminations"] = dict(state["terminations"])
            return state

    def game_started(self, game_index: int, candidate_color: str, worker_id: str | None = None) -> None:
        with self._lock:
            self._state["current_games"][game_index] = {
                "game_index": game_index,
                "game": game_index + 1,
                "candidate_color": candidate_color,
                "plies": 0,
                "fen": None,
                "worker_id": worker_id,
            }
            self._touch()
        self._emit(False)

    def game_move(self, game_index: int, plies: int, fen: str) -> None:
        with self._lock:
            current = self._state["current_games"].get(game_index)
            if current is None:
                return
            current.update({"plies": plies, "fen": fen})
            self._touch()
        self._emit(False)

    def game_requeued(self, game_index: int) -> None:
        with self._lock:
            self._state["current_games"].pop(game_index, None)
            self._touch()
        self._emit(False)

    def game_finished(self, detail: dict[str, Any]) -> None:
        with self._lock:
            self._state["current_games"].pop(detail["game_index"], None)
            self._state["games_completed"] += 1
            key = {"win": "wins", "draw": "draws", "loss": "losses"}[detail["result"]]
            self._state[key] += 1
            terminations = Counter(self._state["terminations"])
            terminations[detail["termination"]] += 1
            self._state["terminations"] = dict(terminations)
            done = self._state["games_completed"]
            self._state["score"] = (self._state["wins"] + 0.5 * self._state["draws"]) / done
            self._touch()
        self._emit(True)

    def finish(self, status: str, decision: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._state["elapsed_seconds"] = round(time.monotonic() - self._started, 1)
            self._state["status"] = status
            self._state["decision"] = decision
            self._state["current_games"] = {}
            self._touch()
        self._emit(True)

    def _touch(self) -> None:
        self._state["updated_at"] = datetime.now(UTC).isoformat()
