from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    model_version INTEGER NOT NULL,
    white_agent TEXT NOT NULL,
    black_agent TEXT NOT NULL,
    result TEXT NOT NULL,
    winner TEXT,
    move_count INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    created_at TEXT NOT NULL,
    pgn TEXT NOT NULL,
    final_fen TEXT NOT NULL,
    termination TEXT NOT NULL,
    moves_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    step INTEGER NOT NULL,
    policy_loss REAL NOT NULL,
    value_loss REAL NOT NULL,
    total_loss REAL NOT NULL,
    games_completed INTEGER NOT NULL,
    throughput REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    version INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    games INTEGER NOT NULL,
    architecture_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    file_size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    model_a INTEGER NOT NULL,
    model_b TEXT NOT NULL,
    games INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_games_iteration ON games(iteration);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics(step);
"""


class Repository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA optimize")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()

    def save_game(self, game: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT OR REPLACE INTO games VALUES
                (:id,:iteration,:model_version,:white_agent,:black_agent,:result,:winner,
                :move_count,:duration_seconds,:created_at,:pgn,:final_fen,:termination,
                :moves_json,:metadata_json)""",
                {
                    **game,
                    "moves_json": json.dumps(game.get("moves", [])),
                    "metadata_json": json.dumps(game.get("metadata", {})),
                },
            )

    def list_games(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM games ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._game_row(row) for row in rows]

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return self._game_row(row) if row else None

    @staticmethod
    def _game_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["moves"] = json.loads(result.pop("moves_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def trim_games(self, keep: int) -> None:
        with self.connection() as db:
            db.execute(
                "DELETE FROM games WHERE id NOT IN (SELECT id FROM games ORDER BY created_at DESC LIMIT ?)",
                (keep,),
            )

    def game_summary(self) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute(
                """SELECT COUNT(*) AS total,
                SUM(CASE WHEN result='1-0' THEN 1 ELSE 0 END) AS white_wins,
                SUM(CASE WHEN result='0-1' THEN 1 ELSE 0 END) AS black_wins,
                SUM(CASE WHEN result='1/2-1/2' THEN 1 ELSE 0 END) AS draws,
                AVG(move_count) AS average_moves, AVG(duration_seconds) AS average_duration
                FROM games"""
            ).fetchone()
        return {key: (row[key] or 0) for key in row.keys()}

    def save_metric(self, metric: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT INTO metrics
                (iteration,step,policy_loss,value_loss,total_loss,games_completed,throughput,created_at)
                VALUES (:iteration,:step,:policy_loss,:value_loss,:total_loss,:games_completed,:throughput,:created_at)""",
                metric,
            )

    def list_metrics(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM metrics ORDER BY step DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        values = {
            **checkpoint,
            "architecture_json": json.dumps(checkpoint["architecture"]),
            "config_json": json.dumps(checkpoint["config"]),
            "evaluation_json": json.dumps(checkpoint.get("evaluation", {})),
        }
        with self.connection() as db:
            db.execute(
                """INSERT OR REPLACE INTO checkpoints VALUES
                (:version,:path,:created_at,:iteration,:games,:architecture_json,:config_json,
                :evaluation_json,:file_size)""",
                values,
            )

    def list_checkpoints(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM checkpoints ORDER BY version DESC").fetchall()
        items = []
        for row in rows:
            item = dict(row)
            for source, target in (
                ("architecture_json", "architecture"),
                ("config_json", "config"),
                ("evaluation_json", "evaluation"),
            ):
                item[target] = json.loads(item.pop(source))
            items.append(item)
        return items

    def get_checkpoint(self, version: int) -> dict[str, Any] | None:
        return next((item for item in self.list_checkpoints() if item["version"] == version), None)

    def save_evaluation(self, evaluation: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT OR REPLACE INTO evaluations VALUES
                (:id,:model_a,:model_b,:games,:wins,:draws,:losses,:created_at,:details_json)""",
                {**evaluation, "details_json": json.dumps(evaluation.get("details", []))},
            )

    def list_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            output.append(item)
        return output

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO runtime_state(key,value_json) VALUES (?,?)",
                (key, json.dumps(value)),
            )

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT value_json FROM runtime_state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
