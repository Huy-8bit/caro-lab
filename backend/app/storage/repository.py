from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
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

SCHEMA_VERSION = 2

# Columns added after the first release. Existing databases are backed up
# (timestamped copy next to the database) before these are applied.
MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "metrics": [
        ("selfplay_seconds", "REAL"),
        ("optimize_seconds", "REAL"),
        ("iteration_seconds", "REAL"),
        ("selfplay_games_per_minute", "REAL"),
        ("replay_draw_share", "REAL"),
    ],
    "evaluations": [("summary_json", "TEXT NOT NULL DEFAULT '{}'")],
}

EVALUATION_COLUMNS = {"id", "model_a", "model_b", "games", "wins", "draws", "losses", "created_at", "details"}


class Repository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.backup_path: Path | None = None
        existed = self.path.exists() and self.path.stat().st_size > 0
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA journal_mode=WAL")
        self._migrate(existed)
        with self.connection() as connection:
            connection.execute("PRAGMA optimize")

    def _missing_columns(self) -> dict[str, list[tuple[str, str]]]:
        missing: dict[str, list[tuple[str, str]]] = {}
        with self.connection() as db:
            for table, columns in MIGRATIONS.items():
                present = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                needed = [column for column in columns if column[0] not in present]
                if needed:
                    missing[table] = needed
        return missing

    def backup(self, suffix: str = "pre-migration") -> Path:
        """Write a consistent, timestamped copy of the database (WAL-safe)."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self.path.with_name(f"{self.path.stem}.{suffix}-{stamp}{self.path.suffix}")
        with self._lock:
            source = sqlite3.connect(self.path, timeout=30)
            destination = sqlite3.connect(target)
            try:
                source.backup(destination)
                # Self-contained single file, no -wal/-shm side files.
                destination.execute("PRAGMA journal_mode=DELETE")
            finally:
                destination.close()
                source.close()
        return target

    def _migrate(self, existed: bool) -> None:
        missing = self._missing_columns()
        if not missing:
            return
        if existed:
            self.backup_path = self.backup()
        with self.connection() as db:
            for table, columns in missing.items():
                for name, definition in columns:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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
        values = {
            "selfplay_seconds": None,
            "optimize_seconds": None,
            "iteration_seconds": None,
            "selfplay_games_per_minute": None,
            "replay_draw_share": None,
            **metric,
        }
        with self.connection() as db:
            db.execute(
                """INSERT INTO metrics
                (iteration,step,policy_loss,value_loss,total_loss,games_completed,throughput,created_at,
                selfplay_seconds,optimize_seconds,iteration_seconds,selfplay_games_per_minute,replay_draw_share)
                VALUES (:iteration,:step,:policy_loss,:value_loss,:total_loss,:games_completed,:throughput,:created_at,
                :selfplay_seconds,:optimize_seconds,:iteration_seconds,:selfplay_games_per_minute,:replay_draw_share)""",
                values,
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
            item["evaluation"] = normalize_promotion(item["evaluation"])
            items.append(item)
        return items

    def get_checkpoint(self, version: int) -> dict[str, Any] | None:
        return next((item for item in self.list_checkpoints() if item["version"] == version), None)

    def save_evaluation(self, evaluation: dict[str, Any]) -> None:
        summary = {key: value for key, value in evaluation.items() if key not in EVALUATION_COLUMNS}
        with self.connection() as db:
            db.execute(
                """INSERT OR REPLACE INTO evaluations
                (id,model_a,model_b,games,wins,draws,losses,created_at,details_json,summary_json)
                VALUES (:id,:model_a,:model_b,:games,:wins,:draws,:losses,:created_at,:details_json,:summary_json)""",
                {
                    **evaluation,
                    "model_b": str(evaluation["model_b"]),
                    "details_json": json.dumps(evaluation.get("details", [])),
                    "summary_json": json.dumps(summary),
                },
            )

    def list_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            summary = json.loads(item.pop("summary_json") or "{}")
            output.append({**summary, **item})
        return output

    def iteration_quality(self, limit: int = 300) -> list[dict[str, Any]]:
        """Per-iteration self-play outcome mix joined with optimizer metrics.

        Outcome counts cover games from the standard start position only, so
        draw-collapse signals are not masked by short curriculum endgames;
        those are counted separately as ``curriculum_*``.
        """
        standard = "COALESCE(json_extract(metadata_json, '$.start'), 'standard') = 'standard'"
        with self.connection() as db:
            games = db.execute(
                f"""SELECT iteration,
                SUM(CASE WHEN {standard} THEN 1 ELSE 0 END) AS games,
                SUM(CASE WHEN {standard} AND result='1/2-1/2' THEN 1 ELSE 0 END) AS draws,
                SUM(CASE WHEN {standard} AND termination LIKE '%repetition%' THEN 1 ELSE 0 END) AS repetitions,
                SUM(CASE WHEN {standard} AND termination='checkmate' THEN 1 ELSE 0 END) AS checkmates,
                SUM(CASE WHEN {standard} AND termination='max move safety limit' THEN 1 ELSE 0 END) AS adjudicated,
                AVG(CASE WHEN {standard} THEN move_count END) AS average_plies,
                SUM(CASE WHEN {standard} THEN 0 ELSE 1 END) AS curriculum_games,
                SUM(CASE WHEN NOT {standard} AND termination='checkmate' THEN 1 ELSE 0 END) AS curriculum_checkmates,
                MAX(model_version) AS model_version
                FROM games GROUP BY iteration ORDER BY iteration DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            metrics = db.execute(
                """SELECT * FROM metrics WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY iteration)
                ORDER BY iteration DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        by_iteration = {row["iteration"]: dict(row) for row in metrics}
        rows = []
        for row in reversed(games):
            item = dict(row)
            metric = by_iteration.get(item["iteration"], {})
            for key in (
                "step", "policy_loss", "value_loss", "total_loss", "throughput", "selfplay_seconds",
                "optimize_seconds", "iteration_seconds", "selfplay_games_per_minute", "replay_draw_share",
            ):
                item[key] = metric.get(key)
            rows.append(item)
        return rows

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


def normalize_promotion(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Present a bootstrap checkpoint honestly.

    Releases before the promotion gate rework stored the first model as
    ``{"status": "bootstrap", "score": 1.0}``. It never played a match, so the
    stored score is dropped on read (the row itself is left untouched).
    """
    if evaluation.get("status") != "bootstrap":
        return evaluation
    return {
        "status": "bootstrap",
        "opponent": None,
        "note": "First model of the run; registered without a promotion match.",
    }
