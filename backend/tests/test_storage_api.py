from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn
from fastapi.testclient import TestClient

from app.main import app
from app.storage.repository import Repository


def game_record() -> dict:
    return {
        "id": "game-one", "iteration": 1, "model_version": 1,
        "white_agent": "Model v1", "black_agent": "Model v1", "result": "1-0",
        "winner": "white", "move_count": 4, "duration_seconds": 1.2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "pgn": "[Result \"1-0\"]\n\n1. e4 e5 2. Bc4 Nc6 1-0",
        "final_fen": chess.Board().fen(), "termination": "checkmate",
        "moves": [{"ply": 1, "uci": "e2e4", "san": "e4", "fen_before": chess.Board().fen(), "fen": chess.Board().fen(), "root_value": 0, "visits": 1}],
        "metadata": {"source": "test"},
    }


def test_game_persistence_and_pgn_replay(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "games.db")
    repository.save_game(game_record())
    saved = repository.get_game("game-one")
    assert saved and saved["moves"][0]["san"] == "e4"
    parsed = chess.pgn.read_game(__import__("io").StringIO(saved["pgn"]))
    board = parsed.board()
    for move in parsed.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
    assert board.fullmove_number == 3


def test_http_and_websocket_contracts() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        status = client.get("/api/status")
        assert status.status_code == 200
        assert "model_version" in status.json()
        with client.websocket_connect("/ws/live") as websocket:
            assert websocket.receive_json() == {"type": "connected", "channel": "live"}
            app.state.broker.publish("live", {"type": "move", "game_id": "live-one", "uci": "e2e4"})
            event = websocket.receive_json()
            assert event["type"] == "move"
            assert event["uci"] == "e2e4"


LEGACY_SCHEMA = """
CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, iteration INTEGER NOT NULL, step INTEGER NOT NULL,
  policy_loss REAL NOT NULL, value_loss REAL NOT NULL, total_loss REAL NOT NULL, games_completed INTEGER NOT NULL,
  throughput REAL NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE evaluations (id TEXT PRIMARY KEY, model_a INTEGER NOT NULL, model_b TEXT NOT NULL, games INTEGER NOT NULL,
  wins INTEGER NOT NULL, draws INTEGER NOT NULL, losses INTEGER NOT NULL, created_at TEXT NOT NULL, details_json TEXT NOT NULL);
INSERT INTO metrics VALUES (1, 1, 32, 7.0, 0.4, 7.4, 10, 2.0, '2026-01-01');
INSERT INTO evaluations VALUES ('old', 2, 'v1', 2, 0, 2, 0, '2026-01-01', '[]');
"""


def test_schema_upgrade_backs_up_and_keeps_existing_rows(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "caro.db"
    with sqlite3.connect(path) as db:
        db.executescript(LEGACY_SCHEMA)
    repository = Repository(path)
    assert repository.backup_path is not None and repository.backup_path.exists()
    assert repository.backup_path.name.startswith("caro.pre-migration-")
    with sqlite3.connect(repository.backup_path) as backup:
        assert backup.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 1
    assert repository.list_metrics()[0]["total_loss"] == 7.4
    assert repository.list_metrics()[0]["selfplay_seconds"] is None
    assert repository.list_evaluations()[0]["id"] == "old"
    # Second start: nothing to migrate, no extra backup.
    assert Repository(path).backup_path is None
    assert len(list(tmp_path.glob("caro.pre-migration-*"))) == 1


def test_fresh_database_needs_no_backup(tmp_path: Path) -> None:
    assert Repository(tmp_path / "new.db").backup_path is None


def test_quality_alert_when_draws_rise_while_loss_falls() -> None:
    from app.training.analytics import quality_report

    rows = []
    for iteration in range(1, 16):
        draws = min(10, 2 + iteration // 2)
        rows.append({
            "iteration": iteration, "games": 10, "draws": draws, "repetitions": max(0, draws - 3),
            "checkmates": 10 - draws, "adjudicated": 0, "total_loss": 8.0 - iteration * 0.3,
        })
    report = quality_report(rows, window=5)
    codes = {alert["code"] for alert in report["alerts"]}
    assert "draw_rate_rising_while_loss_falls" in codes
    assert report["iterations"][-1]["rolling"]["games"] == 50
    healthy = [{**row, "draws": 3, "repetitions": 0, "checkmates": 7} for row in rows]
    assert quality_report(healthy, window=5)["alerts"] == []


def test_evaluation_progress_and_analytics_endpoints() -> None:
    with TestClient(app) as client:
        engine = app.state.engine
        progress = engine.new_evaluation_progress("promotion", "Candidate v2", "Champion v1", 4)
        progress.game_started(0, "white")
        progress.game_finished({"game_index": 0, "result": "win", "termination": "checkmate"})
        body = client.get("/api/evaluations/progress").json()["progress"]
        assert body["title"] == "Candidate v2 vs Champion v1"
        assert (body["games_completed"], body["games_total"], body["wins"]) == (1, 4, 1)
        assert body["eta_seconds"] is not None
        assert client.get("/api/status").json()["evaluation"]["id"] == body["id"]
        with client.websocket_connect("/ws/evaluations") as websocket:
            assert websocket.receive_json()["type"] == "connected"
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "evaluation_progress" and snapshot["progress"]["id"] == body["id"]
        analytics = client.get("/api/analytics/iterations?window=3").json()
        assert analytics["window"] == 3 and "alerts" in analytics
        progress.finish("interrupted")
