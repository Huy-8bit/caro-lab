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
