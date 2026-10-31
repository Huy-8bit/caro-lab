"""Summarise a Caro Lab SQLite database by iteration buckets.

Read-only: the database is opened with ``mode=ro`` so this script can be
pointed at a production ``data/caro.db`` without any risk of modifying it.

    python scripts/analyze_games.py ../data/caro.db --bucket 10
    python scripts/analyze_games.py ../data/caro.db --last 500 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _game_stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    total = len(rows)
    terminations = Counter(row["termination"] for row in rows)
    draws = sum(1 for row in rows if row["result"] == "1/2-1/2")
    repetitions = sum(count for name, count in terminations.items() if "repetition" in name)
    mates = terminations.get("checkmate", 0)
    plies = [row["move_count"] for row in rows]
    return {
        "games": total,
        "draw_rate": draws / total if total else 0.0,
        "decisive_rate": (total - draws) / total if total else 0.0,
        "checkmate_rate": mates / total if total else 0.0,
        "repetition_rate": repetitions / total if total else 0.0,
        "white_wins": sum(1 for row in rows if row["result"] == "1-0"),
        "black_wins": sum(1 for row in rows if row["result"] == "0-1"),
        "avg_plies": sum(plies) / total if total else 0.0,
        "terminations": dict(terminations.most_common()),
    }


def analyze(path: Path, bucket: int = 10, last: int | None = None) -> dict[str, Any]:
    with _connect(path) as db:
        games = db.execute(
            "SELECT iteration, result, termination, move_count, created_at, "
            "COALESCE(json_extract(metadata_json, '$.start'), 'standard') AS start FROM games ORDER BY created_at"
        ).fetchall()
    # Outcome rates describe games from the standard start position; curriculum
    # endgames are short, mostly decisive games and are reported separately.
    curriculum = [row for row in games if row["start"] != "standard"]
    games = [row for row in games if row["start"] == "standard"]
    with _connect(path) as db:
        metrics = db.execute(
            "SELECT iteration, step, policy_loss, value_loss, total_loss FROM metrics ORDER BY step"
        ).fetchall()
        evaluations = [dict(row) for row in db.execute("SELECT * FROM evaluations ORDER BY created_at")]
        checkpoints = [
            {"version": row["version"], "iteration": row["iteration"], "evaluation": json.loads(row["evaluation_json"])}
            for row in db.execute("SELECT version, iteration, evaluation_json FROM checkpoints ORDER BY version")
        ]
    report: dict[str, Any] = {"database": str(path), "overall": _game_stats(games)}
    if curriculum:
        report["curriculum"] = _game_stats(curriculum)
    if last:
        report[f"last_{last}"] = _game_stats(games[-last:])
    buckets: dict[int, list[sqlite3.Row]] = {}
    for row in games:
        buckets.setdefault((max(1, row["iteration"]) - 1) // bucket, []).append(row)
    loss_buckets: dict[int, list[sqlite3.Row]] = {}
    for row in metrics:
        loss_buckets.setdefault((max(1, row["iteration"]) - 1) // bucket, []).append(row)
    report["buckets"] = []
    for key in sorted(buckets):
        losses = loss_buckets.get(key, [])
        entry = {
            "iterations": f"{key * bucket + 1}-{(key + 1) * bucket}",
            **_game_stats(buckets[key]),
        }
        if losses:
            for name in ("policy_loss", "value_loss", "total_loss"):
                entry[name] = sum(row[name] for row in losses) / len(losses)
        report["buckets"].append(entry)
    for item in evaluations:
        details = json.loads(item.pop("details_json"))
        item["terminations"] = dict(Counter(detail.get("termination") for detail in details))
    report["evaluations"] = evaluations
    report["checkpoints"] = checkpoints
    return report


def _print_table(report: dict[str, Any]) -> None:
    print(f"database: {report['database']}")
    for label in [key for key in report if key in {"overall", "curriculum"} or key.startswith("last_")]:
        stats = report[label]
        print(
            f"{label:>9}: {stats['games']} games · draw {stats['draw_rate']:.1%} · repetition {stats['repetition_rate']:.1%}"
            f" · checkmate {stats['checkmate_rate']:.1%} · decisive {stats['decisive_rate']:.1%} · {stats['avg_plies']:.0f} plies"
        )
        print(f"           terminations {stats['terminations']}")
    print()
    print(f"{'iterations':>11} {'games':>5} {'draw':>6} {'repet':>6} {'mate':>6} {'decis':>6} {'plies':>6} {'policy':>7} {'value':>7} {'total':>7}")
    for entry in report["buckets"]:
        print(
            f"{entry['iterations']:>11} {entry['games']:>5} {entry['draw_rate']:>6.1%} {entry['repetition_rate']:>6.1%}"
            f" {entry['checkmate_rate']:>6.1%} {entry['decisive_rate']:>6.1%} {entry['avg_plies']:>6.0f}"
            f" {entry.get('policy_loss', float('nan')):>7.3f} {entry.get('value_loss', float('nan')):>7.3f}"
            f" {entry.get('total_loss', float('nan')):>7.3f}"
        )
    if report["evaluations"]:
        print()
        for item in report["evaluations"]:
            print(
                f"eval v{item['model_a']} vs {item['model_b']}: {item['wins']}W {item['draws']}D {item['losses']}L"
                f" of {item['games']} · {item['terminations']}"
            )
    for item in report["checkpoints"]:
        print(f"checkpoint v{item['version']} @ iteration {item['iteration']}: {item['evaluation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", type=Path)
    parser.add_argument("--bucket", type=int, default=10, help="iterations per group")
    parser.add_argument("--last", type=int, default=None, help="also summarise the most recent N games")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze(args.database, args.bucket, args.last)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_table(report)


if __name__ == "__main__":
    main()
