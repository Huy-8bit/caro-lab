"""Self-play quality signals that the training loss cannot show.

Training loss measures how well the network fits its own recent self-play
targets. When self-play collapses into draws the targets become easy to fit
(value ≈ 0, shuffling policies), so loss can fall while strength stagnates.
These rolling outcome rates and alerts make that visible.
"""

from __future__ import annotations

from typing import Any

RATE_FIELDS = ("draw_rate", "repetition_rate", "checkmate_rate", "decisive_rate", "adjudicated_rate")


def _rates(games: int, draws: int, repetitions: int, checkmates: int, adjudicated: int) -> dict[str, float | None]:
    if not games:
        return {name: None for name in RATE_FIELDS}
    return {
        "draw_rate": draws / games,
        "repetition_rate": repetitions / games,
        "checkmate_rate": checkmates / games,
        "decisive_rate": (games - draws) / games,
        "adjudicated_rate": adjudicated / games,
    }


def quality_report(rows: list[dict[str, Any]], window: int = 5) -> dict[str, Any]:
    window = max(1, int(window))
    iterations = []
    for index, row in enumerate(rows):
        recent = rows[max(0, index - window + 1): index + 1]
        totals = {
            key: sum(int(item.get(key) or 0) for item in recent)
            for key in ("games", "draws", "repetitions", "checkmates", "adjudicated", "curriculum_games", "curriculum_checkmates")
        }
        losses = [item["total_loss"] for item in recent if item.get("total_loss") is not None]
        iterations.append(
            {
                **row,
                **_rates(row["games"], row["draws"], row["repetitions"], row["checkmates"], row["adjudicated"]),
                "rolling": {
                    **_rates(totals["games"], totals["draws"], totals["repetitions"], totals["checkmates"], totals["adjudicated"]),
                    "games": totals["games"],
                    "total_loss": sum(losses) / len(losses) if losses else None,
                    "curriculum_games": totals["curriculum_games"],
                    "curriculum_checkmate_rate": (
                        totals["curriculum_checkmates"] / totals["curriculum_games"] if totals["curriculum_games"] else None
                    ),
                },
            }
        )
    rolling = iterations[-1]["rolling"] if iterations else {
        **_rates(0, 0, 0, 0, 0), "games": 0, "total_loss": None, "curriculum_games": 0, "curriculum_checkmate_rate": None,
    }
    return {"window": window, "iterations": iterations, "rolling": rolling, "alerts": quality_alerts(iterations, window)}


def quality_alerts(iterations: list[dict[str, Any]], window: int) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if not iterations:
        return alerts
    latest = iterations[-1]["rolling"]
    if latest["games"] >= 10:
        if latest["draw_rate"] is not None and latest["draw_rate"] >= 0.9:
            alerts.append({
                "level": "warning",
                "code": "draw_dominated",
                "message": f"Self-play is draw-dominated: {latest['draw_rate']:.0%} draws over the last {window} iterations.",
            })
        if latest["repetition_rate"] is not None and latest["repetition_rate"] >= 0.4:
            alerts.append({
                "level": "warning",
                "code": "repetition_dominated",
                "message": f"{latest['repetition_rate']:.0%} of recent self-play games end by repetition.",
            })
    # Trend: three checkpoints one window apart, all with data.
    if len(iterations) >= 2 * window + 1:
        points = [iterations[-1 - offset * window]["rolling"] for offset in (2, 1, 0)]
        losses = [point["total_loss"] for point in points]
        loss_falling = all(value is not None for value in losses) and losses[0] > losses[1] > losses[2]
        for field, label in (("draw_rate", "Draw rate"), ("repetition_rate", "Repetition rate")):
            values = [point[field] for point in points]
            if any(value is None for value in values):
                continue
            rising = values[1] - values[0] >= 0.05 and values[2] - values[1] >= 0.05
            if rising and loss_falling:
                alerts.append({
                    "level": "warning",
                    "code": f"{field}_rising_while_loss_falls",
                    "message": (
                        f"{label} rose {values[0]:.0%} → {values[1]:.0%} → {values[2]:.0%} while training loss fell "
                        f"{losses[0]:.2f} → {losses[2]:.2f}. Lower loss is not stronger play; check the promotion match."
                    ),
                })
    return alerts
