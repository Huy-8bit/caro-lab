from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class TrainingConfig:
    experiment_name: str = "local-alpha"
    games_per_iteration: int = 2
    parallel_workers: int = 1
    self_play_mode: str = "local"
    worker_sync_interval: int = 5
    worker_job_timeout: int = 300
    mcts_simulations: int = 24
    max_game_moves: int = 160
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    replay_capacity: int = 20_000
    training_updates: int = 8
    checkpoint_interval: int = 10
    evaluation_frequency: int = 10
    promotion_games: int = 8
    promotion_threshold: float = 0.55
    evaluation_simulations: int = 32
    evaluation_opening_plies: int = 4
    stop_after_promotion: bool = True
    autosave_interval: int = 2
    temperature_moves: int = 18
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
    repetition_penalty: float = 0.01
    draw_penalty: float = 0.05
    random_seed: int = 7
    device: str = "auto"
    channels: int = 48
    residual_blocks: int = 2
    history_retention: int = 10_000
    event_sample_rate: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


QUICK_CONFIG = TrainingConfig(
    experiment_name="quick-start",
    games_per_iteration=4,
    parallel_workers=2,
    self_play_mode="local",
    worker_sync_interval=5,
    worker_job_timeout=300,
    mcts_simulations=32,
    max_game_moves=120,
    batch_size=32,
    learning_rate=0.0005,
    replay_capacity=20_000,
    training_updates=16,
    checkpoint_interval=10,
    evaluation_frequency=10,
    promotion_games=8,
    promotion_threshold=0.55,
    evaluation_simulations=32,
    evaluation_opening_plies=4,
    stop_after_promotion=True,
    autosave_interval=2,
    temperature_moves=20,
    repetition_penalty=0.01,
    draw_penalty=0.05,
    channels=24,
    residual_blocks=1,
)
