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
    mcts_simulations: int = 24
    max_game_moves: int = 160
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    replay_capacity: int = 20_000
    training_updates: int = 8
    checkpoint_interval: int = 1
    evaluation_frequency: int = 2
    temperature_moves: int = 18
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
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
    games_per_iteration=1,
    mcts_simulations=4,
    max_game_moves=32,
    batch_size=8,
    replay_capacity=1_000,
    training_updates=2,
    evaluation_frequency=5,
    channels=24,
    residual_blocks=1,
)
