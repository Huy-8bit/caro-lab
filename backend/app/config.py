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
    # Seconds a leased job may go without a worker heartbeat before requeue.
    worker_job_timeout: int = 60
    mcts_simulations: int = 24
    max_game_moves: int = 160
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    replay_capacity: int = 20_000
    training_updates: int = 8
    checkpoint_interval: int = 10
    promotion_games: int = 40
    promotion_threshold: float = 0.55
    evaluation_simulations: int = 32
    evaluation_opening_plies: int = 4
    # Quality gate: a candidate is rejected when its promotion match is
    # dominated by repetitions or has too few decisive games, whatever its score.
    gate_max_repetition_rate: float = 0.5
    gate_min_decisive_rate: float = 0.1
    # A rejected candidate keeps training unless it scored below this, in which
    # case the trainer restarts from the champion weights.
    reject_reset_score: float = 0.35
    stop_after_promotion: bool = True
    autosave_interval: int = 2
    temperature_moves: int = 18
    temperature: float = 1.0
    temperature_final: float = 0.0
    selfplay_opening_plies: int = 4
    # Fraction of self-play games that start from a random reduced-material
    # position (app.chess.endgames) so the network sees short games with real
    # mates. Promotion matches always use the standard start position.
    endgame_curriculum_fraction: float = 0.0
    # Moves played with temperature in a curriculum game. The opening schedule
    # (temperature_moves) would sample nearly every move of a short endgame and
    # throw the win away; root Dirichlet noise still provides exploration.
    endgame_temperature_moves: int = 0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
    # "visits": classic AlphaZero visit-count targets (default).
    # "completed_q": experimental completed-Q improved policy (Gumbel
    # AlphaZero); it increased repetition in our 30-iteration smoke run.
    policy_target: str = "visits"
    completed_q_scale: float = 0.1
    # Search-only draw aversion for the player choosing the move. Training
    # value targets for draws remain exactly 0 (zero-sum).
    draw_contempt: float = 0.15
    # Replay sampling weights per outcome class (decisive games weigh 1.0).
    draw_sample_weight: float = 0.6
    repetition_sample_weight: float = 0.25
    adjudicated_sample_weight: float = 0.3
    # Games longer than this many recorded positions are down-weighted so a
    # 300-ply shuffle does not outweigh a 60-ply mating game by sample count.
    sample_length_reference: int = 80
    replay_max_draw_fraction: float = 0.5
    analytics_window: int = 5
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

    def selfplay_settings(self) -> dict[str, Any]:
        """Everything a stateless worker needs to play one self-play game."""
        return {
            "mcts_simulations": self.mcts_simulations,
            "max_game_moves": self.max_game_moves,
            "temperature_moves": self.temperature_moves,
            "temperature": self.temperature,
            "temperature_final": self.temperature_final,
            "selfplay_opening_plies": self.selfplay_opening_plies,
            "endgame_temperature_moves": self.endgame_temperature_moves,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "c_puct": self.c_puct,
            "draw_contempt": self.draw_contempt,
            "policy_target": self.policy_target,
            "completed_q_scale": self.completed_q_scale,
            "draw_sample_weight": self.draw_sample_weight,
            "repetition_sample_weight": self.repetition_sample_weight,
            "adjudicated_sample_weight": self.adjudicated_sample_weight,
            "sample_length_reference": self.sample_length_reference,
            "event_sample_rate": self.event_sample_rate,
        }

    def evaluation_settings(self) -> dict[str, Any]:
        return {
            "simulations": self.evaluation_simulations,
            "max_plies": self.max_game_moves * 2,
            "c_puct": self.c_puct,
            "draw_contempt": self.draw_contempt,
            "policy_target": self.policy_target,
            "completed_q_scale": self.completed_q_scale,
            "event_sample_rate": self.event_sample_rate,
        }


QUICK_CONFIG = TrainingConfig(
    experiment_name="quick-start",
    games_per_iteration=4,
    parallel_workers=2,
    self_play_mode="local",
    worker_sync_interval=5,
    mcts_simulations=32,
    max_game_moves=120,
    batch_size=32,
    learning_rate=0.0005,
    replay_capacity=20_000,
    training_updates=16,
    checkpoint_interval=10,
    promotion_games=40,
    promotion_threshold=0.55,
    evaluation_simulations=32,
    evaluation_opening_plies=4,
    stop_after_promotion=True,
    autosave_interval=2,
    temperature_moves=20,
    channels=24,
    residual_blocks=1,
)
