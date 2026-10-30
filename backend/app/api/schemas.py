from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainingConfigRequest(BaseModel):
    experiment_name: str = "local-alpha"
    games_per_iteration: int = Field(2, ge=1, le=1_000)
    parallel_workers: int = Field(1, ge=1, le=32)
    mcts_simulations: int = Field(24, ge=1, le=10_000)
    max_game_moves: int = Field(160, ge=4, le=1_000)
    batch_size: int = Field(32, ge=1, le=4_096)
    learning_rate: float = Field(0.001, gt=0, le=1)
    weight_decay: float = Field(0.0001, ge=0, le=1)
    replay_capacity: int = Field(20_000, ge=32)
    training_updates: int = Field(8, ge=1, le=10_000)
    checkpoint_interval: int = Field(1, ge=1)
    evaluation_frequency: int = Field(2, ge=1)
    temperature_moves: int = Field(18, ge=0)
    temperature: float = Field(1.0, ge=0)
    dirichlet_alpha: float = Field(0.3, gt=0)
    dirichlet_epsilon: float = Field(0.25, ge=0, le=1)
    c_puct: float = Field(1.5, gt=0)
    random_seed: int = 7
    device: str = "auto"
    channels: int = Field(48, ge=8, le=512)
    residual_blocks: int = Field(2, ge=0, le=32)
    history_retention: int = Field(10_000, ge=1)
    event_sample_rate: int = Field(1, ge=1)


class StartTrainingRequest(BaseModel):
    config: TrainingConfigRequest | None = None
    resume_version: int | None = None


class EvaluationRequest(BaseModel):
    model_version: int | None = None
    opponent: Literal["random", "heuristic", "checkpoint"] = "random"
    opponent_version: int | None = None
    games: int = Field(4, ge=2, le=100)
    simulations: int = Field(8, ge=1, le=1_000)


class ApiMessage(BaseModel):
    ok: bool = True
    data: dict[str, Any]
