from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainingConfigRequest(BaseModel):
    experiment_name: str = "local-alpha"
    games_per_iteration: int = Field(2, ge=1, le=1_000)
    parallel_workers: int = Field(1, ge=1, le=32)
    self_play_mode: Literal["local", "distributed"] = "local"
    worker_sync_interval: int = Field(5, ge=1, le=1_000)
    worker_job_timeout: int = Field(300, ge=10, le=3_600)
    mcts_simulations: int = Field(24, ge=1, le=10_000)
    max_game_moves: int = Field(160, ge=4, le=1_000)
    batch_size: int = Field(32, ge=1, le=4_096)
    learning_rate: float = Field(0.001, gt=0, le=1)
    weight_decay: float = Field(0.0001, ge=0, le=1)
    replay_capacity: int = Field(20_000, ge=32)
    training_updates: int = Field(8, ge=1, le=10_000)
    checkpoint_interval: int = Field(10, ge=1, le=10_000)
    evaluation_frequency: int = Field(10, ge=1, le=10_000)
    promotion_games: int = Field(8, ge=2, le=200)
    promotion_threshold: float = Field(0.55, ge=0.5, le=1.0)
    evaluation_simulations: int = Field(32, ge=1, le=10_000)
    evaluation_opening_plies: int = Field(4, ge=0, le=20)
    stop_after_promotion: bool = True
    autosave_interval: int = Field(2, ge=1, le=10_000)
    temperature_moves: int = Field(18, ge=0)
    temperature: float = Field(1.0, ge=0)
    dirichlet_alpha: float = Field(0.3, gt=0)
    dirichlet_epsilon: float = Field(0.25, ge=0, le=1)
    c_puct: float = Field(1.5, gt=0)
    repetition_penalty: float = Field(0.01, ge=0, le=1)
    draw_penalty: float = Field(0.05, ge=0, le=1)
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


class WorkerClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)


class WorkerResultRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    record: dict[str, Any]
    samples_blob: str


class WorkerEventRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    event: dict[str, Any]


class ApiMessage(BaseModel):
    ok: bool = True
    data: dict[str, Any]
