from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from app.training.replay import DECISIVE, REPETITION, ReplayBuffer, TrainingSample
from app.training.self_play import SampleWeights


def sample(outcome: str, weight: float = 1.0, game: str = "g") -> TrainingSample:
    return TrainingSample(
        state=np.zeros((20, 8, 8), np.float32),
        policy_index=np.array([3], np.int32),
        policy_prob=np.array([1.0], np.float32),
        value=0.0 if outcome != DECISIVE else 1.0,
        game_id=game,
        ply=0,
        iteration=1,
        weight=weight,
        outcome=outcome,
    )


def test_draw_samples_cannot_dominate_sampling_mass() -> None:
    # 95% of stored samples come from repetition draws, as in the collapsed run.
    buffer = ReplayBuffer(10_000, seed=1, max_draw_fraction=0.5)
    buffer.extend([sample(REPETITION) for _ in range(950)] + [sample(DECISIVE) for _ in range(50)])
    composition = buffer.composition()
    assert composition["counts"][REPETITION] == 950
    assert composition["sampling_share"][REPETITION] == pytest.approx(0.5)
    drawn = Counter()
    for _ in range(200):
        drawn.update(item.outcome for item in buffer.sample(20))
    assert 0.4 < drawn[REPETITION] / sum(drawn.values()) < 0.6


def test_outcome_weights_apply_below_the_cap() -> None:
    buffer = ReplayBuffer(1_000, seed=2, max_draw_fraction=0.9)
    buffer.extend([sample(REPETITION, weight=0.25) for _ in range(100)] + [sample(DECISIVE) for _ in range(100)])
    share = buffer.composition()["sampling_share"]
    assert share[REPETITION] == pytest.approx(0.25 / 1.25)


def test_all_draw_buffer_still_trains() -> None:
    buffer = ReplayBuffer(100, seed=3, max_draw_fraction=0.5)
    buffer.extend([sample(REPETITION, weight=0.25) for _ in range(10)])
    assert len(buffer.sample(8)) == 8


def test_sampling_is_without_replacement_and_capacity_bounded() -> None:
    buffer = ReplayBuffer(5, seed=4)
    buffer.extend([sample(DECISIVE, game=str(index)) for index in range(9)])
    assert len(buffer) == 5
    batch = buffer.sample(10)
    assert len({id(item) for item in batch}) == len(batch) == 5


def test_sample_weights_favour_decisive_games() -> None:
    weights = SampleWeights()
    assert weights.for_outcome(DECISIVE) == 1.0
    assert weights.for_outcome(REPETITION) < weights.for_outcome("draw") < 1.0
