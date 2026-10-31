from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.chess.move_index import POLICY_SIZE

DECISIVE = "decisive"
DRAW = "draw"
REPETITION = "repetition"
ADJUDICATED = "adjudicated"
OUTCOME_CLASSES = (DECISIVE, DRAW, REPETITION, ADJUDICATED)
NON_DECISIVE = frozenset({DRAW, REPETITION, ADJUDICATED})


@dataclass(slots=True)
class TrainingSample:
    """One position with its MCTS policy target and outcome value target.

    The policy is stored sparsely (legal-move indices and visit fractions):
    a dense 20,480-float vector per position would cost ~80 KB, which made a
    100k-sample buffer need several gigabytes.
    """

    state: np.ndarray
    policy_index: np.ndarray
    policy_prob: np.ndarray
    value: float
    game_id: str
    ply: int
    iteration: int
    weight: float = 1.0
    outcome: str = DECISIVE

    @classmethod
    def from_dense(
        cls,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        game_id: str,
        ply: int,
        iteration: int,
        weight: float = 1.0,
        outcome: str = DECISIVE,
    ) -> "TrainingSample":
        index = np.flatnonzero(policy).astype(np.int32)
        return cls(state, index, policy[index].astype(np.float32), value, game_id, ply, iteration, weight, outcome)

    def dense_policy(self) -> np.ndarray:
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[self.policy_index] = self.policy_prob
        return policy


class ReplayBuffer:
    """Bounded FIFO buffer with outcome-aware prioritised sampling.

    Each sample carries a weight set at game time (outcome class and game
    length normalisation). On top of that, the total sampling mass of
    non-decisive samples is capped at ``max_draw_fraction`` whenever decisive
    samples exist, so a run of repetition draws cannot crowd decisive games
    out of the optimizer's batches.
    """

    def __init__(self, capacity: int, seed: int = 0, max_draw_fraction: float = 1.0) -> None:
        self._samples: deque[TrainingSample] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)
        self._lock = Lock()
        self.max_draw_fraction = max_draw_fraction
        self._probabilities: np.ndarray | None = None
        self._composition: dict[str, object] | None = None

    def extend(self, samples: list[TrainingSample]) -> None:
        with self._lock:
            self._samples.extend(samples)
            self._probabilities = None
            self._composition = None

    def _sampling_probabilities(self) -> np.ndarray:
        if self._probabilities is not None:
            return self._probabilities
        weights = np.fromiter((max(0.0, sample.weight) for sample in self._samples), dtype=np.float64)
        drawn = np.fromiter((sample.outcome in NON_DECISIVE for sample in self._samples), dtype=bool)
        draw_mass = weights[drawn].sum()
        decisive_mass = weights[~drawn].sum()
        cap = self.max_draw_fraction
        if 0 < cap < 1 and decisive_mass > 0 and draw_mass / (draw_mass + decisive_mass) > cap:
            weights[drawn] *= cap * decisive_mass / ((1 - cap) * draw_mass)
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(self._samples), dtype=np.float64)
            total = weights.sum()
        self._probabilities = weights / total
        return self._probabilities

    def sample(self, batch_size: int) -> list[TrainingSample]:
        with self._lock:
            if not self._samples:
                return []
            probabilities = self._sampling_probabilities()
            count = min(batch_size, int(np.count_nonzero(probabilities)))
            indices = self._rng.choice(len(self._samples), size=count, replace=False, p=probabilities)
            return [self._samples[int(index)] for index in indices]

    def composition(self) -> dict[str, object]:
        """Raw sample counts and the effective sampling share per outcome class."""
        with self._lock:
            if self._composition is not None:
                return self._composition
            counts = Counter(sample.outcome for sample in self._samples)
            shares: dict[str, float] = {name: 0.0 for name in OUTCOME_CLASSES}
            if self._samples:
                probabilities = self._sampling_probabilities()
                for sample, probability in zip(self._samples, probabilities, strict=True):
                    shares[sample.outcome] = shares.get(sample.outcome, 0.0) + float(probability)
            self._composition = {
                "size": len(self._samples),
                "counts": {name: counts.get(name, 0) for name in OUTCOME_CLASSES},
                "sampling_share": shares,
                "max_draw_fraction": self.max_draw_fraction,
            }
            return self._composition

    def snapshot(self) -> list[TrainingSample]:
        with self._lock:
            return list(self._samples)

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)
