from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass(slots=True)
class TrainingSample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    game_id: str
    ply: int
    iteration: int


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._samples: deque[TrainingSample] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)
        self._lock = Lock()

    def extend(self, samples: list[TrainingSample]) -> None:
        with self._lock:
            self._samples.extend(samples)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        with self._lock:
            if not self._samples:
                return []
            count = min(batch_size, len(self._samples))
            indices = self._rng.choice(len(self._samples), size=count, replace=False)
            snapshot = list(self._samples)
        return [snapshot[int(index)] for index in indices]

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)
