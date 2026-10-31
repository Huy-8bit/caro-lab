from __future__ import annotations

import base64
import io
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.replay import DECISIVE, OUTCOME_CLASSES, TrainingSample
from app.training.self_play import SelfPlayResult


# ---------------------------------------------------------------------------
# Job leasing
# ---------------------------------------------------------------------------

QUEUED, LEASED, COMPLETE, CANCELLED = "queued", "leased", "complete", "cancelled"


@dataclass
class Lease:
    job: dict[str, Any]
    batch: str
    status: str = QUEUED
    worker_id: str | None = None
    heartbeat_at: float | None = None
    attempts: int = 0
    history: list[str] = field(default_factory=list)


class JobBoard:
    """Authoritative lease table for work handed to stateless workers.

    Guarantees, all under one lock:

    * a job is leased to at most one worker at a time;
    * a lease without a heartbeat for ``lease_timeout`` seconds is requeued;
    * when a worker claims again while holding a lease (it restarted, or its
      container was replaced under the same hostname) the old job is requeued;
    * a worker may release its job (graceful shutdown) for immediate requeue;
    * only the first result for a job is accepted, from whichever worker
      finishes it first; later duplicates are rejected.
    """

    def __init__(self, lease_timeout: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.lease_timeout = lease_timeout
        self._clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, Lease] = {}
        self._queue: deque[str] = deque()
        self._results: dict[str, queue.Queue[tuple[str, Any]]] = {}
        self.on_requeue: Callable[[Lease, str], None] | None = None

    def open_batch(self, batch: str) -> None:
        with self._lock:
            self._results.setdefault(batch, queue.Queue())

    def submit(self, batch: str, job: dict[str, Any]) -> str:
        with self._lock:
            job_id = job.setdefault("job_id", uuid.uuid4().hex)
            self._results.setdefault(batch, queue.Queue())
            self._jobs[job_id] = Lease(job=job, batch=batch)
            self._queue.append(job_id)
            return job_id

    def _requeue(self, lease: Lease, reason: str, front: bool = False) -> None:
        lease.history.append(f"{reason}:{lease.worker_id}")
        lease.status = QUEUED
        lease.worker_id = None
        lease.heartbeat_at = None
        job_id = lease.job["job_id"]
        if front:
            self._queue.appendleft(job_id)
        else:
            self._queue.append(job_id)
        if self.on_requeue:
            self.on_requeue(lease, reason)

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        with self._lock:
            for lease in self._jobs.values():
                if lease.status == LEASED and lease.worker_id == worker_id:
                    self._requeue(lease, "abandoned", front=True)
            while self._queue:
                job_id = self._queue.popleft()
                lease = self._jobs.get(job_id)
                if lease is None or lease.status != QUEUED:
                    continue
                lease.status = LEASED
                lease.worker_id = worker_id
                lease.heartbeat_at = self._clock()
                lease.attempts += 1
                return lease.job
            return None

    def heartbeat(self, worker_id: str, job_id: str) -> bool:
        """Renew a lease. False tells the worker to abandon the job."""
        with self._lock:
            lease = self._jobs.get(job_id)
            if lease is None or lease.status != LEASED or lease.worker_id != worker_id:
                return False
            lease.heartbeat_at = self._clock()
            return True

    def release(self, worker_id: str, job_id: str) -> bool:
        with self._lock:
            lease = self._jobs.get(job_id)
            if lease is None or lease.status != LEASED or lease.worker_id != worker_id:
                return False
            self._requeue(lease, "released", front=True)
            return True

    def complete(self, worker_id: str, job_id: str, result: Any) -> bool:
        with self._lock:
            lease = self._jobs.get(job_id)
            if lease is None or lease.status in {COMPLETE, CANCELLED}:
                return False
            lease.status = COMPLETE
            lease.worker_id = worker_id
            results = self._results.get(lease.batch)
            if results is None:
                return False
            results.put((job_id, result))
            return True

    def expire(self) -> list[str]:
        now = self._clock()
        expired = []
        with self._lock:
            for lease in self._jobs.values():
                if lease.status == LEASED and lease.heartbeat_at is not None and now - lease.heartbeat_at > self.lease_timeout:
                    expired.append(lease.job["job_id"])
                    self._requeue(lease, "expired")
        return expired

    def next_result(self, batch: str, timeout: float) -> tuple[str, Any] | None:
        """Take one result; its lease is forgotten once it has been consumed."""
        with self._lock:
            results = self._results.setdefault(batch, queue.Queue())
        try:
            item = results.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            lease = self._jobs.get(item[0])
            if lease is not None and lease.status == COMPLETE:
                del self._jobs[item[0]]
        return item

    def pending_results(self, batch: str) -> int:
        """Results of ``batch`` that are finished but not yet consumed."""
        with self._lock:
            results = self._results.get(batch)
            return results.qsize() if results is not None else 0

    def close_batch(self, batch: str) -> None:
        """Cancel unfinished jobs of a batch and forget all of its leases."""
        with self._lock:
            for job_id, lease in list(self._jobs.items()):
                if lease.batch == batch:
                    if lease.status != COMPLETE:
                        lease.status = CANCELLED
                    del self._jobs[job_id]
            self._results.pop(batch, None)

    def outstanding(self, batch: str) -> int:
        """Jobs of ``batch`` that are queued or being played."""
        with self._lock:
            return sum(1 for lease in self._jobs.values() if lease.batch == batch and lease.status in {QUEUED, LEASED})

    def update_queued(self, batch: str, fields: dict[str, Any]) -> int:
        """Rewrite fields (e.g. the model key) of jobs nobody has started yet."""
        with self._lock:
            updated = 0
            for lease in self._jobs.values():
                if lease.batch == batch and lease.status == QUEUED:
                    lease.job.update(fields)
                    updated += 1
            return updated

    def cancel_queued(self, batch: str) -> int:
        """Drop jobs of ``batch`` nobody has started; leased jobs keep running."""
        with self._lock:
            cancelled = [job_id for job_id, lease in self._jobs.items() if lease.batch == batch and lease.status == QUEUED]
            for job_id in cancelled:
                self._jobs.pop(job_id).status = CANCELLED
            return len(cancelled)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._queue.clear()
            self._results.clear()

    def get(self, job_id: str) -> Lease | None:
        with self._lock:
            return self._jobs.get(job_id)

    def counts(self) -> dict[str, int]:
        with self._lock:
            counts = {QUEUED: 0, LEASED: 0, COMPLETE: 0}
            for lease in self._jobs.values():
                if lease.status in counts:
                    counts[lease.status] += 1
            return counts


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def encode_samples(samples: list[TrainingSample]) -> str:
    if not samples:
        raise ValueError("A distributed result must contain training samples")
    buffer = io.BytesIO()
    lengths = np.asarray([len(sample.policy_index) for sample in samples], dtype=np.int32)
    np.savez_compressed(
        buffer,
        states=np.stack([sample.state for sample in samples]).astype(np.float32, copy=False),
        policy_lengths=lengths,
        policy_index=np.concatenate([sample.policy_index for sample in samples]).astype(np.int32, copy=False),
        policy_prob=np.concatenate([sample.policy_prob for sample in samples]).astype(np.float32, copy=False),
        values=np.asarray([sample.value for sample in samples], dtype=np.float32),
        plies=np.asarray([sample.ply for sample in samples], dtype=np.int32),
        weights=np.asarray([sample.weight for sample in samples], dtype=np.float32),
        outcomes=np.asarray([OUTCOME_CLASSES.index(sample.outcome) for sample in samples], dtype=np.int8),
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_result(record: dict[str, Any], samples_blob: str, iteration: int) -> SelfPlayResult:
    raw = base64.b64decode(samples_blob.encode("ascii"), validate=True)
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        states = arrays["states"]
        lengths = arrays["policy_lengths"]
        policy_index = arrays["policy_index"]
        policy_prob = arrays["policy_prob"]
        values = arrays["values"]
        plies = arrays["plies"]
        weights = arrays["weights"]
        outcomes = arrays["outcomes"]
    count = len(values)
    if any(len(array) != count for array in (states, lengths, plies, weights, outcomes)):
        raise ValueError("Distributed sample arrays have inconsistent lengths")
    if int(lengths.sum()) != len(policy_index) or len(policy_index) != len(policy_prob):
        raise ValueError("Distributed policy arrays have inconsistent lengths")
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError("Distributed value targets must lie in [-1, 1]")
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    game_id = str(record["id"])
    samples = [
        TrainingSample(
            state=states[index],
            policy_index=policy_index[offsets[index]:offsets[index + 1]],
            policy_prob=policy_prob[offsets[index]:offsets[index + 1]],
            value=float(values[index]),
            game_id=game_id,
            ply=int(plies[index]),
            iteration=iteration,
            weight=float(weights[index]),
            outcome=OUTCOME_CLASSES[int(outcomes[index])] if 0 <= int(outcomes[index]) < len(OUTCOME_CLASSES) else DECISIVE,
        )
        for index in range(count)
    ]
    return SelfPlayResult(record=record, samples=samples)


def save_worker_model(path: str | Path, model: PolicyValueNetwork, revision: int, model_version: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(
        {
            "revision": revision,
            "model_version": model_version,
            "spec": model.spec.to_dict(),
            "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(target)


def load_worker_model(payload: bytes, device: torch.device) -> tuple[PolicyValueNetwork, dict[str, Any]]:
    metadata = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    model = PolicyValueNetwork(NetworkSpec(**metadata["spec"])).to(device)
    model.load_state_dict(metadata["model_state"])
    model.eval()
    return model, metadata
