from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.replay import TrainingSample
from app.training.self_play import SelfPlayResult


def encode_samples(samples: list[TrainingSample]) -> str:
    if not samples:
        raise ValueError("A distributed result must contain training samples")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        states=np.stack([sample.state for sample in samples]).astype(np.float32, copy=False),
        policies=np.stack([sample.policy for sample in samples]).astype(np.float32, copy=False),
        values=np.asarray([sample.value for sample in samples], dtype=np.float32),
        plies=np.asarray([sample.ply for sample in samples], dtype=np.int32),
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_result(record: dict[str, Any], samples_blob: str, iteration: int) -> SelfPlayResult:
    raw = base64.b64decode(samples_blob.encode("ascii"), validate=True)
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        states = arrays["states"]
        policies = arrays["policies"]
        values = arrays["values"]
        plies = arrays["plies"]
    count = len(values)
    if states.shape[0] != count or policies.shape[0] != count or len(plies) != count:
        raise ValueError("Distributed sample arrays have inconsistent lengths")
    game_id = str(record["id"])
    samples = [
        TrainingSample(
            state=states[index],
            policy=policies[index],
            value=float(values[index]),
            game_id=game_id,
            ply=int(plies[index]),
            iteration=iteration,
        )
        for index in range(count)
    ]
    return SelfPlayResult(record=record, samples=samples)


def save_worker_model(
    path: str,
    model: PolicyValueNetwork,
    revision: int,
    model_version: int,
) -> None:
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
