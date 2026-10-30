from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from app.storage.repository import Repository
from app.training.network import NetworkSpec, PolicyValueNetwork


def save_checkpoint(
    directory: Path,
    repository: Repository,
    model: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    version: int,
    iteration: int,
    games: int,
    step: int,
    config: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"model-v{version:04d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "version": version,
            "iteration": iteration,
            "games": games,
            "step": step,
            "spec": model.spec.to_dict(),
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    repository.save_checkpoint(
        {
            "version": version,
            "path": str(path),
            "created_at": datetime.now(UTC).isoformat(),
            "iteration": iteration,
            "games": games,
            "architecture": model.spec.to_dict(),
            "config": config,
            "evaluation": {},
            "file_size": path.stat().st_size,
        }
    )
    return path


def load_checkpoint(
    path: str | Path, device: torch.device, optimizer: torch.optim.Optimizer | None = None
) -> tuple[PolicyValueNetwork, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PolicyValueNetwork(NetworkSpec(**payload["spec"])).to(device)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return model, payload
