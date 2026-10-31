from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from app.chess.encoding import INPUT_PLANES
from app.chess.move_index import POLICY_SIZE


@dataclass(slots=True)
class NetworkSpec:
    channels: int = 48
    residual_blocks: int = 2
    policy_size: int = POLICY_SIZE
    input_planes: int = INPUT_PLANES

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class PolicyValueNetwork(nn.Module):
    def __init__(self, spec: NetworkSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or NetworkSpec()
        c = self.spec.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.spec.input_planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(c) for _ in range(self.spec.residual_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, self.spec.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(inputs))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def evaluate_moves(self, inputs: torch.Tensor, move_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy logits for ``move_indices`` only, plus the value.

        Same result as ``forward(inputs)[0][:, move_indices]``, but the final
        policy layer multiplies only the rows of the legal moves instead of all
        ``policy_size`` (20480) outputs. That layer dominates inference cost for
        the small networks used here, so search runs several times faster.
        """
        features = self.tower(self.stem(inputs))
        hidden = self.policy_head[:-1](features)
        output = self.policy_head[-1]
        logits = torch.addmm(output.bias[move_indices], hidden, output.weight[move_indices].T)
        return logits, self.value_head(features).squeeze(-1)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
