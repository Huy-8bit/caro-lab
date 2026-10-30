from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np
import torch

from app.chess.encoding import encode_board
from app.chess.move_index import POLICY_SIZE, move_to_index


@dataclass
class SearchNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "SearchNode"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def terminal_value(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        was_training = self.model.training
        self.model.eval()
        logits, value = self.model(tensor)
        if was_training:
            self.model.train()
        legal = list(board.legal_moves)
        if not legal:
            return {}, terminal_value(board) or 0.0
        indices = torch.tensor([move_to_index(move) for move in legal], device=logits.device)
        priors = torch.softmax(logits[0, indices], dim=0).detach().cpu().numpy()
        return {move: float(prior) for move, prior in zip(legal, priors, strict=True)}, float(value.item())

    def _expand(self, node: SearchNode, board: chess.Board) -> float:
        priors, value = self.evaluate(board)
        node.children = {move: SearchNode(prior=prior) for move, prior in priors.items()}
        return value

    def _select_child(self, node: SearchNode) -> tuple[chess.Move, SearchNode]:
        root_scale = math.sqrt(max(1, node.visit_count))
        return max(
            node.children.items(),
            key=lambda item: -item[1].value
            + self.c_puct * item[1].prior * root_scale / (1 + item[1].visit_count),
        )

    def search(self, board: chess.Board, add_noise: bool = True) -> tuple[np.ndarray, float, SearchNode]:
        root = SearchNode()
        value = terminal_value(board)
        if value is not None:
            return np.zeros(POLICY_SIZE, dtype=np.float32), value, root
        root_value = self._expand(root, board)
        if add_noise and root.children:
            moves = list(root.children)
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
            for move, sample in zip(moves, noise, strict=True):
                child = root.children[move]
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(sample)

        for _ in range(max(1, self.simulations)):
            simulation_board = board.copy(stack=False)
            node = root
            path = [node]
            while node.expanded:
                move, node = self._select_child(node)
                simulation_board.push(move)
                path.append(node)
            leaf_value = terminal_value(simulation_board)
            if leaf_value is None:
                leaf_value = self._expand(node, simulation_board)
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += leaf_value
                leaf_value = -leaf_value

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        total = sum(child.visit_count for child in root.children.values())
        if total:
            for move, child in root.children.items():
                policy[move_to_index(move)] = child.visit_count / total
        return policy, root_value, root

    def select_move(self, root: SearchNode, temperature: float) -> chess.Move:
        moves = list(root.children)
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)
        if temperature <= 1e-6:
            return moves[int(np.argmax(visits))]
        weights = np.power(visits + 1e-10, 1.0 / temperature)
        weights /= weights.sum()
        return moves[int(self.rng.choice(len(moves), p=weights))]
