from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from app.chess.encoding import encode_board
from app.chess.move_index import POLICY_SIZE, move_to_index
from app.chess.rules import game_outcome, search_copy


@dataclass
class SearchNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "SearchNode"] = field(default_factory=dict)
    terminal: bool = False
    # Root only: network value and noise-free priors, for the improved policy.
    network_value: float = 0.0
    raw_priors: dict[chess.Move, float] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        """Mean value from the perspective of the player to move at this node."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def terminal_value(board: chess.Board) -> float | None:
    """Exact game result from the perspective of the player to move."""
    outcome = game_outcome(board)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def draw_value(node_turn: chess.Color, root_turn: chess.Color, contempt: float) -> float:
    """Value of a draw for the player to move at a node, with root contempt.

    The player who is actually choosing a move (the root) treats a draw as
    ``-contempt``. In a negamax tree the same draw is then ``+contempt`` for the
    opponent, so values stay zero-sum along every edge. Each self-play move is a
    fresh search, so both colours dislike draws when it is their own decision,
    while the training value target for a draw stays exactly 0.
    """
    return -contempt if node_turn == root_turn else contempt


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        draw_contempt: float = 0.0,
        seed: int = 0,
        policy_mode: str = "visits",
        value_scale: float = 0.1,
        visit_offset: float = 50.0,
    ) -> None:
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.draw_contempt = draw_contempt
        self.policy_mode = policy_mode
        self.value_scale = value_scale
        self.visit_offset = visit_offset
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Priors over legal moves and the value, with the model in eval mode."""
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            return self._evaluate(board)
        finally:
            if was_training:
                self.model.train()

    @torch.inference_mode()
    def _evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        legal = list(board.legal_moves)
        if not legal:
            return {}, terminal_value(board) or 0.0
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        indices = torch.tensor([move_to_index(move) for move in legal], device=self.device)
        if hasattr(self.model, "evaluate_moves"):
            logits, value = self.model.evaluate_moves(tensor, indices)
            logits = logits[0]
        else:
            full_logits, value = self.model(tensor)
            logits = full_logits[0, indices]
        priors = torch.softmax(logits, dim=0).cpu().numpy()
        return {move: float(prior) for move, prior in zip(legal, priors, strict=True)}, float(value.item())

    def _expand(self, node: SearchNode, board: chess.Board) -> float:
        priors, value = self._evaluate(board)
        node.children = {move: SearchNode(prior=prior) for move, prior in priors.items()}
        return value

    def _select_child(self, node: SearchNode) -> tuple[chess.Move, SearchNode]:
        root_scale = math.sqrt(max(1, node.visit_count))
        return max(
            node.children.items(),
            key=lambda item: -item[1].value
            + self.c_puct * item[1].prior * root_scale / (1 + item[1].visit_count),
        )

    def _outcome_value(self, board: chess.Board, root_turn: chess.Color) -> float | None:
        outcome = game_outcome(board)
        if outcome is None:
            return None
        if outcome.winner is None:
            return draw_value(board.turn, root_turn, self.draw_contempt)
        return 1.0 if outcome.winner == board.turn else -1.0

    def search(self, board: chess.Board, add_noise: bool = True) -> tuple[np.ndarray, float, SearchNode]:
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            return self._search(board, add_noise)
        finally:
            if was_training:
                self.model.train()

    def _search(self, board: chess.Board, add_noise: bool) -> tuple[np.ndarray, float, SearchNode]:
        """Run PUCT search from ``board`` using its real game history.

        A leaf is scored exactly (no network call) when it ends the game under
        :func:`game_outcome`, including a threefold repetition that involves
        positions played before the root, or when the simulated line returns to
        a position already on the current path from the root. The latter is a
        cycle the side to move could keep repeating, so it is scored as a draw;
        this keeps the search from chasing an infinite shuffle whose network
        value happens to look attractive.
        """
        root = SearchNode()
        root_board = search_copy(board)
        root_turn = root_board.turn
        value = self._outcome_value(root_board, root_turn)
        if value is not None:
            root.terminal = True
            return np.zeros(POLICY_SIZE, dtype=np.float32), value, root
        root_key = root_board._transposition_key()
        root.network_value = self._expand(root, root_board)
        root.raw_priors = {move: child.prior for move, child in root.children.items()}
        if add_noise and root.children and self.dirichlet_epsilon > 0:
            moves = list(root.children)
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
            for move, sample in zip(moves, noise, strict=True):
                child = root.children[move]
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(sample)

        # Simulations push moves onto root_board and pop them afterwards, so the
        # real game history stays visible to repetition checks without copying.
        for _ in range(max(1, self.simulations)):
            node = root
            path = [node]
            seen = {root_key}
            leaf_value: float | None = None
            while node.expanded:
                move, node = self._select_child(node)
                root_board.push(move)
                path.append(node)
                key = root_board._transposition_key()
                if key in seen:
                    node.terminal = True
                    leaf_value = draw_value(root_board.turn, root_turn, self.draw_contempt)
                    break
                seen.add(key)
            if leaf_value is None:
                leaf_value = self._outcome_value(root_board, root_turn)
                if leaf_value is not None:
                    node.terminal = True
                else:
                    leaf_value = self._expand(node, root_board)
            for _ in range(len(path) - 1):
                root_board.pop()
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += leaf_value
                leaf_value = -leaf_value

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        moves, probabilities = self.root_policy(root)
        for move, probability in zip(moves, probabilities, strict=True):
            policy[move_to_index(move)] = probability
        return policy, root.value, root

    def improved_policy(self, root: SearchNode) -> tuple[list[chess.Move], np.ndarray]:
        """Policy improvement with completed Q-values (Danihelka et al., 2022).

        ``softmax(log prior + sigma(q))`` where visited moves use their search
        value, unvisited moves the prior-weighted mixed value, values are
        min-max normalised, and ``sigma = (visit_offset + max N) * value_scale``.
        With a small simulation budget, PUCT visit counts mostly echo the
        prior; training on them lets the network reinforce whatever it already
        prefers, including shuffles into a repetition the search has proven
        worse. This target moves probability towards better-valued moves.
        """
        moves = list(root.children)
        prior = np.array([root.raw_priors.get(move, root.children[move].prior) for move in moves], dtype=np.float64)
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)
        q = np.array([-root.children[move].value for move in moves], dtype=np.float64)
        visited = visits > 0
        if visited.any():
            weight = prior[visited].sum()
            mean_q = (prior[visited] * q[visited]).sum() / weight if weight > 0 else q[visited].mean()
            mixed = (root.network_value + visits.sum() * mean_q) / (1 + visits.sum())
        else:
            mixed = root.network_value
        completed = np.where(visited, q, mixed)
        low, high = completed.min(), completed.max()
        normalised = (completed - low) / max(high - low, 1e-8)
        logits = np.log(np.maximum(prior, 1e-12)) + (self.visit_offset + visits.max()) * self.value_scale * normalised
        probabilities = np.exp(logits - logits.max())
        return moves, probabilities / probabilities.sum()

    def root_policy(self, root: SearchNode) -> tuple[list[chess.Move], np.ndarray]:
        """Training target over root moves for the configured ``policy_mode``."""
        moves = list(root.children)
        if not moves:
            return [], np.zeros(0)
        if self.policy_mode == "completed_q":
            return self.improved_policy(root)
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)
        return moves, visits / max(1.0, visits.sum())

    def select_move(self, root: SearchNode, temperature: float, board: chess.Board | None = None) -> chess.Move:
        """Pick a move among the root moves the search actually examined.

        Greedy play takes the most probable move of :meth:`root_policy`;
        temperature sampling draws from it. ``board`` is accepted for API
        compatibility: repetition is handled by search values, not by filtering
        legal moves after the fact.
        """
        moves, probabilities = self.root_policy(root)
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)
        if visits.sum() > 0:
            probabilities = np.where(visits > 0, probabilities, 0.0)
        if probabilities.sum() <= 0:
            probabilities = np.ones(len(moves))
        if temperature <= 1e-6:
            best = np.flatnonzero(probabilities >= probabilities.max() * (1 - 1e-9))
            return moves[int(best[0] if len(best) == 1 else self.rng.choice(best))]
        weights = np.power(probabilities / probabilities.max(), 1.0 / temperature)
        weights /= weights.sum()
        return moves[int(self.rng.choice(len(moves), p=weights))]
