"""MCTS must see real game history and must not chase repetition cycles."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from app.chess.mcts import MCTS, SearchNode
from app.chess.move_index import POLICY_SIZE, move_to_index
from app.training.network import NetworkSpec
from conftest import KNIGHT_CYCLE, board_after


class ScriptedModel(torch.nn.Module):
    """Policy/value stub: optional boosted moves, constant value."""

    def __init__(self, boosted: dict[str, float] | None = None, value: float = 0.0, white_value: float | None = None) -> None:
        super().__init__()
        self.spec = NetworkSpec(channels=8, residual_blocks=0)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.logits = torch.zeros(POLICY_SIZE)
        for uci, logit in (boosted or {}).items():
            self.logits[move_to_index(chess.Move.from_uci(uci))] = logit
        self.constant = value
        self.white_value = white_value

    def forward(self, inputs):
        batch = inputs.shape[0]
        if self.white_value is not None:
            # Consistent evaluation: White is better everywhere (plane 12 = White to move).
            white_to_move = inputs[:, 12, 0, 0] > 0.5
            values = torch.where(white_to_move, self.white_value, -self.white_value)
        else:
            values = torch.full((batch,), self.constant)
        return self.logits.expand(batch, -1).clone(), values


def _walk(node: SearchNode, board: chess.Board, ancestors: set, visit) -> None:
    key = board._transposition_key()
    for move, child in node.children.items():
        board.push(move)
        visit(child, board, ancestors | {key})
        _walk(child, board, ancestors | {key}, visit)
        board.pop()


def test_move_into_real_threefold_is_scored_as_a_contempt_draw() -> None:
    # Black to move; ...Ng8 would produce the third occurrence of the start
    # position. That depends on history before the root, which the old search
    # discarded with board.copy(stack=False).
    board = board_after(*KNIGHT_CYCLE, *KNIGHT_CYCLE[:3])
    repeat = chess.Move.from_uci("f6g8")
    search = MCTS(ScriptedModel(), torch.device("cpu"), simulations=400, draw_contempt=0.2, seed=1)
    _, _, root = search.search(board, add_noise=False)
    child = root.children[repeat]
    assert child.terminal
    # Child value is from White's view (to move after ...Ng8): +contempt,
    # i.e. -contempt for Black, the player actually choosing.
    assert child.value == pytest.approx(0.2)
    others = [node.visit_count for move, node in root.children.items() if move != repeat]
    assert child.visit_count < sum(others) / len(others)
    assert search.select_move(root, 0) != repeat


def test_losing_side_may_still_choose_the_repetition_draw() -> None:
    # Draw aversion is small; a side whose alternatives are clearly worse still
    # takes the draw, which is correct chess and not hard-coded.
    board = board_after(*KNIGHT_CYCLE, *KNIGHT_CYCLE[:3])
    search = MCTS(ScriptedModel(white_value=0.6), torch.device("cpu"), simulations=600, draw_contempt=0.1, seed=1)
    # White is +0.6 in every non-terminal position, so every alternative is
    # worth -0.6 to Black (to move), worse than a -0.1 draw.
    _, _, root = search.search(board, add_noise=False)
    assert search.select_move(root, 0) == chess.Move.from_uci("f6g8")


def test_in_tree_cycles_are_terminal_draws_not_network_evaluations() -> None:
    # The network loves the knight shuffle and rates every position +0.9 for
    # the side to move. Positions that return to one already on the path must
    # never be expanded or evaluated: they are scored as draws.
    model = ScriptedModel({uci: 6.0 for uci in KNIGHT_CYCLE}, value=0.9)
    search = MCTS(model, torch.device("cpu"), simulations=600, draw_contempt=0.15, seed=2)
    board = chess.Board()
    _, _, root = search.search(board, add_noise=False)
    cycles = []

    def visit(node: SearchNode, position: chess.Board, ancestors: set) -> None:
        if position._transposition_key() in ancestors and node.visit_count:
            cycles.append((node, position.turn))

    _walk(root, board.copy(), set(), visit)
    assert cycles, "the shuffle-loving prior should have explored at least one cycle"
    for node, turn in cycles:
        assert node.terminal and not node.children
        expected = -0.15 if turn == chess.WHITE else 0.15
        assert node.value == pytest.approx(expected)


def test_completed_q_policy_does_not_follow_a_prior_onto_a_proven_draw() -> None:
    # With visit counts, a repeating move with ~70% prior gets the most PUCT
    # visits at 32 simulations although the search has proven it a draw. The
    # optional completed-Q policy (policy_target="completed_q") must not pick
    # it, and its target must move probability away from it.
    board = board_after(*KNIGHT_CYCLE, *KNIGHT_CYCLE[:3])
    repeat = chess.Move.from_uci("f6g8")
    model = ScriptedModel({"f6g8": 4.0})  # ~70% prior on the repetition
    visits_search = MCTS(model, torch.device("cpu"), simulations=32, draw_contempt=0.15, dirichlet_epsilon=0.0, seed=7, policy_mode="visits")
    _, _, root = visits_search.search(board, add_noise=False)
    assert visits_search.select_move(root, 0) == repeat  # the old behaviour
    search = MCTS(model, torch.device("cpu"), simulations=32, draw_contempt=0.15, dirichlet_epsilon=0.0, seed=7, policy_mode="completed_q")
    policy, _, root = search.search(board, add_noise=False)
    assert search.select_move(root, 0) != repeat
    assert policy[move_to_index(repeat)] < root.raw_priors[repeat]
    assert abs(policy.sum() - 1.0) < 1e-5
    assert set(np.flatnonzero(policy)) <= {move_to_index(move) for move in board.legal_moves}


def test_greedy_choice_is_always_a_searched_move() -> None:
    search = MCTS(ScriptedModel(), torch.device("cpu"), simulations=5, seed=8)
    _, _, root = search.search(chess.Board(), add_noise=False)
    for _ in range(20):
        assert root.children[search.select_move(root, 0)].visit_count > 0
        assert root.children[search.select_move(root, 1.0)].visit_count > 0


def test_self_play_repetition_draws_are_real_threefolds() -> None:
    # A shuffle-loving network will repeat; the game must only be adjudicated
    # once the position has truly occurred three times (old code stopped at
    # the second occurrence because the next move *could* claim).
    from app.training.self_play import play_self_play_game

    model = ScriptedModel({uci: 8.0 for uci in KNIGHT_CYCLE})
    search = MCTS(model, torch.device("cpu"), simulations=16, dirichlet_epsilon=0.0, seed=6)
    result = play_self_play_game(search, 1, 0, max_moves=40, temperature_moves=0, temperature=0.0)
    assert result.record["termination"] == "threefold repetition"
    final = chess.Board()
    for move in result.record["moves"]:
        final.push_uci(move["uci"])
    assert final.is_repetition(3)
    # Repeated positions are not recorded twice as training samples.
    assert result.record["metadata"]["duplicate_positions_skipped"] > 0
    assert len(result.samples) == len({sample.state.tobytes() for sample in result.samples})


def test_evaluation_search_uses_no_dirichlet_noise() -> None:
    model = ScriptedModel({"e2e4": 3.0})
    search = MCTS(model, torch.device("cpu"), simulations=8, dirichlet_epsilon=0.0, seed=4)
    _, _, first = search.search(chess.Board(), add_noise=True)
    _, _, second = MCTS(model, torch.device("cpu"), simulations=8, dirichlet_epsilon=0.0, seed=99).search(chess.Board(), add_noise=True)
    assert {m: c.prior for m, c in first.children.items()} == {m: c.prior for m, c in second.children.items()}


def test_search_leaves_the_callers_board_untouched() -> None:
    board = board_after("e2e4", "e7e5")
    before = (board.fen(), list(board.move_stack))
    MCTS(ScriptedModel(), torch.device("cpu"), simulations=50, seed=5).search(board)
    assert (board.fen(), list(board.move_stack)) == before
