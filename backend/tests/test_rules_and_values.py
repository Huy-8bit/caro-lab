"""Terminal rules, reward signs and value-target perspective."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from app.chess.mcts import MCTS, draw_value, terminal_value
from app.chess.rules import MAX_MOVE_TERMINATION, game_outcome, search_copy, termination_name
from app.training.network import NetworkSpec, PolicyValueNetwork
from app.training.replay import ADJUDICATED, DECISIVE, DRAW, REPETITION
from app.training.self_play import classify_outcome, play_self_play_game, value_target
from conftest import KNIGHT_CYCLE, board_after


def test_checkmate_is_a_loss_for_the_side_to_move() -> None:
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert game_outcome(mate).termination == chess.Termination.CHECKMATE
    assert terminal_value(mate) == -1.0
    assert classify_outcome(mate, game_outcome(mate)) == (DECISIVE, 1.0)


def test_stalemate_and_insufficient_material_are_draws() -> None:
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert termination_name(game_outcome(stale)) == "stalemate"
    assert terminal_value(stale) == 0.0
    bare = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    assert termination_name(game_outcome(bare)) == "insufficient material"
    assert classify_outcome(bare, game_outcome(bare)) == (DRAW, 0.0)


def test_fifty_move_rule_applies_at_one_hundred_half_moves_only() -> None:
    assert game_outcome(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 99 80")) is None
    fifty = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")
    assert termination_name(game_outcome(fifty)) == "fifty moves"


def test_checkmate_takes_priority_over_the_fifty_move_rule() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 90")
    assert game_outcome(board).termination == chess.Termination.CHECKMATE


def test_threefold_requires_an_actual_third_occurrence() -> None:
    # The old adjudicator used outcome(claim_draw=True), which ends the game
    # here although the position has occurred only twice.
    board = board_after(*KNIGHT_CYCLE, *KNIGHT_CYCLE[:3])
    assert board.outcome(claim_draw=True) is not None
    assert not board.is_repetition(3)
    assert game_outcome(board) is None
    board.push_uci(KNIGHT_CYCLE[3])
    assert termination_name(game_outcome(board)) == "threefold repetition"
    assert classify_outcome(board, game_outcome(board)) == (REPETITION, 0.0)


def test_search_copy_keeps_the_history_needed_for_repetition() -> None:
    board = board_after("e2e4", "e7e5", *KNIGHT_CYCLE, *KNIGHT_CYCLE[:3])
    copied = search_copy(board)
    assert len(copied.move_stack) == board.halfmove_clock == 7
    copied.push_uci(KNIGHT_CYCLE[3])
    assert copied.is_repetition(3)
    # A history-less copy (the old MCTS behaviour) misses it.
    blind = board.copy(stack=False)
    blind.push_uci(KNIGHT_CYCLE[3])
    assert not blind.is_repetition(3)


def test_value_target_follows_the_side_to_move() -> None:
    assert value_target(1.0, chess.WHITE) == 1.0
    assert value_target(1.0, chess.BLACK) == -1.0
    assert value_target(-1.0, chess.BLACK) == 1.0
    assert value_target(0.0, chess.BLACK) == 0.0


def test_material_adjudication_keeps_the_white_perspective_sign() -> None:
    white_up = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    outcome_class, white_value = classify_outcome(white_up, None)
    assert outcome_class == ADJUDICATED and white_value > 0
    assert value_target(white_value, white_up.turn) < 0  # black to move is losing


def test_draw_contempt_is_negamax_consistent() -> None:
    # The player choosing the move dislikes the draw; its opponent sees the
    # mirror value, so a draw backed up one ply is again -contempt for the root.
    root = chess.WHITE
    assert draw_value(chess.WHITE, root, 0.2) == pytest.approx(-0.2)
    assert draw_value(chess.BLACK, root, 0.2) == pytest.approx(0.2)
    assert -draw_value(chess.BLACK, root, 0.2) == draw_value(chess.WHITE, root, 0.2)


class _MateInOne(torch.nn.Module):
    """Uniform priors and a neutral value: only exact terminals matter."""

    def __init__(self) -> None:
        super().__init__()
        self.spec = NetworkSpec(channels=8, residual_blocks=0)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return torch.zeros(inputs.shape[0], 20_480), torch.zeros(inputs.shape[0])


def test_mcts_backs_up_a_mate_as_a_win_for_the_mover() -> None:
    board = chess.Board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1")  # Qg7# is available
    search = MCTS(_MateInOne(), torch.device("cpu"), simulations=300, seed=1)
    _, root_value, root = search.search(board, add_noise=False)
    mate = chess.Move.from_uci("g1g7")
    assert root.children[mate].terminal
    assert root.children[mate].value == pytest.approx(-1.0)  # checkmated side to move
    chosen = search.select_move(root, 0)
    board.push(chosen)
    assert board.is_checkmate()
    assert root_value > 0.5


def test_self_play_value_targets_are_zero_sum_per_game() -> None:
    torch.manual_seed(0)
    model = PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=0))
    search = MCTS(model, torch.device("cpu"), simulations=4, seed=3)
    result = play_self_play_game(search, 1, 0, max_moves=30, temperature_moves=30, temperature=1.0, opening_plies=2)
    meta = result.record["metadata"]
    white_value = meta["value_target"]
    for sample in result.samples:
        white_to_move = sample.state[12, 0, 0] == 1.0
        assert sample.value == pytest.approx(white_value if white_to_move else -white_value, abs=1e-5)
    if meta["outcome_class"] in {DRAW, REPETITION}:
        assert all(sample.value == 0.0 for sample in result.samples)
    # Opening plies are played but never become training samples.
    assert result.record["moves"][0]["opening"] and result.record["moves"][1]["opening"]
    assert all(sample.ply >= 2 for sample in result.samples)
    assert np.isclose(result.samples[0].policy_prob.sum(), 1.0)


def test_max_move_limit_is_reported_and_adjudicated() -> None:
    model = PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=0))
    search = MCTS(model, torch.device("cpu"), simulations=1, seed=4)
    result = play_self_play_game(search, 1, 0, max_moves=2, temperature_moves=1, temperature=1.0)
    assert result.record["termination"] == MAX_MOVE_TERMINATION
    assert result.record["metadata"]["outcome_class"] == ADJUDICATED
    assert result.record["metadata"]["value_target_source"] == "material_bootstrap"
