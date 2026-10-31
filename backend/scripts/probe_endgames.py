"""Read-only probe: does a checkpoint understand material and can it mate?

    python scripts/probe_endgames.py checkpoints/training-latest.pt --positions 20

1. Value sanity: mean network value of random won positions (KQ v K, KR v K,
   KQ v KR) from the stronger side's perspective, separately for each side to
   move. A network that learned material should be clearly positive.
2. Conversion: plays random KQ v K and KR v K positions with greedy,
   noise-free search for both sides and counts checkmates within 50 moves.

Positions come from the curriculum generator with fixed seeds, so two
checkpoints are compared on identical positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chess.encoding import encode_board  # noqa: E402
from app.chess.endgames import FAMILIES, random_endgame  # noqa: E402
from app.chess.mcts import MCTS  # noqa: E402
from app.chess.rules import game_outcome  # noqa: E402
from app.training.checkpoints import load_checkpoint  # noqa: E402

PROBE_SEED = 900_000


def positions(family: str, count: int) -> list[chess.Board]:
    """``count`` positions of one family, found by scanning fixed seeds."""
    found, seed = [], PROBE_SEED
    names = {item.name for item in FAMILIES}
    assert family in names, family
    while len(found) < count:
        board, name = random_endgame(seed)
        seed += 1
        if name == family:
            found.append(board)
    return found


def strong_color(board: chess.Board) -> chess.Color:
    material = {color: sum(len(board.pieces(piece, color)) * value for piece, value in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9))) for color in (chess.WHITE, chess.BLACK)}
    return chess.WHITE if material[chess.WHITE] > material[chess.BLACK] else chess.BLACK


@torch.inference_mode()
def value_report(model: torch.nn.Module, count: int) -> dict[str, dict[str, float]]:
    report = {}
    for family in ("KQvK", "KRvK", "KQvKR"):
        to_move: dict[str, list[float]] = {"strong to move": [], "weak to move": []}
        for board in positions(family, count):
            strong = strong_color(board)
            _, value = model(torch.from_numpy(encode_board(board)).unsqueeze(0))
            sign = 1.0 if board.turn == strong else -1.0  # value is for the side to move
            to_move["strong to move" if board.turn == strong else "weak to move"].append(sign * float(value))
        report[family] = {key: float(np.mean(values)) for key, values in to_move.items() if values}
    return report


def conversion_report(model: torch.nn.Module, count: int, simulations: int) -> dict[str, dict[str, float]]:
    report = {}
    for family in ("KQvK", "KRvK"):
        mates, plies = 0, []
        for index, start in enumerate(positions(family, count)):
            board = start.copy()
            search = MCTS(model, torch.device("cpu"), simulations=simulations, draw_contempt=0.15, seed=index, dirichlet_epsilon=0.0)
            for _ in range(100):
                if game_outcome(board) is not None:
                    break
                _, _, root = search.search(board, add_noise=False)
                board.push(search.select_move(root, 0.0))
            outcome = game_outcome(board)
            if outcome is not None and outcome.termination == chess.Termination.CHECKMATE:
                mates += 1
                plies.append(len(board.move_stack))
        report[family] = {"mate_rate": mates / count, "avg_plies_to_mate": float(np.mean(plies)) if plies else float("nan")}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for path in args.checkpoints:
        model, _ = load_checkpoint(path, torch.device("cpu"))
        model.eval()
        print(f"== {path}")
        for family, values in value_report(model, args.positions).items():
            print(f"  value {family:6s} " + "  ".join(f"{key}: {value:+.2f}" for key, value in values.items()))
        for family, stats in conversion_report(model, args.positions, args.simulations).items():
            print(f"  convert {family:5s} mate {stats['mate_rate']:.0%}  avg plies {stats['avg_plies_to_mate']:.0f}")


if __name__ == "__main__":
    main()
