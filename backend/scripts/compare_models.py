"""Play two checkpoint files against each other (read-only).

Uses the same protocol as the promotion gate: paired randomized openings with
colours swapped, no Dirichlet noise, greedy moves, real draw rules.
Nothing is written to the database or the checkpoint directory.

    python scripts/compare_models.py ../checkpoints/model-v0002.pt ../checkpoints/model-v0001.pt \
        --games 40 --simulations 48 --parallel 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.arena import evaluate_checkpoint  # noqa: E402
from app.training.checkpoints import load_checkpoint  # noqa: E402


class _NoRepository:
    def save_evaluation(self, evaluation):  # pragma: no cover - never called with persist=False
        raise RuntimeError("compare_models is read-only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("opponent", type=Path)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--max-plies", type=int, default=320)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--draw-contempt", type=float, default=0.15)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    candidate, candidate_payload = load_checkpoint(args.candidate, device)
    opponent, opponent_payload = load_checkpoint(args.opponent, device)
    candidate.eval()
    opponent.eval()
    games = args.games + args.games % 2

    def progress(event: dict) -> None:
        if event.get("type") == "evaluation_complete":
            return

    result = evaluate_checkpoint(
        candidate, int(candidate_payload["version"]), device, _NoRepository(), "checkpoint", games,
        simulations=args.simulations, max_plies=args.max_plies, opponent_model=opponent,
        opponent_version_label=f"v{opponent_payload['version']}", seed=args.seed, opening_plies=args.opening_plies,
        persist=False, draw_contempt=args.draw_contempt, parallel=args.parallel, event_callback=progress,
    )
    if args.json:
        print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))
        return
    print(f"v{candidate_payload['version']} vs v{opponent_payload['version']}: {result['wins']}W {result['draws']}D {result['losses']}L"
          f" of {result['games']} · score {result['score']:.1%}"
          f" · Elo {result['elo_diff']:+.0f} [{result['elo_low']:+.0f}, {result['elo_high']:+.0f}]")
    print(f"repetition {result['repetition_rate']:.1%} · decisive {result['decisive_rate']:.1%} · terminations {result['termination_counts']}")


if __name__ == "__main__":
    main()
