"""Open a live window, watch the PPO drive, and document the session.

Every watch run (by default) writes under results/simulations/:

    session.json          — metrics
    episode_001.gif       — recording
    README.md             — short summary

    python -m src.watch
    python -m src.watch --model models/final_model.zip --episodes 5
    python -m src.watch --no-record   # live only, no GIFs
"""

from __future__ import annotations

import argparse
import sys

from src.cli_utils import find_latest_model, strip_shell_comments
from src.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch CarRacing live and log/record the simulation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model .zip (default: best → final → newest checkpoint)",
    )
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--resize", type=int, default=84)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps (e.g. 0.02 for slower playback)",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode GIFs + session.json (default: on)",
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(strip_shell_comments())


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path:
        latest = find_latest_model()
        if latest is None:
            print(
                "No checkpoint under models/ yet. Train a smoke run first:\n"
                "  python -m src.train --config configs/ppo_carracing.yaml "
                "--timesteps 10000 --n-envs 2\n"
                "Then re-run: python -m src.watch",
                file=sys.stderr,
            )
            raise SystemExit(1)
        model_path = str(latest)

    print(f"Watching: {model_path} (pygame window should open)")
    if args.record:
        print("Recording + logging this simulation under results/simulations/")

    metrics = evaluate(
        model_path=model_path,
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        frame_stack=args.frame_stack,
        resize=args.resize,
        render_mode="human",
        delay=args.delay,
        record=args.record,
        fps=args.fps,
    )
    print(
        f"\nMean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} "
        f"over {metrics['episodes']} episodes"
    )
    print(f"Session folder: {metrics['session_dir']}")


if __name__ == "__main__":
    main()
