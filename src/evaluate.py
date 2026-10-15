"""Evaluate a trained PPO CarRacing agent (optional live window + docs)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.cli_utils import find_latest_model, strip_shell_comments
from src.env import make_eval_env, resolve_env_id
from src.simulation_log import SimulationSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO on CarRacing (add --render to watch live)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to SB3 model .zip (default: latest under models/)",
    )
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic actions"
    )
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--resize", type=int, default=84)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a live pygame window (render_mode=human)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["human", "rgb_array"],
        help="Override render mode (default: human if --render, else none)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps when rendering (e.g. 0.02)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Legacy single metrics JSON path (also written inside session dir)",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode GIFs + session.json under results/simulations/ (default: on)",
    )
    parser.add_argument("--fps", type=int, default=30, help="GIF frame rate when recording")
    return parser.parse_args(strip_shell_comments())


def _maybe_show_frame(frame: np.ndarray | None, screen, clock, fps: int) -> object:
    """Blit an rgb_array frame to a pygame window; returns (screen, clock)."""
    if frame is None:
        return screen, clock
    import pygame

    frame = np.asarray(frame)
    if frame.ndim != 3:
        return screen, clock
    h, w = frame.shape[:2]
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("CarRacing — evaluating (also recording)")
        clock = pygame.time.Clock()
    # pygame wants (W, H, 3) with axes swapped from numpy (H, W, 3)
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pass
    if clock is not None:
        clock.tick(fps)
    return screen, clock


def evaluate(
    model_path: str,
    env_id: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    frame_stack: int,
    resize: int,
    render_mode: str | None = None,
    delay: float = 0.0,
    record: bool = True,
    fps: int = 30,
    session: SimulationSession | None = None,
) -> dict:
    resolved = resolve_env_id(env_id)

    # Need rgb_array frames when documenting; show pygame ourselves if "human" wanted.
    want_human = render_mode == "human"
    env_render = "rgb_array" if (record or want_human) else render_mode

    env = make_eval_env(
        env_id=resolved,
        seed=seed,
        render_mode=env_render,
        frame_stack=frame_stack,
        resize=resize,
    )
    model = PPO.load(model_path)

    if session is None and record:
        session = SimulationSession(
            kind="eval",
            model_path=model_path,
            meta={
                "env_id": resolved,
                "episodes_requested": episodes,
                "deterministic": deterministic,
                "seed": seed,
            },
            record=True,
            fps=fps,
        )
    elif session is None:
        session = SimulationSession(
            kind="eval",
            model_path=model_path,
            meta={
                "env_id": resolved,
                "episodes_requested": episodes,
                "deterministic": deterministic,
                "seed": seed,
            },
            record=False,
            fps=fps,
        )

    rewards: list[float] = []
    lengths: list[int] = []
    screen = None
    clock = None

    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            frames: list[np.ndarray] = []

            frame = env.render() if env_render == "rgb_array" else None
            if frame is not None:
                frames.append(np.asarray(frame))
                if want_human:
                    screen, clock = _maybe_show_frame(frame, screen, clock, fps)

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, _info = env.step(action)
                if env_render == "rgb_array":
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                        if want_human:
                            screen, clock = _maybe_show_frame(frame, screen, clock, fps)
                ep_reward += float(reward)
                ep_len += 1
                if delay > 0:
                    time.sleep(delay)

            rewards.append(ep_reward)
            lengths.append(ep_len)
            rec = session.save_episode_recording(
                ep + 1,
                frames,
                reward=ep_reward,
                length=ep_len,
                seed=seed + ep,
            )
            rec_note = f" → {rec}" if rec else ""
            print(
                f"Episode {ep + 1}/{episodes}: reward={ep_reward:.2f} "
                f"length={ep_len}{rec_note}"
            )
    finally:
        env.close()
        if screen is not None:
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass

    metrics = {
        "model": model_path,
        "env_id": resolved,
        "episodes": episodes,
        "deterministic": deterministic,
        "render_mode": render_mode,
        "recorded": record,
        "session_dir": str(session.root),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "rewards": rewards,
    }
    session.finalize(summary={k: metrics[k] for k in (
        "mean_reward", "std_reward", "min_reward", "max_reward", "mean_length"
    )})
    return metrics


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path:
        latest = find_latest_model()
        if latest is None:
            raise SystemExit(
                "No model found under models/. Train first:\n"
                "  python -m src.train --config configs/ppo_carracing.yaml "
                "--timesteps 10000 --n-envs 2"
            )
        model_path = str(latest)
        print(f"Using model: {model_path}")

    deterministic = not args.stochastic
    if args.render_mode is not None:
        render_mode = args.render_mode
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None

    # Watching live: fewer episodes by default unless user overrode
    episodes = args.episodes
    if render_mode == "human" and args.episodes == 20:
        episodes = 3
        print("Live render: using 3 episodes (pass --episodes N to override)")

    metrics = evaluate(
        model_path=model_path,
        env_id=args.env_id,
        episodes=episodes,
        seed=args.seed,
        deterministic=deterministic,
        frame_stack=args.frame_stack,
        resize=args.resize,
        render_mode=render_mode,
        delay=args.delay,
        record=args.record,
        fps=args.fps,
    )

    out_path = Path(args.out) if args.out else Path(metrics["session_dir"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"\nMean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} "
        f"over {metrics['episodes']} episodes"
    )
    print(f"Wrote metrics to {out_path}")
    print(f"Session folder: {metrics['session_dir']}")


if __name__ == "__main__":
    main()
