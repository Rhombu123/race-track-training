"""Record a demo GIF/MP4 of a trained CarRacing PPO agent.

Also writes a documented session under results/simulations/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from src.cli_utils import find_latest_model, strip_shell_comments
from src.env import make_eval_env, resolve_env_id
from src.simulation_log import SimulationSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record CarRacing demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to SB3 model .zip (default: latest under models/)",
    )
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--resize", type=int, default=84)
    parser.add_argument(
        "--out",
        type=str,
        default="results/demo.gif",
        help="Convenience output path (.gif or .mp4); session copy also saved",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write results/simulations/Run N - … (record)/ (default: on)",
    )
    return parser.parse_args(strip_shell_comments())


def record(
    model_path: str,
    env_id: str,
    seed: int,
    max_steps: int,
    frame_stack: int,
    resize: int,
    out_path: str,
    fps: int,
    deterministic: bool,
    document: bool = True,
) -> Path:
    resolved = resolve_env_id(env_id)
    env = make_eval_env(
        env_id=resolved,
        seed=seed,
        render_mode="rgb_array",
        frame_stack=frame_stack,
        resize=resize,
    )
    model = PPO.load(model_path)

    session = None
    if document:
        session = SimulationSession(
            kind="record",
            model_path=model_path,
            meta={"env_id": resolved, "seed": seed, "max_steps": max_steps},
            record=True,
            fps=fps,
        )

    obs, _info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    ep_reward = 0.0

    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame))

    step = 0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, _info = env.step(action)
        ep_reward += float(reward)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if done or truncated:
            break

    env.close()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise RuntimeError("No frames captured; check render_mode support.")

    to_save = frames
    if out.suffix.lower() == ".gif" and len(to_save) > 300:
        to_save = to_save[::2]

    duration = 1.0 / fps
    if out.suffix.lower() == ".gif":
        imageio.mimsave(out, to_save, duration=duration, loop=0)
    else:
        imageio.mimsave(out, frames, fps=fps)

    print(f"Saved {len(frames)} frames to {out} ({step + 1} env steps)")

    if session is not None:
        session.save_episode_recording(
            1,
            frames,
            reward=ep_reward,
            length=step + 1,
            seed=seed,
            extra={"convenience_out": str(out)},
        )
        session.finalize(summary={"convenience_out": str(out)})

    return out


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path:
        latest = find_latest_model()
        if latest is None:
            raise SystemExit("No model under models/. Train first.")
        model_path = str(latest)
        print(f"Using model: {model_path}")

    record(
        model_path=model_path,
        env_id=args.env_id,
        seed=args.seed,
        max_steps=args.max_steps,
        frame_stack=args.frame_stack,
        resize=args.resize,
        out_path=args.out,
        fps=args.fps,
        deterministic=not args.stochastic,
        document=args.record,
    )


if __name__ == "__main__":
    main()
