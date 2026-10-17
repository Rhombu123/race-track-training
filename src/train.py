"""Train a PPO agent on CarRacing with Stable-Baselines3.

Training does NOT open a pygame window. After a run finishes (or if a
checkpoint already exists), watch the car with:

    python -m src.evaluate --model models/best/best_model.zip --render
    # or: python -m src.watch
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed

from src.cli_utils import find_latest_model, strip_shell_comments
from src.env import make_vec_env, resolve_env_id
from src.simulation_log import start_train_session


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on CarRacing (no live window — use "
            "`python -m src.evaluate --render` or `python -m src.watch` after)"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_carracing.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override number of parallel envs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint .zip to continue training, or 'auto' to "
            "load best/final/latest under models/. Without this, every run "
            "starts from a random policy (repeating --timesteps 10000 does "
            "NOT stack learning)."
        ),
    )
    return parser.parse_args(strip_shell_comments())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_random_seed(seed)

    env_id = resolve_env_id(cfg.get("env_id", "CarRacing-v3"))
    n_envs = int(args.n_envs or cfg.get("n_envs", 4))
    use_subproc = bool(cfg.get("use_subproc", True))
    frame_stack = int(cfg.get("frame_stack", 4))
    resize = int(cfg.get("resize", 84))
    total_timesteps = int(args.timesteps or cfg.get("total_timesteps", 1_000_000))
    device = args.device or cfg.get("device", "auto")

    log_dir = Path(cfg.get("log_dir", "logs"))
    model_dir = Path(cfg.get("model_dir", "models"))
    best_dir = model_dir / "best"
    ckpt_dir = model_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Env: {env_id} | n_envs={n_envs} | timesteps={total_timesteps} | device={device}")

    train_env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        use_subproc=use_subproc,
        frame_stack=frame_stack,
        resize=resize,
    )

    eval_env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed + 10_000,
        use_subproc=False,
        frame_stack=frame_stack,
        resize=resize,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=max(int(cfg.get("eval_freq", 10_000)) // n_envs, 1),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 5)),
        deterministic=True,
        render=False,
    )
    # Resume must not clobber a stronger best_model.zip: EvalCallback starts at
    # -inf and would overwrite on the first eval even if that score is worse.
    best_score_path = best_dir / "best_score.json"
    prior_best = float("-inf")
    if best_score_path.is_file():
        try:
            prior_best = float(json.loads(best_score_path.read_text(encoding="utf-8"))["best_mean_reward"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            prior_best = float("-inf")
    eval_npz = log_dir / "eval" / "evaluations.npz"
    if eval_npz.is_file():
        try:
            loaded = np.load(eval_npz)
            if "results" in loaded and len(loaded["results"]):
                prior_best = max(prior_best, float(np.mean(loaded["results"][-1])))
                # Also consider historical peak in this log
                prior_best = max(prior_best, float(np.max(np.mean(loaded["results"], axis=1))))
        except (OSError, ValueError):
            pass
    if prior_best > float("-inf"):
        eval_callback.best_mean_reward = prior_best
        print(f"Preserving EvalCallback best_mean_reward floor at {prior_best:.2f}")

    best_zip = best_dir / "best_model.zip"
    if best_zip.is_file():
        shutil.copy2(best_zip, best_dir / "best_model.protected.zip")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(cfg.get("save_freq", 25_000)) // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo_carracing",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    policy_kwargs = {}
    ppo_kwargs = dict(
        policy=cfg.get("policy", "CnnPolicy"),
        env=train_env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 1024)),
        batch_size=int(cfg.get("batch_size", 128)),
        n_epochs=int(cfg.get("n_epochs", 4)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir / "tb"),
        policy_kwargs=policy_kwargs,
    )

    resume_path: str | None = args.resume
    if resume_path is not None and resume_path.lower() == "auto":
        latest = find_latest_model(model_dir)
        if latest is None:
            print("No checkpoint under models/ for --resume auto; starting fresh.")
            resume_path = None
        else:
            resume_path = str(latest)

    if resume_path:
        print(f"Continuing training from {resume_path}")
        print(f"(adding ~{total_timesteps} more timesteps on top of that checkpoint)")
        model = PPO.load(resume_path, env=train_env, device=device)
        # Keep tensorboard logging on resume
        model.tensorboard_log = str(log_dir / "tb")
    else:
        if total_timesteps < 100_000:
            print(
                "\n*** NOTE: Starting a NEW random policy "
                f"for only {total_timesteps} steps. ***\n"
                "CarRacing usually needs hundreds of thousands of steps in ONE "
                "continuous run before driving looks better.\n"
                "Repeating short trains without --resume does not stack learning.\n"
                "  Continue previous weights:\n"
                "    python -m src.train --config configs/ppo_carracing.yaml "
                "--resume auto --timesteps 100000 --n-envs 2\n"
                "  Or one longer run:\n"
                "    python -m src.train --config configs/ppo_carracing.yaml "
                "--timesteps 250000 --n-envs 2\n"
            )
        model = PPO(**ppo_kwargs)

    # Persist run metadata for README / reproducibility + simulations log
    meta = {
        "env_id": env_id,
        "seed": seed,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "device": device,
        "config": cfg,
        "resume": resume_path,
        "log_dir": str(log_dir),
        "model_dir": str(model_dir),
    }
    (model_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_session = start_train_session(meta)
    print(f"Training session log → {train_session.root}/")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=cfg.get("tb_log_name", "ppo_carracing"),
        progress_bar=True,
        reset_num_timesteps=resume_path is None,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}.zip")
    print(f"Best model (by eval) under {best_dir}/")

    best_score_path.write_text(
        json.dumps(
            {
                "best_mean_reward": float(eval_callback.best_mean_reward),
                "num_timesteps": int(model.num_timesteps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_session.finalize(
        summary={
            "final_model": str(final_path) + ".zip",
            "best_model_dir": str(best_dir),
            "num_timesteps": int(model.num_timesteps),
            "best_mean_reward": float(eval_callback.best_mean_reward),
        }
    )
    print(
        "\nTraining does not show a window. Watch (and auto-record) with:\n"
        "  python -m src.watch\n"
        f"Session docs: {train_session.root}/"
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
