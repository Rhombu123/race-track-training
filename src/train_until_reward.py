"""Resume-train + live-eval loop until an episode reward hits a target.

Usage:
  python -m src.train_until_reward --target 925 --train-steps 20000 --eval-episodes 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.cli_utils import strip_shell_comments

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/eval until a target episode reward")
    p.add_argument("--target", type=float, default=925.0)
    p.add_argument("--train-steps", type=int, default=20_000, help="Timesteps per train round")
    p.add_argument("--eval-episodes", type=int, default=5, help="Live episodes after each train")
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--delay", type=float, default=0.02, help="Render delay for live watch")
    p.add_argument("--max-rounds", type=int, default=50, help="Safety cap on train+eval rounds")
    p.add_argument(
        "--eval-only-first",
        action="store_true",
        help="Run one live eval before training (in case we already hit target)",
    )
    return p.parse_args(strip_shell_comments())


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _latest_eval_session() -> Path | None:
    sims = ROOT / "results" / "simulations"
    if not sims.is_dir():
        return None
    evals = [
        p
        for p in sims.iterdir()
        if p.is_dir() and ("(eval)" in p.name or p.name.endswith("_eval"))
    ]
    if not evals:
        return None
    return max(evals, key=lambda p: p.stat().st_mtime)


def _session_max_reward(session_dir: Path) -> tuple[float | None, dict]:
    sj = session_dir / "session.json"
    if not sj.is_file():
        return None, {}
    data = json.loads(sj.read_text(encoding="utf-8"))
    rewards = [float(e["reward"]) for e in data.get("episodes") or []]
    if not rewards:
        return data.get("mean_reward"), data
    return max(rewards), data


def _resolve_eval_model() -> str:
    """Prefer the just-trained final checkpoint; fall back to best."""
    final = ROOT / "models" / "final_model.zip"
    best = ROOT / "models" / "best" / "best_model.zip"
    if final.is_file():
        return "models/final_model.zip"
    if best.is_file():
        return "models/best/best_model.zip"
    raise FileNotFoundError("No models/final_model.zip or models/best/best_model.zip found")


def _maybe_promote_final(episode_max: float, global_best: float) -> None:
    """If this round's max beat our hunt best, keep final as the demo weights too."""
    final = ROOT / "models" / "final_model.zip"
    best = ROOT / "models" / "best" / "best_model.zip"
    if not final.is_file() or episode_max < global_best:
        return
    best.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(final, best)
    print(f"Promoted final_model → best_model (episode max {episode_max:.2f})", flush=True)


def live_eval(
    args: argparse.Namespace,
    py: str,
    *,
    model_path: str | None = None,
    seed: int = 42,
) -> tuple[float | None, Path | None]:
    model = model_path or _resolve_eval_model()
    print(f"Live-eval model: {model} (seed={seed})", flush=True)
    code = _run(
        [
            py,
            "-m",
            "src.watch",
            "--model",
            model,
            "--episodes",
            str(args.eval_episodes),
            "--delay",
            str(args.delay),
            "--seed",
            str(seed),
            "--record",
        ]
    )
    if code != 0:
        print(f"Eval failed with exit {code}", flush=True)
        return None, None
    session = _latest_eval_session()
    if session is None:
        return None, None
    best, data = _session_max_reward(session)
    mean = data.get("mean_reward")
    print(
        f"Eval session {session.name}: max={best} mean={mean} "
        f"(n={data.get('n_episodes')})",
        flush=True,
    )
    return best, session


def train_round(args: argparse.Namespace, py: str, round_i: int) -> int:
    return _run(
        [
            py,
            "-m",
            "src.train",
            "--config",
            "configs/ppo_carracing.yaml",
            "--resume",
            "auto",
            "--timesteps",
            str(args.train_steps),
            "--n-envs",
            str(args.n_envs),
            "--device",
            args.device,
        ]
    )


def main() -> None:
    args = parse_args()
    py = sys.executable
    log_path = ROOT / "results" / "train_until_reward_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"Target episode reward: {args.target}", flush=True)
    print(f"Train {args.train_steps}/round · live eval {args.eval_episodes} eps", flush=True)
    print("=" * 60, flush=True)

    global_best = float("-inf")
    winning_session: str | None = None

    if args.eval_only_first:
        mx, session = live_eval(args, py)
        if mx is not None:
            global_best = max(global_best, mx)
        if mx is not None and mx >= args.target:
            print(f"\nHIT TARGET {args.target} with max={mx} → {session}", flush=True)
            return

    for round_i in range(1, args.max_rounds + 1):
        print("\n" + "#" * 60, flush=True)
        print(f"# ROUND {round_i}/{args.max_rounds}  (best so far {global_best:.2f})", flush=True)
        print("#" * 60, flush=True)

        code = train_round(args, py, round_i)
        if code != 0:
            raise SystemExit(f"Training failed on round {round_i} with exit {code}")

        # Always score the freshly trained weights (not a stale best.zip).
        # Rotate seeds each round so we sample new tracks.
        seed = 42 + (round_i - 1) * max(args.eval_episodes, 1)
        mx, session = live_eval(
            args, py, model_path="models/final_model.zip", seed=seed
        )
        entry = {
            "round": round_i,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "max_reward": mx,
            "session": str(session) if session else None,
            "target": args.target,
            "model": "models/final_model.zip",
            "seed": seed,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if mx is not None:
            if mx > global_best:
                _maybe_promote_final(mx, global_best)
                global_best = mx
                winning_session = str(session)
            if mx >= args.target:
                _maybe_promote_final(mx, mx)
                print("\n" + "=" * 60, flush=True)
                print(f"SUCCESS: episode reward {mx:.2f} >= {args.target}", flush=True)
                print(f"Session: {session}", flush=True)
                print("=" * 60, flush=True)
                return

        print(
            f"Round {round_i} done. Need {args.target}; best this round={mx}; "
            f"global best={global_best:.2f}",
            flush=True,
        )

    print(
        f"\nStopped after {args.max_rounds} rounds without hitting {args.target}. "
        f"Best seen: {global_best:.2f} ({winning_session})",
        flush=True,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
