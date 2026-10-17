"""Search for a high episode reward, then live-replay the winning seed.

Fast headless eval sweeps seeds; when target is hit, opens a pygame watch
for that exact seed so you can see the lap.

  python -m src.hunt_reward --target 925 --batch 20 --max-batches 100
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
    p = argparse.ArgumentParser(description="Hunt a target episode reward")
    p.add_argument("--target", type=float, default=925.0)
    p.add_argument("--model", type=str, default="models/best/best_model.zip")
    p.add_argument("--batch", type=int, default=20, help="Episodes per headless batch")
    p.add_argument("--max-batches", type=int, default=80)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.02, help="Delay for winning live replay")
    p.add_argument(
        "--live-every",
        type=int,
        default=5,
        help="Also show a short live batch every N headless batches (0=never)",
    )
    p.add_argument("--live-episodes", type=int, default=3)
    return p.parse_args(strip_shell_comments())


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _latest_eval() -> Path | None:
    sims = ROOT / "results" / "simulations"
    evals = [p for p in sims.iterdir() if p.is_dir() and "(eval)" in p.name]
    return max(evals, key=lambda p: p.stat().st_mtime) if evals else None


def headless_batch(py: str, model: str, episodes: int, seed: int) -> dict:
    out = ROOT / "results" / "_hunt_metrics.json"
    code = _run(
        [
            py,
            "-m",
            "src.evaluate",
            "--model",
            model,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--no-record",
            "--out",
            str(out),
        ]
    )
    if code != 0:
        raise SystemExit(f"Headless eval failed ({code})")
    session = _latest_eval()
    data = json.loads((session / "session.json").read_text(encoding="utf-8"))
    rewards = [(e["episode"], e["reward"], e.get("seed")) for e in data.get("episodes") or []]
    best = max(rewards, key=lambda t: t[1]) if rewards else None
    return {
        "session": str(session),
        "mean": data.get("mean_reward"),
        "best": best,
        "rewards": rewards,
    }


def live_replay(py: str, model: str, seed: int, episodes: int, delay: float) -> Path | None:
    code = _run(
        [
            py,
            "-m",
            "src.watch",
            "--model",
            model,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--delay",
            str(delay),
            "--record",
        ]
    )
    if code != 0:
        print(f"Live replay failed ({code})", flush=True)
        return None
    return _latest_eval()


def main() -> None:
    args = parse_args()
    py = sys.executable
    log = ROOT / "results" / "hunt_reward_log.jsonl"
    global_best = float("-inf")
    winner = None

    print("=" * 60, flush=True)
    print(f"Hunting episode reward >= {args.target}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print("=" * 60, flush=True)

    seed = args.start_seed
    for batch_i in range(1, args.max_batches + 1):
        print(
            f"\n# HEADLESS BATCH {batch_i}/{args.max_batches} "
            f"seeds {seed}..{seed + args.batch - 1}  (best={global_best:.2f})",
            flush=True,
        )
        result = headless_batch(py, args.model, args.batch, seed)
        ep_i, mx, win_seed = result["best"]
        print(
            f"Batch max={mx:.2f} (ep {ep_i}, seed={win_seed}) mean={result['mean']:.2f} "
            f"→ {result['session']}",
            flush=True,
        )
        entry = {
            "batch": batch_i,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "start_seed": seed,
            "max_reward": mx,
            "winning_seed": win_seed,
            "mean_reward": result["mean"],
            "session": result["session"],
            "target": args.target,
        }
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if mx > global_best:
            global_best = mx
            winner = entry

        if mx >= args.target:
            print("\n" + "=" * 60, flush=True)
            print(f"SUCCESS headless: reward {mx:.2f} on seed {win_seed}", flush=True)
            print("Replaying live so you can watch it…", flush=True)
            print("=" * 60, flush=True)
            live_replay(py, args.model, int(win_seed), episodes=1, delay=args.delay)
            return

        if args.live_every and batch_i % args.live_every == 0:
            print(f"\n(Live demo of current best-ish seeds starting at {seed})…", flush=True)
            live_replay(
                py,
                args.model,
                seed,
                episodes=args.live_episodes,
                delay=args.delay,
            )

        seed += args.batch

    print(
        f"\nGave up after {args.max_batches} batches. Best={global_best:.2f} ({winner})",
        flush=True,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
