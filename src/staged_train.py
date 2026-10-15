"""Staged PPO training with per-stage console logs + episode recordings.

Default campaign: 30 stages × 10_000 timesteps = 300_000 total.
Each stage:
  1. Train (fresh on stage 1, --resume auto afterward)
  2. Tee full console output to stage_XX/console.log
  3. Evaluate/record episodes → stage_XX/episodes/ (+ session.json)

Usage:
  python -m src.staged_train
  python -m src.staged_train --stages 30 --timesteps 10000 --n-envs 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.cli_utils import strip_shell_comments

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged CarRacing PPO training campaign")
    p.add_argument("--stages", type=int, default=30, help="Number of stages (default 30 → 300k)")
    p.add_argument("--timesteps", type=int, default=10_000, help="Timesteps per stage")
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--config", type=str, default="configs/ppo_carracing.yaml")
    p.add_argument("--eval-episodes", type=int, default=3, help="Episodes to record after each stage")
    p.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wipe models/ and logs/ before stage 1 (default: on)",
    )
    p.add_argument(
        "--campaign-dir",
        type=str,
        default=None,
        help="Output root (default: results/staged_campaigns/<timestamp>)",
    )
    p.add_argument("--start-stage", type=int, default=1, help="1-based stage to start from")
    return p.parse_args(strip_shell_comments())


def _run_logged(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    print(f">>> console → {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"started: {datetime.now(timezone.utc).isoformat()}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f"\nfinished: {datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"exit_code: {code}\n")
    return code


def reset_training(cwd: Path) -> None:
    for name in ("models", "logs"):
        path = cwd / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        if name == "models":
            (path / "best").mkdir(exist_ok=True)
            (path / "checkpoints").mkdir(exist_ok=True)
    print("Reset models/ and logs/", flush=True)


def main() -> None:
    args = parse_args()
    cwd = ROOT
    total_steps = args.stages * args.timesteps

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = (
        Path(args.campaign_dir)
        if args.campaign_dir
        else cwd / "results" / "staged_campaigns" / stamp
    )
    campaign.mkdir(parents=True, exist_ok=True)

    manifesto = {
        "project": "Race Track Training",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": args.stages,
        "timesteps_per_stage": args.timesteps,
        "target_total_timesteps": total_steps,
        "n_envs": args.n_envs,
        "device": args.device,
        "eval_episodes": args.eval_episodes,
        "config": args.config,
        "start_stage": args.start_stage,
        "reset": args.reset and args.start_stage == 1,
    }
    (campaign / "campaign.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    (campaign / "README.md").write_text(
        "\n".join(
            [
                "# Staged Race Track Training Campaign",
                "",
                f"- Stages: **{args.stages}** × **{args.timesteps}** steps = **{total_steps}** total",
                f"- Device: `{args.device}` | envs: `{args.n_envs}`",
                f"- After each stage: `{args.eval_episodes}` recorded episodes + console logs",
                "",
                "Per stage folder: `stage_XX/` with `console_train.log`, `console_eval.log`, `summary.json`, `episodes/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60, flush=True)
    print("Race Track Training — staged campaign", flush=True)
    print(f"Campaign dir: {campaign}", flush=True)
    print(f"Plan: {args.stages} × {args.timesteps} = {total_steps} timesteps", flush=True)
    print("=" * 60, flush=True)

    if args.reset and args.start_stage == 1:
        reset_training(cwd)

    py = sys.executable
    stage_summaries: list[dict] = []

    for stage in range(args.start_stage, args.stages + 1):
        stage_dir = campaign / f"stage_{stage:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        cumulative = stage * args.timesteps
        print("\n" + "#" * 60, flush=True)
        print(f"# STAGE {stage}/{args.stages}  (+{args.timesteps} → ~{cumulative} total)", flush=True)
        print("#" * 60, flush=True)

        train_cmd = [
            py,
            "-m",
            "src.train",
            "--config",
            args.config,
            "--timesteps",
            str(args.timesteps),
            "--n-envs",
            str(args.n_envs),
            "--device",
            args.device,
        ]
        if stage > 1:
            train_cmd.extend(["--resume", "auto"])

        train_log = stage_dir / "console_train.log"
        code = _run_logged(train_cmd, train_log, cwd)
        if code != 0:
            summary = {
                "stage": stage,
                "status": "train_failed",
                "exit_code": code,
                "train_log": str(train_log),
            }
            (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            stage_summaries.append(summary)
            (campaign / "progress.json").write_text(
                json.dumps({"completed_stages": stage_summaries, "failed_at": stage}, indent=2),
                encoding="utf-8",
            )
            raise SystemExit(f"Stage {stage} training failed with exit {code}")

        episodes_dir = stage_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)
        eval_cmd = [
            py,
            "-m",
            "src.evaluate",
            "--model",
            "models/final_model.zip",
            "--episodes",
            str(args.eval_episodes),
            "--record",
            "--fps",
            "30",
            "--out",
            str(episodes_dir / "metrics.json"),
        ]
        # Prefer final_model; fall back handled inside evaluate if missing — use auto via omitting? 
        # Explicit final is what we just saved.
        if not (cwd / "models" / "final_model.zip").is_file():
            best = cwd / "models" / "best" / "best_model.zip"
            if best.is_file():
                eval_cmd[eval_cmd.index("--model") + 1] = str(best)

        eval_log = stage_dir / "console_eval.log"
        eval_code = _run_logged(eval_cmd, eval_log, cwd)

        # Copy newest simulation session into stage episodes folder for documentation
        sims = cwd / "results" / "simulations"
        copied = None
        if sims.is_dir():
            eval_sessions = sorted(
                [
                    p
                    for p in sims.iterdir()
                    if p.is_dir()
                    and (
                        p.name.endswith("_eval")
                        or p.name.endswith("(eval)")
                        or " (eval)" in p.name
                    )
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if eval_sessions:
                dest = episodes_dir / "simulation_session"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(eval_sessions[0], dest)
                copied = str(dest)

        summary = {
            "stage": stage,
            "status": "ok" if eval_code == 0 else "eval_failed",
            "cumulative_timesteps_target": cumulative,
            "train_exit": code,
            "eval_exit": eval_code,
            "train_log": str(train_log),
            "eval_log": str(eval_log),
            "episodes_copy": copied,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        # Pull mean reward from metrics if present
        metrics_path = episodes_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                summary["mean_reward"] = metrics.get("mean_reward")
                summary["session_dir"] = metrics.get("session_dir")
            except json.JSONDecodeError:
                pass

        (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        stage_summaries.append(summary)
        (campaign / "progress.json").write_text(
            json.dumps(
                {
                    "completed_stages": len(stage_summaries),
                    "target_stages": args.stages,
                    "stages": stage_summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Stage {stage}/{args.stages} done. "
            f"mean_reward={summary.get('mean_reward')} → {stage_dir}",
            flush=True,
        )

    manifesto["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifesto["status"] = "completed"
    (campaign / "campaign.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    print("\n" + "=" * 60, flush=True)
    print(f"Campaign complete: {args.stages} stages, ~{total_steps} timesteps", flush=True)
    print(f"Docs: {campaign}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
