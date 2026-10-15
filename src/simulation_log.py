"""Timestamped simulation / training session logs for documentation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

SIMULATIONS_ROOT = Path("results/simulations")
INDEX_PATH = SIMULATIONS_ROOT / "index.jsonl"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _friendly_stamp(when: datetime | None = None) -> str:
    """Human-readable UTC timestamp for folder names."""
    when = when or datetime.now(timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _next_run_number() -> int:
    """Next Run N based on existing `Run N - ...` folders."""
    SIMULATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in SIMULATIONS_ROOT.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("Run "):
            continue
        try:
            num = int(path.name.split(" - ", 1)[0].removeprefix("Run ").strip())
        except ValueError:
            continue
        highest = max(highest, num)
    return highest + 1


def make_run_dir_name(kind: str, when: datetime | None = None, run_number: int | None = None) -> str:
    """Folder name like: `Run 12 - 2026-10-15 01:02:36 UTC (eval)`."""
    n = run_number if run_number is not None else _next_run_number()
    return f"Run {n} - {_friendly_stamp(when)} ({kind})"


def _append_index(entry: dict[str, Any]) -> None:
    SIMULATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class SimulationSession:
    """One documented run: metrics JSON + optional per-episode GIFs."""

    def __init__(
        self,
        kind: str,
        *,
        model_path: str | None = None,
        meta: dict[str, Any] | None = None,
        root: Path | None = None,
        record: bool = True,
        fps: int = 30,
    ):
        self.kind = kind
        self.record = record
        self.fps = fps
        started = datetime.now(timezone.utc)
        self.started_at = started.isoformat()
        display_name = make_run_dir_name(kind, when=started)
        self.root = root or (SIMULATIONS_ROOT / display_name)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {
            "kind": kind,
            "display_name": self.root.name,
            "started_at": self.started_at,
            "model": model_path,
            "session_dir": str(self.root),
            **(meta or {}),
        }
        self.episodes: list[dict[str, Any]] = []
        self._write_session()

    def _write_session(self) -> Path:
        payload = {
            **self.meta,
            "episodes": self.episodes,
            "n_episodes": len(self.episodes),
        }
        if self.episodes:
            rewards = [e["reward"] for e in self.episodes]
            payload["mean_reward"] = float(np.mean(rewards))
            payload["std_reward"] = float(np.std(rewards))
            payload["min_reward"] = float(np.min(rewards))
            payload["max_reward"] = float(np.max(rewards))
        path = self.root / "session.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_episode_recording(
        self,
        episode_index: int,
        frames: list[np.ndarray],
        *,
        reward: float,
        length: int,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save episode GIF (if recording) and append episode metrics."""
        recording_path: str | None = None
        if self.record and frames:
            out = self.root / f"episode_{episode_index:03d}.gif"
            to_save = frames
            if len(to_save) > 300:
                to_save = to_save[::2]
            imageio.mimsave(
                out,
                [np.asarray(f) for f in to_save],
                duration=1.0 / self.fps,
                loop=0,
            )
            recording_path = str(out)

        entry: dict[str, Any] = {
            "episode": episode_index,
            "reward": float(reward),
            "length": int(length),
            "seed": seed,
            "recording": recording_path,
            **(extra or {}),
        }
        self.episodes.append(entry)
        self._write_session()
        return Path(recording_path) if recording_path else None

    def finalize(self, summary: dict[str, Any] | None = None) -> Path:
        self.meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        if summary:
            self.meta.update(summary)
        path = self._write_session()
        _append_index(
            {
                "run": self.root.name,
                "kind": self.kind,
                "session_dir": str(self.root),
                "started_at": self.started_at,
                "finished_at": self.meta["finished_at"],
                "model": self.meta.get("model"),
                "n_episodes": len(self.episodes),
                "mean_reward": self.meta.get("mean_reward"),
            }
        )
        readme = self.root / "README.md"
        lines = [
            f"# {self.root.name}",
            "",
            f"- Kind: `{self.kind}`",
            f"- Started: `{self.started_at}`",
            f"- Finished: `{self.meta['finished_at']}`",
            f"- Model: `{self.meta.get('model')}`",
            f"- Episodes: {len(self.episodes)}",
        ]
        if self.episodes:
            lines.append(
                f"- Mean reward: {self.meta.get('mean_reward'):.2f} "
                f"± {self.meta.get('std_reward'):.2f}"
            )
        lines.extend(["", "## Files", "", "- `session.json` — full metrics"])
        if self.record:
            lines.append("- `episode_XXX.gif` — per-episode recordings")
        lines.append("")
        readme.write_text("\n".join(lines), encoding="utf-8")
        print(f"Documented simulation → {self.root}/")
        return path


def start_train_session(meta: dict[str, Any]) -> SimulationSession:
    """Create a training documentation folder (no episode GIFs)."""
    return SimulationSession(
        kind="train",
        model_path=meta.get("resume"),
        meta=meta,
        record=False,
    )
