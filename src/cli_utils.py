"""Shared CLI helpers for train / evaluate / watch."""

from __future__ import annotations

import sys
from pathlib import Path


def strip_shell_comments(argv: list[str] | None = None) -> list[str]:
    """Drop a trailing ``# ...`` fragment from argv (common paste mistake).

    Interactive zsh usually treats ``#`` as a comment; some shells / paste
    paths pass ``# quick smoke`` through to argparse and fail with
    ``unrecognized arguments``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    cleaned: list[str] = []
    for arg in args:
        if arg.startswith("#"):
            break
        if " #" in arg:
            # e.g. pasted as a single token somehow
            cleaned.append(arg.split(" #", 1)[0].rstrip())
            break
        cleaned.append(arg)
    return cleaned


def find_latest_model(models_dir: str | Path = "models") -> Path | None:
    """Prefer best → final → newest checkpoint under ``models/``."""
    root = Path(models_dir)
    candidates = [
        root / "best" / "best_model.zip",
        root / "final_model.zip",
    ]
    for path in candidates:
        if path.is_file():
            return path

    ckpt_dir = root / "checkpoints"
    if ckpt_dir.is_dir():
        zips = sorted(
            ckpt_dir.glob("*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if zips:
            return zips[0]
    return None
