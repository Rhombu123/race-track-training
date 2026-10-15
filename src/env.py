"""CarRacing environment factory and observation wrappers."""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def resolve_env_id(preferred: str = "CarRacing-v3") -> str:
    """Return a registered CarRacing env id, preferring v3 then v2."""
    candidates = [preferred, "CarRacing-v3", "CarRacing-v2"]
    seen: set[str] = set()
    for env_id in candidates:
        if env_id in seen:
            continue
        seen.add(env_id)
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(
        "Neither CarRacing-v3 nor CarRacing-v2 is registered. "
        "Install gymnasium[box2d] and ensure Box2D builds correctly."
    )


class ChannelFirstWrapper(gym.ObservationWrapper):
    """Squeeze stacked grayscale (stack, H, W, 1) -> (stack, H, W) for CnnPolicy.

    SB3 CnnPolicy expects channel-first images. After GrayScale(keep_dim=True)
    + Resize + FrameStack, observations are typically (stack, H, W, 1).
    Treating the stack dimension as channels yields (4, 84, 84).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        shape = obs_space.shape
        if len(shape) == 4 and shape[-1] == 1:
            new_shape = (shape[0], shape[1], shape[2])
        elif len(shape) == 3:
            new_shape = shape
        else:
            raise ValueError(f"Unexpected observation shape: {shape}")
        self.observation_space = spaces.Box(
            low=float(np.min(obs_space.low)),
            high=float(np.max(obs_space.high)),
            shape=new_shape,
            dtype=obs_space.dtype,
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = obs[..., 0]
        return obs


def make_env(
    env_id: Optional[str] = None,
    seed: int = 0,
    rank: int = 0,
    render_mode: Optional[str] = None,
    monitor: bool = True,
    frame_stack: int = 4,
    resize: int = 84,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a preprocessed CarRacing environment."""

    resolved_id = resolve_env_id(env_id or "CarRacing-v3")

    def _init() -> gym.Env:
        kwargs: dict = {"continuous": True}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(resolved_id, **kwargs)
        env = RecordEpisodeStatistics(env)
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (resize, resize))
        env = FrameStackObservation(env, stack_size=frame_stack)
        env = ChannelFirstWrapper(env)
        if monitor:
            env = Monitor(env)
        env.reset(seed=seed + rank)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + rank)
        return env

    return _init


def make_vec_env(
    env_id: Optional[str] = None,
    n_envs: int = 4,
    seed: int = 0,
    use_subproc: bool = True,
    frame_stack: int = 4,
    resize: int = 84,
) -> VecEnv:
    """Create a vectorized CarRacing env ready for SB3 CnnPolicy.

    On macOS, SubprocVecEnv + pygame/Box2D is brittle (SDL double-load /
    broken pipes). Default to DummyVecEnv there unless the caller forces
    subproc via ``use_subproc=True`` *and* sets env var
    ``CARRACING_FORCE_SUBPROC=1``.
    """
    env_fns = [
        make_env(
            env_id=env_id,
            seed=seed,
            rank=i,
            frame_stack=frame_stack,
            resize=resize,
        )
        for i in range(n_envs)
    ]
    force_subproc = os.environ.get("CARRACING_FORCE_SUBPROC") == "1"
    on_darwin = sys.platform == "darwin"
    if use_subproc and n_envs > 1 and (force_subproc or not on_darwin):
        try:
            # spawn avoids fork+pygame issues on some platforms
            return SubprocVecEnv(env_fns, start_method="spawn")
        except Exception as exc:
            print(f"SubprocVecEnv failed ({exc}); falling back to DummyVecEnv")
            return DummyVecEnv(env_fns)
    if use_subproc and n_envs > 1 and on_darwin and not force_subproc:
        print(
            "macOS: using DummyVecEnv (set CARRACING_FORCE_SUBPROC=1 to force SubprocVecEnv)"
        )
    return DummyVecEnv(env_fns)


def make_eval_env(
    env_id: Optional[str] = None,
    seed: int = 0,
    render_mode: Optional[str] = "rgb_array",
    frame_stack: int = 4,
    resize: int = 84,
) -> gym.Env:
    """Single non-vectorized env for evaluation / recording."""
    return make_env(
        env_id=env_id,
        seed=seed,
        rank=0,
        render_mode=render_mode,
        monitor=True,
        frame_stack=frame_stack,
        resize=resize,
    )()
