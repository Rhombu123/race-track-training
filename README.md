# Race Track Training (CarRacing) — PPO

Train a **PPO** agent to drive in **Gymnasium CarRacing** from pixel observations, then **watch it live** in a pygame window.

**Staged campaign** (30 × 10k = 300k steps, with per-stage console logs + episode GIFs):

```bash
python -m src.staged_train --stages 30 --timesteps 10000 --n-envs 2 --device cpu
```

> **Env note:** Uses Gymnasium `CarRacing-v3` (falls back to `v2`) — the maintained equivalent of OpenAI Gym `CarRacing-v0`.

## Two-step workflow (this is the main path)

1. **Train** — logs only, **no pygame window**
2. **Watch** — `evaluate --render` or `watch` opens the live car window

```bash
cd "/Users/abdal-1/Desktop/CursorTemp/Lunar Lander AI"
source .venv/bin/activate
```

### Step 1 — Train (no window)

**Important:** each `train` without `--resume` starts a **brand-new** random policy. Running `--timesteps 10000` five times is still only ~10k of learning each time — it does **not** add up to 50k. Use one longer run, or continue with `--resume auto`.

Smoke check (pipeline only — the car will still look bad):

```bash
python -m src.train --config configs/ppo_carracing.yaml --timesteps 10000 --n-envs 2
```

Continue from the last checkpoint (this is how short runs stack):

```bash
python -m src.train --config configs/ppo_carracing.yaml --resume auto --timesteps 100000 --n-envs 2
```

Longer single run (practical — expect this to take a while):

```bash
python -m src.train --config configs/ppo_carracing.yaml --timesteps 250000 --n-envs 2 --device cpu
```

Full default (1M steps from YAML):

```bash
python -m src.train --config configs/ppo_carracing.yaml
```

### Step 2 — Watch the car live (pygame window)

`watch` / `evaluate` **log and record each simulation by default** under `results/simulations/Run N - YYYY-MM-DD HH:MM:SS UTC (kind)/` (`session.json` + `episode_XXX.gif`). Use `--no-record` to skip GIFs.

```bash
python -m src.watch
# or:
python -m src.evaluate --model models/best/best_model.zip --render
```

Slower playback / more episodes:

```bash
python -m src.evaluate --model models/best/best_model.zip --render --delay 0.02 --episodes 5
python -m src.watch --model models/final_model.zip --episodes 5
```

See [`results/README.md`](results/README.md) for the documentation layout.

---

## Setup (first time)

```bash
cd "/Users/abdal-1/Desktop/CursorTemp/Lunar Lander AI"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Score without a window

```bash
python -m src.evaluate --model models/best/best_model.zip --episodes 20
```

## Record a demo GIF

```bash
python -m src.record_demo --model models/best/best_model.zip --out results/demo.gif
```

## TensorBoard

```bash
tensorboard --logdir logs/tb
```

## Resume training

```bash
python -m src.train --config configs/ppo_carracing.yaml \
  --resume models/final_model.zip --timesteps 500000
```

---

## macOS / pygame / Box2D gotchas

- **Copy-paste:** do not append shell comments like `# quick smoke` on the same line if your shell passes them to Python (this project strips a trailing `# ...`, but bare commands are safer).
- **Train ≠ window:** training never opens a live view; always use `evaluate --render` or `watch` afterward.
- **SDL dual-load warnings:** you may see `Class SDL_* is implemented in both .../cv2/... and .../pygame/...`. Noisy but usually OK. Keep `opencv-python-headless` (not full `opencv-python`). Prefer one OpenCV package only.
- **SubprocVecEnv on Mac:** CarRacing + pygame in subprocesses can hit broken pipes. This repo uses **DummyVecEnv on macOS** by default. To force subprocesses: `CARRACING_FORCE_SUBPROC=1`.
- **Device:** if MPS/`auto` misbehaves, pass `--device cpu`.
- **Box2D:** `gymnasium[box2d]` wheels usually install fine on macOS. On Linux build failures: `sudo apt-get install -y swig build-essential` then reinstall requirements.

## Project layout

```
configs/ppo_carracing.yaml    # hyperparameters
src/env.py                    # grayscale, resize 84, frame stack 4, VecEnv
src/train.py                  # SB3 PPO + TensorBoard + best-model callback
src/evaluate.py               # score + optional --render (live watch)
src/watch.py                  # load latest model + human render
src/record_demo.py            # export GIF/MP4
notebooks/ppo_explained.ipynb # PPO math ↔ this codebase
models/                       # your checkpoints (gitignored)
logs/                         # TensorBoard / eval (gitignored)
results/                      # put demo.gif here after you record it
```

## Artifacts after you train

| Path | Meaning |
|------|---------|
| `models/best/best_model.zip` | Best mean eval reward |
| `models/final_model.zip` | End-of-run checkpoint |
| `models/checkpoints/` | Periodic saves |
| `logs/tb/` | TensorBoard |
| `results/simulations/Run N - …/` | Per-run docs: `session.json` + episode GIFs |
| `results/simulations/index.jsonl` | Catalog of all simulations |
| `results/demo.gif` | Convenience clip from `record_demo.py` |

## Expected training time

| Run | Timesteps | Purpose |
|-----|-----------|---------|
| Smoke | 10k | Verify pipeline |
| Practical | ~250k | First watchable policy |
| Target (config default) | 1M–2M | Stronger on-track driving |

On laptop CPU, 1M steps can take several hours. Prefer GPU/`cuda` when available (`--device cuda` or `device: auto` in YAML).

**Success narrative:** higher (positive) mean reward ≈ more track progress / staying on asphalt. Early / random policies often score negative.

## Observation pipeline

RGB → **grayscale** → **resize 84×84** → **stack 4 frames** → `(4, 84, 84)` → SB3 `CnnPolicy`.

Frame stacking approximates velocity so the pixel MDP is closer to Markov.

## Interview notes

- **Clipped surrogate:** PPO limits the policy likelihood ratio with `clip_range=0.2` so one bad update cannot destroy the policy.
- **GAE:** advantages blend multi-step TD errors (`gae_lambda=0.95`) to trade bias vs variance.
- **Actor–critic:** CNN features → Gaussian policy + value head; on-policy rollouts, then several epochs of minibatch updates.
- **Engineering:** VecEnv, `EvalCallback` for best model, TensorBoard, fixed seed in YAML.

Deep dive: [`notebooks/ppo_explained.ipynb`](notebooks/ppo_explained.ipynb).

## License

MIT-style use for portfolio / interview demos. Gymnasium / SB3 retain their own licenses.
