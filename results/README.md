# Results & simulation documentation

Every **train**, **watch**, **evaluate**, and **record_demo** run creates a folder under:

```text
results/simulations/Run N - YYYY-MM-DD HH:MM:SS UTC (kind)/
```

Example: `Run 2 - 2026-10-15 01:02:36 UTC (eval)`

| File | Contents |
|------|----------|
| `session.json` | Full metrics (model path, rewards, seeds, timestamps) |
| `episode_XXX.gif` | Per-episode recording (watch / evaluate / record) |
| `README.md` | Short human-readable summary |

A running catalog is appended to `results/simulations/index.jsonl`.

Campaign write-up: [`CAMPAIGN_REPORT.md`](CAMPAIGN_REPORT.md) (also under `staged_campaigns/.../`).

## Typical commands

```bash
# Train → logs a train session (config + model paths)
python -m src.train --config configs/ppo_carracing.yaml --resume auto --timesteps 100000 --n-envs 2

# Watch live AND document (GIFs + session.json) — default
python -m src.watch

# Evaluate with docs (no window)
python -m src.evaluate --model models/best/best_model.zip --episodes 5

# Live only, skip GIFs
python -m src.watch --no-record

# Showcase clip + session folder
python -m src.record_demo --model models/best/best_model.zip --out results/demo.gif
```
