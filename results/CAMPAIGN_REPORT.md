# Campaign Report — 30 × 10K Staged PPO (CarRacing)

**Campaign:** `results/staged_campaigns/20261015T011504Z/`  
**Window:** 2026-10-15T01:15:04Z → 02:15:55Z (~61 minutes)  
**Plan:** 30 stages × 10,000 timesteps = 300,000 total · CPU · 2 envs · 3 recorded eval episodes per stage  

**Artifacts:** 30/30 stages `ok` · 30 train console logs · 30 eval console logs · 90 episode GIFs · `models/final_model.zip` + `models/best/`

---

## First-person report

I ran the full staged campaign: thirty back-to-back 10K-step trials, each one resumed from the previous checkpoint, and after every stage I logged the console output and recorded three evaluation episodes to GIF.

### What went well

I got exactly what I asked the pipeline to do. Every stage finished cleanly — train exit 0, eval exit 0 — with a `console_train.log`, a `console_eval.log`, and a copied simulation session under `stage_XX/episodes/`. That gave me a complete paper trail from a random policy at stage 1 to a 300K-step agent at stage 30.

Learning was real, not just noise. Eval mean reward started around **−48** (stage 1) and **−77** (stage 2), flipped clearly positive by stage 3 (**~187**), and kept climbing through the middle of the campaign. The peak came at stage 21 with about **743** mean reward. Training rollouts told the same story more smoothly: end-of-stage `ep_rew_mean` went from about **−49** (stage 1) to **~52** (stage 10) to **~437** (stage 20) to **~528** (stage 30). So by the time I finished, the car was no longer a random wanderer — it was putting in real laps and collecting track reward.

Resuming with `--resume auto` after stage 1 worked the way it should. Short 10K slices stacked into one continuous 300K run instead of thirty disconnected restarts. On a laptop CPU with two DummyVecEnv workers, the whole campaign still finished in about an hour, which is usable for iterative experiments.

### What went badly

Evaluation was noisy and sometimes lied about progress. I only recorded three episodes per stage, and CarRacing variance is brutal — stage 8 hit **~369**, then stages 9–10 dropped back near **~100**, then stage 11 jumped to **~611**. That zigzag made it hard to trust any single stage’s number without looking at the trend.

The late campaign was the real problem. After the stage-21 peak (**~743**), performance fell hard: stage 23 **~301**, 24 **~122**, 25 **~176**, and stage 26 almost collapsed to **~5**. Stages 29–30 recovered somewhat (**~547** / **~294**), but I finished far below my best. In other words, the final checkpoint is not my best driver — and if I only watched stage 30’s GIFs, I’d undersell what this run actually learned.

Early stages also looked worse than “just starting”: stage 2’s eval (**−77**) was worse than stage 1. That’s expected with a tiny eval budget, but it still made the first documentation block look like a regression when the policy was barely learning yet.

I also burned time before this successful run: background `nohup` launches died mid-stage without a clear traceback, so I had to switch to a managed foreground/tee process to keep the campaign alive.

### What I would improve

1. **Ship the best model, not only the final one.** Peak eval was ~743 at stage 21; final was ~294. For demos and reports I should evaluate and promote `models/best/best_model.zip` (or per-stage best) instead of assuming the last resume is the hero.

2. **Stabilize the score.** Bump eval episodes from 3 to at least 10–20 when deciding whether a stage “improved,” and keep 3 GIFs for video. That would cut the false cliffs and celebrations.

3. **Fight late instability.** Something degraded after ~210K steps — too-aggressive updates, entropy collapse, or overfitting to recent tracks. I’d try a lower late-stage learning rate, stronger KL/entropy monitoring, or shorter resume chunks with an explicit “keep best weights if eval drops.”

4. **Make logging safer for long runs.** Prefer a supervised process (or a watchdog that restarts from `--start-stage`) over fragile detached shells, and write a rolling `progress.json` chart so I can spot collapse without grepping thirty logs.

5. **Speed / capacity.** Two CPU envs got me through 300K, but more parallel envs or a GPU would let me push toward the 1M+ regime where CarRacing usually looks reliably “good,” and where late-stage dips might average out.

### Bottom line

I successfully completed all 30 × 10K trials with full console logs and recordings. The agent clearly learned to drive and reached a strong mid-campaign peak, but the run ended unstable and below that peak. My next move is to demo from the best checkpoint, widen eval for cleaner stage decisions, and tighten the late-training schedule so the last mile doesn’t throw away the best lap.

---

## Stage eval mean rewards (3 episodes each)

| Stage | Mean reward | Stage | Mean reward | Stage | Mean reward |
|------:|------------:|------:|------------:|------:|------------:|
| 1 | −48.07 | 11 | 610.59 | 21 | **742.74** |
| 2 | −77.37 | 12 | 348.60 | 22 | 623.43 |
| 3 | 187.32 | 13 | 364.57 | 23 | 301.24 |
| 4 | 49.18 | 14 | 448.60 | 24 | 122.09 |
| 5 | 188.21 | 15 | 375.07 | 25 | 176.37 |
| 6 | 79.99 | 16 | 617.69 | 26 | 4.88 |
| 7 | 91.12 | 17 | 665.07 | 27 | 103.69 |
| 8 | 369.19 | 18 | 453.68 | 28 | 138.51 |
| 9 | 102.07 | 19 | 451.17 | 29 | 546.81 |
| 10 | 105.08 | 20 | 483.43 | 30 | 293.81 |

Overall mean across stages: **297.29** · First-5 mean: **59.85** · Last-5 mean: **217.54**
