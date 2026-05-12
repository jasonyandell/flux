---
title: Todo
kind: todo
first_seen: workspace
last_updated: workspace
status: active
---

Active threads, scratchpad for now. Moving to GitHub Issues or [beads](https://github.com/...) once volume justifies it.

History lives in [[log|log.md]]. Theory-shaped questions live in [[questions/open|questions/open.md]].

## Open — AI / evolution

- **PPO policy commitment.** Training is running but policy entropy still sits at ~2.92 vs log(19)=2.94 (near uniform random) while `explained_variance ≈ 0.74`. The value head is learning; the policy hasn't started concentrating probability mass. Watch `action_entropy` dropping as the leading indicator. If it stays flat for many more iters, candidates to try: larger `--entropy-coef` ramp-down, longer rollouts, reward reshape (right now `mean_total_reward = 21.58` is a structural constant in symmetric self-play).

- **`mx.compile` on the PPO train step.** Perf subagent measured ~30% in isolation. Not landed yet — current ~5s/iter is acceptable. See [[decisions/ppo-gnn]] § Performance.

- **Older replays have no flow data.** Replays written before the flow-recording fix (everything in the greatest-hits.json from before the PPO-flows landing) render without arrows. Either re-record once we have a stronger policy or live with the visual gap for that historical batch.

- **v2 saturation run.** v2 caught v1 in ~6.5% of v1's generation budget but has only ~347 gens of compute. Open question: does v2 saturate at v1's ceiling, above it, or below it? See [[decisions/v2-vision]]. Decision point: keep v1 around or retire it once v2 saturates.

- **League / pool sampling for champions.** Discussed earlier in the MLX-loop thread but not built into `evolve_mlx.py`. AlphaZero-style mixing of past champions into the opponent pool. Three sub-questions still open: pure-random vs latest-N vs Elo-weighted; whether the pool is per-model (v1 vs v2) or shared; how aggressive seat interacts with a champion pool.

- **rtNEAT / proper NEAT.** Current evolution is [[topics/neuroevolution|tier 1]] (fixed topology, weights only). Tier 2 (structural mutations, innovation numbers, speciation) and tier 3 (rtNEAT continuous replacement) are open. No urgency — MLX evolution already shows wins.

- **AI that paths to enemy bases (not just weakest local).** From [[questions/open]] possible-directions list. BFS-based target prioritization would break the four-AI weakest-local-neighbor attractor without needing neural.

- **Apple Neural Engine for deployed champions (stunt).** MLX trains on Metal (GPU); ANE is a separate fixed-function block reached via Core ML, not MLX. Wrong fit for training — no gradients, no scatter-add, the network is too small for Core ML conversion overhead to pay back. Could pay off as a *deploy* step once we have a champion worth shipping: freeze it → Core ML → run human-vs-AI mode through `MLComputeUnits.cpuAndNeuralEngine` for sub-ms, low-power inference (think iPhone). Pure stunt value. Replay pipeline is now solid ([[decisions/replay-rendering]]); the gating item is "champion worth packaging," which v2 saturation will tell us.

## Done — landed

- **MLX evolution loop.** Shipped end-to-end. `python/flux/{mlx_step,mlx_batch,game_loop,evolve_mlx}.py`, `python/scripts/train.py`. See [[topics/neuroevolution]] Tier 5.
- **Python pipeline forks (browser bridge).** Resolved: Vite-static serves `public/champions/` and `public/replays/` directly. No Python HTTP server. Replays drive the browser; "record next tournament" is replaced by automatic replay write on each batch from `train.py`. See [[decisions/replay-rendering]].
- **v2 wider-vision experiment.** Started. See [[decisions/v2-vision]].
- **v3 GNN saturation thread.** Trained ~500 gens on radius 9 (multiple wandb runs, with and without aggressive). Weak — fitness stayed negative; aggressive kept winning. Architecture is fine; the bottleneck is credit assignment. See [[decisions/v3-gnn]] § Result and the forward plan in [[decisions/ppo-gnn]].
- **PPO + GNN implementation.** Shipped end-to-end. `python/flux/ppo.py` + `python/scripts/train_ppo.py`. Active wandb run is the source of truth. Iter time dropped from 22s → 5s via update_epochs cut + eval coalescing. Full instrumentation panel landed (clip_fraction, explained_variance, ratio_max, grad_norm, action_entropy, per-rollout outcomes, end-state image every 20 iters). See [[decisions/ppo-gnn]].
- **Greatest-hits replay cycle.** `python/scripts/build_greatest_hits.py` scans `.flxr` headers, filters fitness > 0 and ≥200 frames, sorts longest-first, writes `public/replays/greatest-hits.json`. Browser exposes a `greatestHits` tunable that swaps the player's index URL; auto-speed targets ~2s per replay for max-DPS phone viewing. See [[decisions/replay-rendering]] § greatest hits.
- **PPO replays carry flow data.** `train_ppo.py` now pulls game-0 `flow_src/dst/player/valid` and writes Flow objects into each recorded frame. The browser then renders directional arrows. Fixed in same session as the bug.

## Open — capture / video

- **Tournament-from-pool video pipeline (browser-only version).** Load multiple champion JSON files, assign to seats randomly, auto-record N consecutive games, save webms. ~1 day. Doesn't depend on Python — uses files from the existing save-champion button. Closes the "I want recordings of evolved AI wars" thread.

## Open — game model / bugs

- **Graph topology variants.** From [[questions/open]]: multiple parallel routes as a structural alternative to a single central choke. Untested — could change AI dynamics meaningfully but no concrete plan.

- **Further constant tuning.** `REGEN_PER_SEC` already bumped 1.0 → 1.1 for pacing. `TRANSFER_PER_SEC` and `ATTACK_BONUS` are untouched and probably fine; revisit only if game pacing shifts under heavier evolution.

## Process

- **Move to GH issues or [beads](beads://...) once we have >10 active items.** This file is a holding pen.
