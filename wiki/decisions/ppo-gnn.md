---
title: PPO + GNN Policy
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Choice

Replace the NEAT-style evolutionary loop with **PPO** (Proximal Policy Optimization) while keeping v3's **2-layer GCN policy** architecture. Add a value head for advantage estimation. Train via MLX autograd on Apple Silicon.

The GNN forward pass, the step kernel, replays, the aggressive seat, and wandb logging shape are reused as-is. The learning algorithm is what changes.

## Why

The pattern across three architectures said architecture wasn't the bottleneck:

- v1 (2-hop MLP, 3571 weights): plateaued at fitness ~1540 after 5000+ generations.
- v2 (3-hop MLP, 6451 weights): caught v1 in ~100 generations, plateaued at the same level.
- v3 (2-layer GNN, 2995 weights, radius 9): sat at fitness ~-2168 after ~500 generations. See [[v3-gnn]] § Result.

NEAT gets **one scalar per game** to update the whole genome. PPO gets **one gradient per decision**, with a value baseline doing the credit assignment. For a network whose payoff is "learn specific multi-hop coordination patterns," the latter is what matters.

## Architecture as built

- **Policy head:** 2 GCN message-passing layers, hidden=32, per-cell 4-channel input (`strength_norm`, `is_mine`, `is_enemy`, `is_neutral`). 19 action logits per cell.
- **Value head:** mean-pool the second MP layer's activations over each seat's owned cells → small MLP (hidden=16) → scalar `V(state, seat)`. Per-seat because each of the 12 FFA seats sees a different perspective.
- **Parameter sharing:** one network applied per-seat. Input channels are already seat-relative so the network is seat-agnostic. One set of weights serves all 12 seats.

Constants live in `python/flux/ppo.py`: `NEIGHBOR_STRIDE=18`, `IN_DIM=4`, `HIDDEN=32`, `POLICY_OUT=19`, `VALUE_HIDDEN=16`.

## Multi-agent handling

- 12-seat FFA, **self-play** by default (no aggressive opponent — flag `--aggressive-seat -1` disables it).
- Each game yields 12 trajectories — one per seat — per rollout. Rich data per game.
- All seats use the latest policy during a rollout; experience from all 12 updates the same parameters.
- If `--aggressive-seat >= 0`, that seat's actions are overridden by the hand-coded heuristic and its trajectory is masked out of the policy loss (`seat_mask`).

## Reward shaping

**Dense per-AI-tick cell-delta** + a small terminal bonus on win. The cell-delta term is `cells_owned_at_t - cells_owned_at_{t-1}` per seat, gated by liveness so dead games don't contribute spurious signal. Naturally positive in early-game expansion, near-zero in stalemate.

Initial training shows `mean_total_reward ≈ 21.58` per rollout, glued to a structural constant — symmetric self-play with near-uniform random policy produces near-deterministic expansion totals. Policy commitment will need to drive that number to break.

## Rollout strategy

- Run `G` parallel games for `max_ticks` (or until single-seat win), collect `(owner, strength, action, log_prob, value, reward, done)` per AI-tick per seat.
- After collection, compute **GAE-λ** advantages (γ=0.99, λ=0.95) and normalized returns.
- K epochs of **minibatch SGD** on the PPO clipped surrogate loss (clip_eps=0.2) + value MSE (coef 0.5) + entropy bonus (coef 0.01).
- Adam, lr=3e-4.

## Performance

Real-size config (G=4, max_ticks=5000, radius=9, mb=128, update_epochs=2):

| stage    | baseline | post-fix |
|----------|----------|----------|
| rollout  | 4.5s     | 2.5s     |
| update   | 17.9s    | 2.5s     |
| total    | ~22s     | ~5s      |

**4.7× speedup.** The wins:

- **`update_epochs 4 → 2`** — half the gradient work; the bulk of the win. PPO with `update_epochs=2` still produces useful gradients while skipping the diminishing-returns later epochs.
- **Eval coalescing in rollout** — single `mx.eval(logits, value, actions, owner, strength)` per AI tick instead of four separate evals; ~19% rollout speedup.
- **Hoisted `seat_mask` onto GPU once per update** instead of per minibatch — negligible but free.
- **Metric side-channel** — `loss_fn` appends `(policy_loss, value_loss, entropy, approx_kl, clip_fraction, ratio_mean, ratio_max)` to a python list; the main loop pops them after `grad_fn` returns and evaluates alongside the gradients. No redundant forward pass for metrics.

A perf subagent found that `mx.compile` on the train step gives another ~30% in isolation. Not landed yet — current speed is acceptable.

## File map

- `python/flux/ppo.py` — `GNNActorCritic` mlx.nn.Module. Forward returns `(policy_logits, value)`.
- `python/scripts/train_ppo.py` — main entry. `uv run python scripts/train_ppo.py --games-per-rollout 4 --max-ticks 5000 --aggressive-seat -1 --update-epochs 2 --wandb`. Auto-resume from `python/checkpoints/ppo/latest.npz` unless `--fresh`.
- `python/flux/mlx_batch.py` — `build_flows_from_actions(actions_all, owner, graph_neighbors)` builds flow tensors from sampled actions; used during rollout.
- `python/checkpoints/ppo/latest.npz` — policy + Adam state, snapshotted every 5 iters and at shutdown.
- `public/replays/train_ppo_*.flxr` — per-iter replay (game 0). Recorded flows are written into each frame.
- Wandb run prefix: `ppo-*`.

## Wandb instrumentation

The training loop logs ~20 metrics per iter. Headline panels:

- **PPO update health:** `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction` (fraction of ratios outside `[1-ε, 1+ε]`), `ratio_mean`, `ratio_max`, `grad_norm`, `weight_norm`.
- **Value head sanity:** `explained_variance` (the key metric — 1.0 = perfect predictor, 0 = predicts mean), `value_mean`/`value_std`, `return_mean`/`return_std`.
- **Raw reward signal:** `reward_step_mean`, `reward_step_std`, `reward_step_max`, `reward_step_min`.
- **Behaviour:** `action_entropy` (entropy of the 19-way distribution; log(19) ≈ 2.94 is uniform random), `action_pick_top_frac` (fraction landing on the most-common action), `action_self_frac` (fraction choosing index 18 = "no flow").
- **Game outcomes per rollout:** `cells_max_end`, `cells_min_end`, `dominance` (max/total owned), `alive_seats_end`, `neutral_frac_end`.
- **Heavy artifact:** `end_state` image (per-game ownership grid, each row = one of G games) emitted every 20 iters.

## Replay integration

Each rollout writes a per-iter `.flxr` to `public/replays/`. Game 0's flows at each AI-tick are pulled from `flow_src/dst/player/valid` and stored in the frame so the browser can render directional arrows. `metadata.best_fitness` is set to `mean_reward` (currently structural ~21.58 — not a great fitness proxy yet, but standardizes on the field name the browser top-bar already reads).

The greatest-hits cycle in the browser ([[replay-rendering]] § greatest hits) walks `public/replays/greatest-hits.json` built by `python/scripts/build_greatest_hits.py` — filters fitness > 0 and frame count, sorts longest-first, keeps top 30.

## MLX specifics

- MLX autograd works end-to-end on this codebase. First-iteration kernel compile is **substantial** at `max_ticks=5000` — iter 1 can take 1–2 minutes vs <5s per iter steady state. Don't kill a hung-looking process under 2 minutes.
- Apple Silicon unified memory means rollout buffers stay in MLX — no explicit CPU↔GPU transfers, but a single Python process can show 20–40GB resident as MLX caches buffers. Almost all of that is reusable cache, not strictly required.
- `mx.full_like` does not exist (`mx.zeros_like` does). Use `mx.full(shape, value, dtype=...)`.
- Output buffering bites: under `uv run python ...` with redirected stdout, the script's `print()` is block-buffered. Use `PYTHONUNBUFFERED=1` or `python -u` if you want to see iter timings live.

## What stays the same

- Game step, replay format (`.flxr`), browser replay player.
- Aggressive AI (still available as an opponent seat).
- Wandb logging shape — same project, distinguishable run prefix.
- The GNN architecture from [[v3-gnn]].

## What changed

- Learning algorithm: NEAT → PPO.
- Population concept: gone. One policy, one set of weights.
- Mutation / tournament selection: gone. Replaced by gradient updates.
- Reward: per-game scalar fitness → per-AI-tick dense cell-delta + terminal bonus.

## Current observation

At iter ~196 (resumed under `ppo-r9-ep2-instrumented`):

- `explained_variance ≈ 0.74` — value head IS learning structure.
- `entropy ≈ 2.92` (max is 2.94) — policy is still near-uniform random.
- `mean_total_reward = 21.58` — pinned to a structural symmetry constant; will move once policy starts committing.

Pattern matches early PPO: value head trains fast on the dense reward, policy needs longer before it starts shifting probability mass meaningfully. Watch `action_entropy` dropping as the indicator that strategy is emerging.

## Status

**Active.** Implementation shipped; instrumented training run in progress.
