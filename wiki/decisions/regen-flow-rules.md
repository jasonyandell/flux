---
title: Regen-Flow Game Rules
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Choice

A second, parallel ruleset for the game. Existing rules (the "transfer" ruleset documented in [[continuous-flow-model]] / [[attack-bonus]]) stay intact and continue to drive all existing replays and the live browser game. The new ruleset, **regen-flow**, decouples attack cost from sender health and adds linear regen scaling. Training, replays, and the browser all carry a `ruleset` tag (`"transfer" | "regen-flow"`) so the two worlds don't bleed into each other.

## Why

Across v1/v2/v3/PPO it became clear that the transfer-flow model conflates several mechanics: a flowing cell *bleeds* into its destination, *and* the destination accrues damage from non-friendly inflow. The sender's health drops merely from sending, which means even uncontested attacks attrite both sides. That made loops self-defeating (every member is bleeding) and conflated "I'm attacking" with "I'm choosing to take damage." The new model separates the two.

## Mechanics

Per cell `C` (strength `s`, owner `O`, `K` friendly outflows) per tick:

- **Idle** (`K = 0`): `s ← s + regen(s) · dt + support_in − attack_in`. The cell banks regen + any net friendly support.
- **Sending** (`K ≥ 1`): forfeit regen as a growth term; instead, the cell's output capacity is `regen(s) + passthrough_carry`. Each outflow delivers `min(output_capacity, MAX_OUTPUT_PER_SEC) / K · dt`. The sender's strength only changes by `−attack_in` (no support banked while sending — see passthrough below).

Where:

```
regen(s) = REGEN_BASE_PER_SEC · (1 + REGEN_SLOPE · (s − 1))
```

with `REGEN_BASE_PER_SEC = 0.5` and `REGEN_SLOPE = 2.0`. Linear scaling — at strength 1, regen = 0.5/s; at strength 10, regen = 9.5/s. Big cells are *much* better at idle growth than at projecting force.

**Passthrough (1-tick lag).** A sending cell's *output_capacity* is `regen(s) + passthrough_carry`, where `passthrough_carry` is the friendly support that cell received on the *previous* tick (and only while it was sending — idle cells bank support directly into strength). This makes sending cells **transparent relays**: support arrives, doesn't add to strength, gets re-emitted next tick as part of outflow. Loops self-amplify because each downstream member carries forward what its upstream sent, plus its own regen. Capped at `MAX_OUTPUT_PER_SEC = 100` per outflow so a long chain can't fire an arbitrarily large insta-kill bolt. Captured cells reset their passthrough to 0.

**Damage is symmetric.** A friendly destination gains `flux`, an enemy destination loses `flux`. No attacker multiplier. Whatever edge the attacker has is strategic — picking when and where to engage — not built into the math.

**Capture.** When a cell's net delta would push strength below 0, the cell flips to the largest non-owner contributor and resets to `CAPTURE_STRENGTH = 1.0`. Deterministic. Any leftover damage past 0 is wasted; the captured cell never inherits an oversized strength from the attacking blow.

**Cap.** Strength is capped at `MAX_STRENGTH = 100` as before. Overage-propagation through outflows (the "loop becomes a power generator once members cap out" mechanic) is *partially* implemented now via the passthrough mechanic — capped cells still pass through whatever flows in, since passthrough is decoupled from strength banking. True overage of inflow exceeding `MAX_OUTPUT_PER_SEC` is still wasted.

## Dead cells + random starts

Two board-randomization knobs are wired through:

- **`--num-dead-cells N`** marks N random cells as **dead** per game (independently chosen each game in the rollout). Dead cells cannot be targeted by flows (flows pointing at them are dropped at flow-build time), can never receive support, never regen, can't be captured. They're untouchable obstacles. The replay header carries the per-game-0 dead set in `metadata.dead_cells`.
- **`--randomize-starts`** places the P seats at random distinct (non-dead) cells per game rather than the deterministic evenly-spaced perimeter layout. Some games will be wildly unfair (one seat surrounded by enemies, another in a quiet corner); the policy learns to handle the distribution.

The GNN input is extended to **5 channels**: `strength_norm, is_mine, is_enemy, is_neutral, is_dead`. A cell that's dead is `is_neutral=0, is_dead=1` so the policy can distinguish "untouchable obstacle" from "capturable empty cell." This breaks checkpoint compatibility with the 4-channel ruleset (existing transfer-flow checkpoints can't load into the new GNN).

## Strategic consequences

- **Loops are positive-sum at small/medium sizes.** Each member of a friendly loop forfeits `regen(self)` and receives `regen(neighbor) / K`. With equal-strength members and one outflow each, that nets `regen − regen = 0` — break-even. But because `regen` grows fast with `s`, a loop with even one larger member pumps strength downstream to its smaller neighbors. Heterogeneous loops are the engine.
- **Big idle cells are accumulators.** A strength-90 cell regens at ~89/s if idle; if it sends, it can only project ~89/s in one direction. Disengagement is the growth strategy.
- **Engagement is a commitment.** A cell that's sending pauses its own growth and projects its full capacity outward. There's a real cost to staying engaged.
- **Defense alone loses.** Symmetric damage means a defender forfeiting regen to counter-attack still pays the regen tax, but the attacker isn't required to defend in return — they pick the timing. Pure defense attrits.

## File map

- `src/game/step_regen.ts` — TS reference. Same `GameState` shape; new step function. (Live browser play still uses `step.ts` / transfer-flow.)
- `python/flux/step_regen.py` — Python parity reference. NumPy-equivalent precision.
- `python/flux/mlx_step_regen.py` — batched MLX kernel for training. Reuses `build_flows_from_actions` from `mlx_batch.py` (flow tensor shape unchanged).
- `python/scripts/train_ppo.py` — accepts `--ruleset regen-flow`, dispatches `step_fn = step_batched_regen`. Default checkpoint becomes `python/checkpoints/ppo-regen/latest.npz`.
- Replay metadata gains `ruleset: "regen-flow"` (default `"transfer"` for older `.flxr` files).

The replay binary format itself is unchanged — only the metadata JSON gains a field.

## What about the browser

For replay rendering, the browser does not simulate — it just plays back recorded frames. So no TS step changes are needed there. The frame shape is identical.

For live play, the user can still play under the old transfer rules; switching live play to regen-flow is a follow-up (would require wiring `step_regen` through the `aiThink`/sim loop).

## Reward shaping (active)

The dense per-AI-tick reward, applied per (game, seat), is:

```
reward = cell_delta_coef · Δcells
       + engagement_coef · (cells_sending / cells_owned)
       − idle_capped_coef · (idle_near_cap_cells / cells_owned)
       + output_boost_coef · avg(output_rate / MAX_OUTPUT_PER_SEC of sending cells)
```

Default coefs: `engagement=0.01`, `idle_capped=0.02`, `output_boost=0.05`. Each is a normalized fraction in `[0, 1]`, so dead/empty seats contribute 0 naturally.

- **engagement** pushes idle cells off the bench.
- **idle_capped** punishes the specific waste case (maxed cell sitting on regen it can't bank).
- **output_boost** is the chain-builder: a cell whose `regen + passthrough` is high (because upstream friends are pumping into it) earns more than an isolated sender. This is the credit-assignment link that makes "form a chain" directly visible to gradient descent without needing to trace 4-cell-deep downstream capture rewards.

Empirically: with all four terms active, `mean_total_R` climbs from ~31 → ~44 in the first 300 iters at radius=5 / P=3 / d=1, vs ~27 → ~27 stuck under cell-delta-only. Entropy commits faster too (2.94 → 2.61 in ~300 iters vs 2.94 → 2.80 in ~230 iters).

## What still needs deciding

- **K > 1 in training.** PPO actions are one-per-cell, so K is always 0 or 1 in trained rollouts. Multi-outflow behavior is exercised only via the TS path for now.
- **Browser live-play wiring.** `step_regen.ts` exists but isn't wired into the live sim. Replays render fine (just frame playback). If human play under regen-flow is wanted, the live `aiThink` loop needs to be parameterized on ruleset.
- **Dead-cell visualization in the browser.** Replay metadata carries the dead set as `metadata.dead_cells`, but the renderer doesn't yet color them distinctly — they show as strength-0 neutrals. Easy follow-up.
- **True overage cap (above MAX_OUTPUT_PER_SEC).** Currently capped per-outflow; chains across many capped cells still bottleneck at the per-edge cap. May want a softer overflow that propagates further before being wasted.

## Status

**Active.** Implementation shipped; first training run is `ppo-regen-r5-p3` at radius 5 / 3 seats / 8 games per rollout / 3000 max ticks / self-play.
