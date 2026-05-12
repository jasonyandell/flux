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

- **Idle** (`K = 0`): `s ← s + regen(s) · dt`.
- **Sending** (`K ≥ 1`): forfeit regen entirely. Each outflow delivers `regen(s) / K · dt` to its destination. The sender's own strength does not decrease from sending.

Where:

```
regen(s) = REGEN_BASE_PER_SEC · (1 + REGEN_SLOPE · (s − 1))
```

with `REGEN_BASE_PER_SEC = 0.5` and `REGEN_SLOPE = 2.0`. Linear scaling — at strength 1, regen = 0.5/s; at strength 10, regen = 9.5/s. Big cells are *much* better at idle growth than at projecting force.

**Damage is symmetric.** A friendly destination gains `flux`, an enemy destination loses `flux`. No attacker multiplier. Whatever edge the attacker has is strategic — picking when and where to engage — not built into the math.

**Capture.** When a cell's net delta would push strength below 0, the cell flips to the largest non-owner contributor and resets to `CAPTURE_STRENGTH = 1.0`. Deterministic. Any leftover damage past 0 is wasted; the captured cell never inherits an oversized strength from the attacking blow.

**Cap.** Strength is capped at `MAX_STRENGTH = 100` as before. Overage-propagation through outflows (the "loop becomes a power generator once members cap out" mechanic that came up in the design discussion) is a v1 follow-up — currently overage past the cap is discarded.

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

## What still needs deciding

- **Overage propagation through caps.** A capped cell receiving inflow could pass overage through its own outflows additively. Discussed but not implemented; current code discards.
- **K > 1 in training.** PPO actions are one-per-cell, so K is always 0 or 1 in trained rollouts. Multi-outflow behavior is exercised only via the TS path for now.
- **Reward shaping for regen-flow.** Currently reusing `cell_delta_reward` from the transfer ruleset. Likely needs tuning — regen-flow rewards expansion more directly (you can attack without bleeding) so absolute reward magnitudes will differ.

## Status

**Active.** Implementation shipped; first training run is `ppo-regen-r5-p3` at radius 5 / 3 seats / 8 games per rollout / 3000 max ticks / self-play.
