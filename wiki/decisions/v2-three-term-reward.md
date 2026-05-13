---
title: v2 — three-term reward (power, waste, time)
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-13
status: active
---

## What

v2 reward shape, per AI tick, per seat:

```
step_reward[p] =
    + power_coef   * Δ(Σ strength_owned[p])  # cells filling toward MAX
    + capture_coef * Δ(cells_owned[p])        # net cells gained this tick
    - waste_coef   * waste_per_player[p]      # dead-end pressure
    - time_coef                                # impatience → speed
terminal_reward[winner] += win_bonus
```

Defaults: `power_coef=0.05`, `capture_coef=0.0` (off by default;
`50.0` enables it), `waste_coef=0.05`, `time_coef=0.01`, `win_bonus=50`.

**Why `capture_coef` exists** (added 2026-05-13 after multiple runs went
pacifist): `Δ(strength_owned)` saturates to zero once all owned cells are
at MAX, so a closed-loop policy gets no power signal in steady state.
`Δ(cells_owned)` doesn't saturate — captures and losses always register.
With `capture_coef = 50` each capture is worth +50 reward directly,
separate from the strength term. This makes captures economically
motivated even when held territory is fully developed.

## Why this set

PPO does better with a small set of clean signals than a dense stack of
overlapping ones — v1's hard-learned lesson.

- **Power** is the stock measure (sum of strength on owned cells). The
  per-tick delta avoids huge magnitudes. Higher strength already implicitly
  rewards more regen (regen scales with strength), so this single term
  captures "more territory + more developed" together.
- **Waste** is the regen a cell *didn't send*. Simple framing: if you have
  any outflows set, your regen flows out through them — that counts as
  "sent" regardless of whether the per-edge cap clipped some of it (the
  cap is a system limit, not the policy's fault). The only true waste is
  overflow at a cell with zero active outflows.

  Implementation: `WASTE_WEIGHT_NO_SPILL = 1.0`, `WASTE_WEIGHT_CAP_BOUND
  = 0.0` in `python/flux_v2/state.py`. This is the third iteration of the
  waste rule; earlier attempts weighted both kinds of waste (1:1 and 10:1
  and 100:1) but the cap-bound signal added high variance that made PPO
  unstable. The "regen-not-sent" framing collapses to no_spill-only and
  is much cleaner.
- **Time** is a small per-tick penalty that pushes finish-the-game pressure.
- **Win bonus** is a terminal scalar for the last-alive seat.

## What's gone

- v1's `engagement_coef` ("fraction of cells with active outflows") makes no
  sense in v2 — a stable loop has every cell active permanently and shouldn't
  be rewarded extra for that. Persistence makes activity measures meaningless.
- v1's `idle_capped_coef`, `output_boost_coef` — same reason. Subsumed by
  power Δ + waste.

## Deliberately not counted

**Overkill** (an attacker pumping 100 pressure into a 5-strength cell) is not
counted as waste. The attacker can't know the defender's strength at commit
time; penalizing it teaches the policy to be too cautious. Open to revisit
if overkill dominates observed waste during training.

## Implementation

- `python/scripts/train_v2.py::collect_rollout` — computes
  `r_power`, `r_waste`, `r_time` per (G, P) per AI tick, stores them
  separately for wandb panels alongside the combined reward.
- The trainer broadcasts those panels (`reward_power_iter`,
  `reward_waste_iter`, `reward_time_iter`) for tuning.

Related: [[v2-edge-pressure-state]] (where waste comes from).
