---
title: v2 — reward stack
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-13c
status: active
---

## What

v2 reward shape, per AI tick, per seat. The filename is historical: the
initial shape was three-term, but the active trainer now keeps each
action-conditioned signal separate so experiments can turn them on/off.

```
step_reward[p] =
    + power terms                             # owned strength / damage work
    + capture_coef * Δ(cells_owned[p])        # net cells gained
    + transit_coef * transit_credit[p]        # optional friendly relay credit
    + kill_pressure_coef * kills[p]           # per-tick attributed eliminations
    - waste_coef * waste_per_player[p]        # dead-end pressure
    - time_coef                               # impatience → speed
terminal_reward[winner] += win_bonus
```

The historical CLI defaults are not the known-good recipe. Current successful
runs use explicit flags from [[v2-training-runs]]. The strict transit experiment
started with `--transit-coef 0.001`; `0.1` was immediately too hot.

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
  cap is a system limit, not the policy's fault).

  Three categories, three weights in `python/flux_v2/state.py`:

  | Category | Trigger | Weight | Attributed to |
  |---|---|---|---|
  | `no_spill` | source overflows with zero outflows | `1.0` | source |
  | `cap_bound` | per-edge clip on the way out | `0.0` (off) | source |
  | `dest_terminated` | active outflow lands on a friendly cell at MAX strength AND with zero outflows (pure sink) | `0.3` | source |

  `dest_terminated` was added after `v2-killer-tuned` (iter 253) where the
  red player visibly crammed massive friendly pressure into a dead-end MAX
  cell at the map edge. A MAX cell *with* outflows is a combo relay (good —
  pressure passes through), so we only penalize the no-outflow sink case.
  Pass-through is what makes combos work; the rule is careful not to break
  that. See [[v2-training-runs]] for the run that surfaced the pathology.
- **Transit credit** is the positive twin of `dest_terminated` waste. A source
  cell gets credit when its active outflow sends pressure into a friendly
  relay: a destination with active outflows of its own. `TRANSIT_CREDIT_STRICT`
  is true, so the destination must also be at MAX strength. That targets the
  combo/back-line pattern without paying ordinary fill traffic. The trainer
  exposes this as `--transit-coef`; default is off, first live run uses
  `0.001`.
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

- `python/scripts/train_v2.py::collect_rollout` — computes `r_power`,
  `r_capture`, `r_waste`, `r_transit`, `r_time`, and `r_kill` per (G, P)
  per AI tick, stores them separately for wandb panels alongside the
  combined reward.
- The trainer broadcasts those panels (`reward_power_iter`,
  `reward_capture_iter`, `reward_waste_iter`, `reward_transit_iter`,
  `reward_time_iter`, `reward_kill_iter`) for tuning.

Related: [[v2-edge-pressure-state]] (where waste comes from).
