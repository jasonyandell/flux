---
title: v2 — 13-action Set/Clear/Noop space
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-13
status: active
---

## What

Each AI tick, per owned cell, the policy emits one action from a 13-action
space:

- `0..5` — **set** slot `k` (idempotent if already on)
- `6..11` — **clear** slot `k` (idempotent if already off)
- `12` — **no-op**

Multi-outflow is supported: a cell can have several outflows set
simultaneously across consecutive AI ticks.

## Why Set/Clear instead of toggle

Idempotent, state-independent semantics. Each action token has the same
meaning regardless of current outflow state, so the network doesn't need
the current outflow vector as input to predict its effect.

With K=6 the output layer is small either way (13 vs 7 toggle), so toggle's
compactness advantage doesn't pay for its state-dependence cost. The policy
*does* get the current outflow count as a feature (so it can sense its
persistent decisions), but it never has to invert "what does toggle do here?"

## Mutation invariants

Resolved at AI tick time, not at physics tick time:

1. **No friendly bidirectional flow.** If c sets a slot toward friendly d and
   d's back-edge is also set, higher cell-index keeps its outflow; the lower
   loses.
2. **Capture clears origin.** When my cell's owner changes (captured by an
   enemy), all its `outflow_intent` bits reset to 0.
3. **Stale targets stay on.** If c→d was set and d gets captured by an enemy,
   c's outflow stays set. The pressure now arrives as enemy damage at the new
   owner of d. The raised `CAPTURE_STRENGTH = 50` makes this livable.

## Implementation

- `python/flux_v2/state.py` — `NUM_ACTIONS = 13`, action code constants.
- `python/flux_v2/step.py::apply_actions` — pure-Python implementation with
  invariant resolution.
- `python/flux_v2/mlx_step.py::apply_actions_batched` — batched MLX version.
- `python/flux_v2/ppo.py` — actor head outputs 13 logits per cell.

Related: [[v2-edge-pressure-state]].
