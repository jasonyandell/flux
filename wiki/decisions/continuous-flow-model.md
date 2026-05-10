---
title: Continuous flow model
kind: decision
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Decision

Node strength is a continuous scalar. There are no in-flight unit entities. Flows are toggleable boolean rates per ordered (src, dst) pair across an edge, with at most one flow per undirected edge (see [[one-flow-per-edge]]).

Per tick:

- Owned nodes regen strength at a fixed rate.
- For each active flow, the source loses `TRANSFER_PER_SEC * dt`. If the destination is friendly it gains `TRANSFER_PER_SEC * dt * (1 + LOOP_BONUS)` — friendly circulation generates strength, see [[loop-bonus]]. Otherwise the destination loses `TRANSFER_PER_SEC * dt`.
- When a node's strength crosses zero, ownership flips to the largest contributing enemy and the surplus becomes the new owner's strength.

## Why

- State is two scalars per node and a list of toggles. No transit positions, no interpolation, no per-particle rendering.
- The per-tick contribution at a node is naturally a sum of independent partial contributions per player, which keeps `step` short, pure, and trivially deterministic.
- Headless simulations are fast because the work per tick is `O(nodes + flows)` arithmetic.

## Rejected

- **Discrete ships in transit** (Galcon-style): doubles the state and the renderer for marginal feel improvement.
- **`ticksToMilestone` integer countdowns**: considered briefly. Attractive for bit-exact determinism, but the continuous model is already deterministic given fixed `dt`, and integer countdowns made the resolve step harder to reason about.
