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

- Owned nodes regen strength at `REGEN_PER_SEC` per tick.
- For each active flow, the source loses `TRANSFER_PER_SEC * dt`. If the destination is friendly it gains the same amount; otherwise the destination loses `TRANSFER_PER_SEC * dt * (1 + ATTACK_BONUS)` — combat is asymmetric, see [[attack-bonus]].
- When a node's strength crosses zero, ownership flips to the largest contributing enemy and the surplus becomes the new owner's strength.

The wall-breaker / "choo-choo" dynamic (a chain of friendly cells sustaining an attack on a fortress) emerges from these three mechanics together without a separate bonus: friendly flow delivers `+k` to the front-line attacker, the attacker's outgoing drain is offset, the attacker delivers `(1+ATTACK_BONUS)*k` damage at the enemy. A 3-cell chain delivers ~3× the total damage of a single isolated attacker before depleting. See the retired [[inbound-bonus]] for the math walk that confirmed no extra mechanic was needed.

## Why

- State is two scalars per node and a list of toggles. No transit positions, no interpolation, no per-particle rendering.
- The per-tick contribution at a node is naturally a sum of independent partial contributions per player, which keeps `step` short, pure, and trivially deterministic.
- Headless simulations are fast because the work per tick is `O(nodes + flows)` arithmetic.

## Rejected

- **Discrete ships in transit** (Galcon-style): doubles the state and the renderer for marginal feel improvement.
- **`ticksToMilestone` integer countdowns**: considered briefly. Attractive for bit-exact determinism, but the continuous model is already deterministic given fixed `dt`, and integer countdowns made the resolve step harder to reason about.
