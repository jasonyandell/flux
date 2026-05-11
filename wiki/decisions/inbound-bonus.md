---
title: Inbound bonus
kind: decision
first_seen: 2026-05-10
last_updated: 2026-05-10
status: retired
---

## Retired

Briefly tried a per-tick regen bonus proportional to active inbound friendly flow count:

```ts
regen = REGEN_PER_SEC + INBOUND_BONUS * inbound_friendly_flow_count
```

Implemented in `src/game/step.ts` and `src/gpu/shaders/step.wgsl` with `INBOUND_BONUS = 0.05`, then reverted within the same session.

The wall-breaker / "choo-choo" dynamic the bonus was reaching for was already produced by the three existing mechanics (see [[continuous-flow-model]]): `+k` friendly inbound delivery + base regen + `(1+ATTACK_BONUS)*k` outgoing damage. A 3-cell chain delivers ~3× the total damage of a single isolated attacker without help from this bonus. The proposed `+0.05/sec` per inbound flow was invisible next to the `+3/sec` inbound transfer it sat beside, and for hub destinations the cell was MAX-capped so the bonus did literally nothing.

Kept as a record of a considered-then-rejected addition.
