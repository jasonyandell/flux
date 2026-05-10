---
title: Loop bonus
kind: decision
first_seen: 2026-05-10
last_updated: 2026-05-10
status: active
---

## Decision

Friendly destinations receive a multiplier `1 + LOOP_BONUS` (default `0.5`) on incoming transfer. Source loss is unchanged. Enemy and neutral destinations are unaffected.

```ts
const k = TRANSFER_PER_SEC * dt;
forces[flow.src][flow.player] -= k;
const friendly = state.nodes[flow.dst].owner === flow.player;
forces[flow.dst][flow.player] += friendly ? k * (1 + LOOP_BONUS) : k;
```

## Why

Before this change a closed friendly loop (e.g. `a → b → c → a` with all three owned by the same player) was zero-sum on transfer: each node lost `k` as a source and gained `k` as a destination. Only regen contributed. Circulation felt like a wash; players had no reason to maintain a chain in their own territory.

With the bonus, the same 3-cycle nets `+REGEN + LOOP_BONUS * TRANSFER_PER_SEC * dt` per node per tick on top of the baseline. At defaults (`REGEN_PER_SEC = 1`, `TRANSFER_PER_SEC = 3`, `LOOP_BONUS = 0.5`, `dt = 0.1`) that is `+0.25/tick` extra per node — a friendly 3-cycle grows 2.5× faster than three idle owned nodes.

## Consequences

- The `step` signature is unchanged. Pure-core invariants hold.
- `MAX_STRENGTH` already caps growth; no runaway. A maxed-out friendly cycle just stops gaining.
- Enemy/neutral targets are unchanged, so combat math (capture conditions, flips) is unaffected.
- The dumb AI (`src/ai/dumb.ts`) does not deliberately build chains, so headless sim outcomes are unaffected at this scale — the symmetric stalemate persists. A loop-aware AI is now a meaningfully more powerful opponent than the current heuristic.

## Rejected

- **Multiplier on source loss instead** (cheaper to send into friendly): symmetric numerically (`k` net), but obscures the "circulation generates" mental model. The user explicitly framed this as growth, not as a discount.
- **Bonus scales with cycle length**: would require cycle detection in `step`. Out of scope and adds work proportional to flows × graph traversal each tick.
