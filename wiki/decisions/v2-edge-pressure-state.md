---
title: v2 — edges are first-class state
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-13
status: active
---

## What

In flux v2, each directed half-edge `(c, slot k)` carries a persistent
`edge_pressure` value. It's rewritten every game tick from the source cell's
overflow and read by the destination cell next tick. The policy chooses which
slots are *active* (via `outflow_intent[c, k]`); the simulation does the rest.

## Why

v1's failure mode was that "every cell re-decides its outflow every 5 ticks."
A loop of N cells needed N cells × hundreds of consecutive correct decisions
for the loop to actually persist. PPO found good single-step policies but
never built emergent structure.

v2's fix: **persistence over re-assertion**. Set an outflow once; it stays on
until the policy clears it. The edge holds the carried flux, propagating one
hop per tick. Multi-hop transport, stable loops, and maxed-cell fanout all
fall out of the same fill-then-overflow rule — no special cases.

## How (per-tick algorithm)

For each cell `c` (reading end-of-last-tick state):

```
pressure_in_friendly = regen(strength_c)
                     + Σ edge_pressure[d, slot_d→c] for friendly neighbors d
pressure_in_enemy    = Σ edge_pressure[d, slot_d→c] for enemy neighbors d

grew         = min(pressure_in_friendly, MAX − strength_c)
new_strength = clamp(strength_c + grew − pressure_in_enemy, 0, MAX)
overflow     = pressure_in_friendly − grew

if num_active_outflows > 0 and overflow > 0:
    per_edge = min(overflow / num_active_outflows, MAX_EDGE)
    edge_pressure_next[c, active_slots] = per_edge
    waste += (overflow / num_active − per_edge) * num_active
else:
    edge_pressure_next[c, *] = 0
    if overflow > 0: waste += overflow
```

`OPPOSITE_SLOT[k] = (k + 3) % 6` resolves the back-edge between any pair.

## Implementation

- `python/flux_v2/state.py` — State dataclass: `(N, owner, strength,
  outflow[N, K], edge_pressure[N, K])`.
- `python/flux_v2/step.py` — pure reducer (`tick`, `apply_actions`,
  `waste_per_cell_for_tick`).
- `python/flux_v2/mlx_step.py` — same algorithm batched across G games. The
  persistent `edge_pressure` array is part of the batched state.
- `python/tests/test_v2_step.py` — covers loop persistence, capture
  strength, multi-hop transport, no-friendly-bidirectional, waste accounting.
- `python/tests/test_v2_mlx_parity.py` — random-init parity between pure
  reducer and MLX batched path (25 ticks, dead cells, random outflows).

Related: [[v2-vision]] (the earlier 3-hop input experiment), [[ppo-gnn]]
(v1 trainer this branched off of).
