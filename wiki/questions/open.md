---
title: Open questions
kind: question
first_seen: bootstrap
last_updated: workspace
status: active
---

## Open (v2 frontier)

### Can a learned policy beat `lightning_sum` / `wave_long` under big-bag rules?

As of 2026-05-15 it has not. Architectural variants tried:

- PPO + GCN with node-centric Set/Clear actions
  ([[../decisions/ppo-gnn|ppo-gnn]],
  [[../topics/v2-training-runs|v2-training-runs]]).
- PPO with the attention head from `lightning_attn` —
  ([[../topics/v2-edge-loop-emergence|v2-edge-loop-emergence]]), abandoned
  after losing decisively (see
  [[../topics/v2-overnight-research|v2-overnight-research]]).

Open lines:

- The proposed edge-voting policy
  ([[../topics/v2-edge-voting-policy|v2-edge-voting-policy]]) — does
  per-edge logit aggregation with visibility normalization close the gap
  where node-centric Set/Clear couldn't?
- Curriculum (`bfs` → `lightning` → `lightning_sum`) instead of mixed-seat
  self-play that satisfies on parity rather than dominance.
- More PPO capacity (e.g. HIDDEN=64) plus longer training (1000+ iters,
  entropy 0.01 → 0.001).

### Is `wave_long`'s gating doing anything, or is `sum_long` the simpler win?

Overnight exp 25 split `wave_long` into components: `sum_wave` (gate only)
contributed 0/0 coherent decisions; `sum_long` (γ=0.94 only) contributed
5/6. The gate is doing nothing measurable; the long-range field is what
wins. Replicating this at larger boards (R=25+) and across more random
seeds is open.

### Why does γ=0.94 help `sum` and hurt `max`?

Exp 27 measured this directly: `sum(γ=0.94)` vs `max(γ=0.94)` won 16/16
coherent decisions. The proposed mechanism is "integrating aggregators
benefit from longer reach because all far-off contributions enrich the
signal; max picks one term, so longer reach just adds distracting noise."
The generalisation — *aggregator semantics constrain useful hyperparameter
regimes* — is the open testable principle for future solver design.

### Is the regen-flow ruleset materially better for emergence?

[[../decisions/regen-flow-rules|regen-flow-rules]] is alive as a parallel
ruleset (sender has no health cost; damage is symmetric; linear regen
scaling; deterministic capture). No head-to-head against the transfer-flow
ruleset under big-bag has been recorded. Open: does either ruleset
produce more legible emergent behavior, and which is the better learning
target for PPO?

## Closed / answered

### Dumb AI stalemates against itself (v1, answered)

`npm run sim` of dumb-vs-dumb produced 100% draws at the tick limit on
the 1000-cell hex board, and the four "weakest-local-neighbor" AIs in
[[../decisions/ai-zoo|ai-zoo]] all stalemate against each other for the
same reason — the contested center is never the weakest neighbor and
attacks bypass the choke. Mitigated by
[[../decisions/stasis-detection|stasis-detection]] (browser ends wedged
sessions) and answered by
[[../topics/neuroevolution|neuroevolution]] (Tier 5 MLX champions
*sometimes* beat the zoo, reproducibly enough to call it validated).
The deeper structural issue — that the v1 transfer-flow rules conflate
"I'm attacking" with "I'm taking damage" — is what v2's edge-pressure
state ([[../decisions/v2-edge-pressure-state|v2-edge-pressure-state]])
fixes.
