---
title: v2 edge-voting policy spec
kind: topic
first_seen: workspace
last_updated: 2026-05-15
status: active
---

## Scope note (2026-05-15)

This page specifies the *spatial* output factorization — how the
policy expresses per-edge intent. It does not address temporal
commitment (throttle, target persistence, multi-tick options); see
[[v2-temporal-strategy]] for that layer. The two are complementary: an
edge-voting head emits *what flow field* to maintain, a slow recurrent
manager decides *which goal* the flow field is serving and *how long*
to ride it.

## Purpose

The v2 learner should move from node-centric cell actions toward an edge-centric
flow field. Local flow optimization is close to deterministic: pressure should
move through useful relays and prefer enemy/neutral targets when release is
productive. The hard learning problem is not "what is a pipe?" It is when to
hold, release, redirect, follow through, defend, or abandon.

This spec defines a representation that exposes local edge semantics without
hard-coding pulse timing. It gives the learner better physical affordances while
leaving charge/release behavior emergent.

## Current Limit

The active PPO/GNN policy is node-centric. For each seat and cell it sees nine
aggregate node features: strength, ownership flags, outflow count, total
friendly inbound pressure, total enemy inbound pressure, and total outgoing
pressure. A 3-layer GCN mixes those node embeddings over about three graph
hops, then emits 13 per-cell logits: `Set 0..5`, `Clear 0..5`, `No-op`.

The world state is already edge-rich (`outflow[c,k]`,
`edge_pressure[c,k]`), but the model does not directly score a candidate edge
as source + slot + destination. Edge semantics such as "this outflow targets an
enemy" are currently derived in reward/metrics code, not represented as a
first-class policy input.

## Proposal

Use a distributed edge-voting policy.

For each seat:

1. Every owned or nearby visible node observes its local patch.
2. The observer emits suggestions for directed edges it can see, not only for
   its own outgoing slots.
3. Suggestions for the same directed edge `(src, slot)` are summed or averaged.
4. The final edge intent becomes a small action decision:
   - strongly positive: open / keep open.
   - strongly negative: clear / keep closed.
   - near zero: hold current state.

This turns each cell into an advisor over the local flow field. The source can
vote from stored pressure; the destination can vote from target quality; nearby
frontier cells can vote from attack opportunity; upstream cells can vote from
route coherence.

## Edge Type Features

Edge type is derived per observing seat from `owner[src]`, `owner[dst]`,
`neighbors[src,k]`, destination state, and current outflow state. It does not
need to be stored in the reducer.

Initial directed edge categories:

| type | meaning |
|---|---|
| `mine_to_enemy` | owned source points at an enemy cell |
| `mine_to_neutral` | owned source points at a neutral cell |
| `mine_to_friendly_relay` | owned source points at friendly cell with active outflows |
| `mine_to_friendly_sink` | owned source points at friendly MAX cell with no active outflows |
| `mine_to_friendly_fill` | owned source points at friendly below-MAX cell |
| `enemy_to_mine` | enemy outflow points at one of the seat's cells |
| `enemy_to_enemy` | other-seat pressure between non-owned cells |
| `blocked` | off-grid or dead destination |

Useful continuous edge features:

- source strength and headroom.
- destination strength and headroom.
- current `outflow[src,k]`.
- current `edge_pressure[src,k]`.
- inbound pressure at source and destination.
- destination active outflow count.
- destination enemy/neutral/friendly/dead flags.
- whether destination is frontier.
- distance from observer to source and destination.

## Voting Shape

The first version can be simple:

```txt
observer embedding
  + edge feature embedding
  + relative position / distance embedding
        -> vote_open_score(edge)
        -> vote_clear_score(edge)
        -> optional edge_value_channels(edge)
```

Votes are aggregated per directed edge:

```txt
open_score[src,k]  = weighted_sum(observer_votes_open)
clear_score[src,k] = weighted_sum(observer_votes_clear)
intent[src,k]      = open_score - clear_score
```

Weights should normalize for visibility count so central edges do not win only
because more observers can see them. Distance weighting is allowed; source and
destination observers can have stronger voices than third-hop observers.

## Design Guardrails

- Observer votes are advice; only edges whose source belongs to the acting seat
  are actuated in the first version.
- Aggregation must be position-invariant enough that central edges do not win
  from visibility count alone, and perimeter edges do not disappear because
  fewer cells can observe them.
- Deterministic local-flow targets are teachers, not law. Use them as soft
  pretraining or auxiliary signals so they teach edge vocabulary without
  freezing pulse timing.
- Keep the v2 action surface (`Set`, `Clear`, `No-op`) until the edge head is
  proven. Direct multi-edge gate updates would be a rules/trainer change, not
  just a representation change.
- Keep the current node-centric PPO path and non-learning heuristics as
  baselines until the edge-voting path beats them on named tiny arenas and big
  replay inspection.

## Multi-Signal Channels

Do not collapse the whole problem into one scalar too early. The edge policy
should expose several interpretable channels before final gating:

- `attack_flow`: pressure into enemy cells.
- `expand_flow`: pressure into neutral cells.
- `relay_flow`: pressure into friendly relays.
- `fill_flow`: pressure into friendly below-MAX cells.
- `sink_risk`: pressure into friendly dead-end MAX cells.
- `threat_in`: enemy pressure toward owned cells.
- `stored_pressure`: available pressure that could be released later.
- `front_break`: pressure likely to create a capture.
- `follow_through`: pressure that can exploit a newly opened/captured front.

The final action can be learned from these channels, but the channels should be
logged separately. This keeps the system inspectable and lets training tell the
difference between "good flow exists" and "now is the right time to release."

## Preserve Pulse Emergence

The policy must not become "always maximize current flow." That would destroy
charging and multi-pulse timing.

Required design constraints:

- Keep `No-op` / hold as a real outcome.
- Use a deadband around zero intent so current valve state can persist.
- Do not reward raw total flow without context.
- Reward enemy delivery, captures, relay usefulness, and low sink waste.
- Treat stored usable pressure as information and possibly mild value, not as
  an automatic penalty.
- Avoid a hard-coded global "pulse now" action in the first version.

The learner should discover timing from persistent valves and delayed payoff,
not from a scripted pulse controller.

## Training Paths

### Deterministic Local Flow Pretraining

Train edge votes against an algorithmic local-flow target:

- enemy/neutral edges are positive when release is locally useful.
- friendly relays are positive when they continue a route toward useful sinks.
- friendly dead-end MAX sinks are negative.
- blocked edges are negative.
- below-MAX friendly fill edges are context-dependent.

This teaches the model the physics vocabulary cheaply.

### Auxiliary Edge-Type Prediction

Add small auxiliary losses to predict derived edge categories and local channel
values. This can make the representation edge-aware without making those
categories the final policy.

### RL for Timing and Tradeoffs

After pretraining, use PPO or another actor-critic only for the choices that
remain non-deterministic:

- hold vs release.
- focus vs fanout.
- attack vs expand.
- defend vs finish.
- clear stale route vs keep pressure staged.
- first break vs follow-through.

The RL action can still be low-level `Set/Clear/No-op`, but the perception and
auxiliary channels should make the local flow facts obvious.

## Metrics

Track policy behavior by edge channel, not just reward:

- active `mine_to_enemy` pressure.
- active `mine_to_neutral` pressure.
- active `mine_to_friendly_relay` pressure.
- active `mine_to_friendly_sink` pressure.
- enemy pressure per frontier cell.
- stored pressure behind frontier.
- release bursts: short-window changes in enemy-directed pressure.
- follow-through: enemy-directed pressure after capture.
- stale open slots with low or negative intent.

Pulse behavior should be measured as burst structure, not hand-authored as a
rule.

## Early Success Criteria

The first version is useful if it can prove these before a long PPO run:

- Edge categories are computed by a pure feature builder with tests around
  owner changes, dead cells, friendly relays, friendly sinks, and invalid slots.
- Replay/training logs expose edge-channel pressure and burst/follow-through
  metrics without changing game physics.
- A non-learning flow heuristic built from the same features beats inert/random
  baselines on tiny arenas and gives readable failures when it loses.
- An edge-aware model can predict edge categories/channels better than a trivial
  baseline before RL is asked to learn timing.
- PPO fine-tuning improves timing-sensitive metrics without collapsing into
  always-open flow.

## Implementation Status

The first implementation slice is wired, behind baseline-preserving surfaces:

- `python/flux_v2/edge_features.py` is the shared NumPy category/feature
  authority for `(G, S, N, K, F_edge)`.
- `python/scripts/train_v2.py` logs edge-channel metrics and still defaults to
  the node-centric model.
- `python/flux_v2/edge_flow.py` provides a non-learning edge-flow heuristic for
  tiny arenas and teacher/baseline use.
- `python/flux_v2/ppo.py::EdgeAwareActorCritic` is available via
  `scripts/train_v2.py --model edge`, with a separate default checkpoint path.
- `python/scripts/pretrain_v2_edge_aux.py` smoke-trains the edge-type/channel
  auxiliary heads before long PPO runs.

This is still a staging path, not a proven replacement for the baseline PPO.
The next frontier is auxiliary pretraining quality and a real `--model edge`
run on tiny arenas.

## First Implementation Slice

1. Add a pure derived edge-feature builder for `(G, S, N, K, F_edge)`.
2. Add logging/metrics for edge categories using existing replay/training
   state.
3. Build a non-learning edge-flow heuristic from the same features.
4. Add an edge-voting policy head behind a flag while keeping the current
   node-centric PPO path intact.
5. Pretrain or auxiliary-train on deterministic edge categories before running
   RL.

## Open Questions

- Should observers include only owned cells, all visible cells, or all cells
  within the GCN patch?
- Should final action selection open/clear multiple edges per AI tick or stay
  with one cell-level action at first?
- Should edge votes aggregate by sum, mean, attention, or max?
- Should pulse timing be learned purely from low-level gates, or should a later
  small mode head modulate thresholds without directly commanding pulses?

Related: [[v2-rules-one-pager]], [[v2-edge-pressure-state]],
[[v2-set-clear-actions]], [[v2-three-term-reward]], [[v2-training-runs]].
