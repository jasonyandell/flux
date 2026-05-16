---
title: v2 ML gameplay opportunities
kind: topic
first_seen: workspace
last_updated: workspace
status: active
---

## What this page is

An ML-scientist read of where learning is most likely to improve flux v2
gameplay, grounded in the current solver and PPO evidence. It is not a runbook;
for launches use [[v2-training-runs]]. For the current solver frontier use
[[v2-vectorized]] and [[v2-overnight-research]].

## Bottom line

The strongest near-term ML role is not "replace the best solver from scratch."
The best solvers have exposed a useful prior: long-range integrated pressure
fields win, while global pulse gates, edge-attention release heads, and
defensive loop storage lose or stalemate. Learning should start from that
mechanistic prior and improve the parts the solver cannot tune locally:
timing, opponent-specific tradeoffs, defense, finishing, and when to deviate
from the long-field flow.

## Current evidence

- PPO/GNN has learned value structure and some decisive behavior, but it has
  not beaten the best big-bag solvers under the current methodology
  ([[v2-training-runs]], [[ppo-gnn]]).
- Matched-pair solver work found the robust top mechanism:
  long-horizon `sum` aggregation, especially `sum_long` / `wave_long`
  ([[v2-overnight-research]]).
- The decisive mechanistic clue is that `gamma=0.94` helps `sum` but hurts
  `max`: integrated aggregators benefit from far-field information, while
  single-winner aggregators turn longer reach into noise
  ([[questions/open]]).
- The edge-voting path is partially implemented but not yet proven
  ([[v2-edge-voting-policy]]). It points in the right direction if it uses
  solver structure as teacher and diagnostics rather than expecting PPO to
  rediscover local flow vocabulary unaided.

## Best ML opportunities

### Distill the solver before doing RL

Use `lightning_sum_long` or `wave_long` as a behavioral teacher, then ask RL to
improve beyond it. Pure PPO from a random policy spends too much budget learning
basic flow physics that the solver already knows.

Candidate experiment:

1. Behavior-clone `lightning_sum_long` with `--pretrain-solver
   lightning_sum_long`.
2. Fine-tune in PPO self-play with some fixed solver seats masked from loss.
3. Evaluate trained-vs-solver with matched-pair seat rotation, not raw win
   rates.

Success criterion: the learned policy keeps solver-level expansion speed while
improving finish rate, defense, or late-game cleanup on same-board pairs.

### Learn residuals over the long-field solver

The best baseline may be a hybrid: solver proposes the default edge intent,
the model learns a residual to open, clear, or hold when local heuristics are
wrong. This is lower variance than asking the policy to emit all 13 actions
from scratch.

Useful residual targets:

- hold pressure before an exposed break;
- clear stale friendly sinks;
- defend when enemy pressure is converging;
- follow through after capture;
- stop over-fanning when a focused kill is available.

### Make edge representation the learning surface

The node-centric policy sees aggregate inbound/outbound pressure. The game is
actually a persistent directed-edge system. A useful learner should score
candidate edges and expose interpretable channels before collapsing to
Set/Clear/No-op.

The current [[v2-edge-voting-policy]] direction is promising if it keeps:

- visibility-normalized aggregation, so central cells do not win by observer
  count alone;
- auxiliary edge-category/channel losses, so the network learns the physics
  vocabulary before RL asks for timing;
- logs for attack, expansion, relay, sink, threat, stored pressure, release
  burst, and follow-through channels.

### Use tournaments as supervised datasets

The vectorized runner makes matched-pair tournaments cheap enough to generate
preference data. Instead of treating solver tournaments only as scoreboards,
extract decision states where two policies diverge and the board-coherent
winner is known.

That enables:

- pairwise preference learning over actions or edge intents;
- contrastive "why did sum_long beat max here?" datasets;
- active data collection from close states rather than millions of routine
opening moves.

### Train on curricula that preserve game pressure

Self-play parity is too easy: all seats can become equally mediocre. Fixed
opponent seats already exist in `train_v2.py`; use them deliberately.

Reasonable ladder:

1. clone and beat `bfs`;
2. beat `lightning` / max;
3. beat `lightning_sum`;
4. beat `lightning_sum_long`;
5. mixed league with stale checkpoints and solver seats.

The evaluation unit should be matched-pair regret against the current solver
champion, not only mean reward.

## Watch-outs

- Do not trust single-direction tournament win rates near 50/50. The measured
  seat-position floor is about 6 percentage points.
- Do not over-index on terminal rewards. With `gamma=0.99` and long games,
  terminal bonuses are nearly invisible to early decisions.
- Do not reward raw activity. Persistent valves make "active edge count" a
  misleading objective.
- Do not assume larger networks help. Hidden=64 collapsed quickly in prior
  PPO runs; representation and curriculum look more important than capacity.
- Treat [[todo]] cautiously. Some entries mention regen-flow or older run
  framing that appears stale relative to the current v2 vectorized and
  big-bag solver pages.

## Next experiments

1. Rerun the matched-pair solver hierarchy on the vectorized code, because
   the current official rankings predate the semantic deltas in
   [[v2-vectorized]].
2. Behavior-clone `lightning_sum_long`, then PPO fine-tune against fixed solver
   seats. Evaluate with matched pairs.
3. Run the edge-aware auxiliary pretrain until edge category/channel accuracy
   is clearly above trivial baselines, then try the same clone plus PPO path
   with `--model edge`.
4. Build a residual-policy prototype around long-field solver suggestions.
   Even a small residual head is a sharper test than another from-scratch PPO
   run.
