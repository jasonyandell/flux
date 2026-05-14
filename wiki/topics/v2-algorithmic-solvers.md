---
title: v2 algorithmic solvers
kind: topic
first_seen: 2026-05-13
last_updated: 2026-05-13
status: active
---

## What this page is

Hand-written, non-NN solvers for v2 — useful as baselines for PPO, sanity
checks on game dynamics, and a parallel design track exploring how much
structure the v2 ruleset will give up to purely local heuristics.

Two solvers live in `python/flux_v2/`:

- **BFS solver** (`solver.py`) — frontier-aware greedy. Each owned cell sets
  outflows toward enemy / neutral neighbors (attack) and toward friendly
  neighbors strictly closer to the nearest non-friendly (relay). Dead-end
  MAX sinks excluded. One action per cell per AI tick, priority
  `SET attack > SET relay > CLEAR stale > NOOP`.
- **Lightning solver** (`solver_lightning.py`) — purely local potential
  field, no global computation. Each cell stores a scalar `pot[c]` computed
  by Bellman-style diffusion from intrinsic sources:

  ```
  enemy cell      pot ← weak_bonus × (1 - strength/MAX)        (weakest glow brightest)
  neutral cell    pot ← expand_bonus                           (constant pull)
  friendly cell   pot ← defense_bonus × (inbound_enemy / MAX_EDGE)   (default 0)
  dead cell       0                                            (walls)
  pot[c] ← max(intrinsic[c], γ · max_neighbor(pot))            (iterated to convergence)
  ```

  Frontier cells always attack any non-friendly neighbor (the air-breakdown
  rule — pressure discharges at any exposed boundary). Interior cells
  relay along the steepest uphill gradient toward weak targets.

Runner: `python/scripts/run_v2_solver.py`. Per-seat solver assignment via
`--seats bfs,lightning,bfs,lightning,bfs,lightning`. Writes a `.flxr` v2
replay to `public/v2/replays/` when `--write-replay` is passed; metadata
records the seat → solver mapping.

## Why we want these

- **Baseline for PPO.** A reasonable hand-written opponent the trained
  policy should beat. Right now PPO plays self-play only; it has no fixed
  reference to anchor improvement.
- **Sanity check on game dynamics.** Solver self-play surfaces capture
  rates, game lengths, and stalemate patterns without confounding by a
  learning loop.
- **Parallel design track.** Lightning is the "slime-mold" idea from
  [[v2-training-runs]] in its purest form — emergent global structure from
  purely local rules, no planning, no priority queue. If it competes with
  PPO at all, that's strong evidence the game's local geometry carries
  most of the strategy.

## Bug fix shipped alongside

While testing the solver, found that the pure reducer `python/flux_v2/step.py`
never captured neutral cells: `is_friendly_d` / `is_enemy_d` were gated by the
receiver's `is_alive` mask (`owner ≥ 0`), which is False for NEUTRAL receivers,
so `pressure_in_enemy[neutral]` was always 0 and `will_capture_neutral` was
unreachable. The MLX path (`mlx_step.py`) doesn't have this gate — it's
correct. The parity test only ran 25 ticks on a board with no chained
pressure, so it never built up enough overflow to capture a strength-10
neutral. Fix replaced `is_alive` with `not_dead` in the enemy gate;
regression test added in `tests/test_v2_step.py::test_neutral_capture`.

## Head-to-head result (initial)

Configuration: radius=6, 6 players, 10 dead cells, `max_ticks=4000`,
`ai_period_ticks=5`. Alternating seats
`bfs,lightning,bfs,lightning,bfs,lightning`, 40 games (seed 200):

```
total: 40 games, mean ticks 1331
  bfs       (3 seats): 21 wins
  lightning (3 seats): 19 wins
```

Effectively tied. Earlier iterations of the lightning rule were much worse
— going from "steepest-uphill only" (0-8 vs BFS) to "always attack
frontier neighbors + relay interior by gradient" closed the gap entirely.
The hybrid keeps the emergent property of focus-toward-weakest while not
giving up neutral territory.

## Open knobs

The lightning solver has three intrinsic-source weights (`weak_bonus`,
`expand_bonus`, `defense_bonus`) and a `γ` decay. Current values come from
a single tuning pass — there's likely a better operating point. Future
experiments:

- **Tune by tournament.** Sweep `expand_bonus` ∈ {0.3, 0.6, 0.9} and
  `γ` ∈ {0.7, 0.85, 0.95} on a large game sample.
- **Defense-aware.** `defense_bonus > 0` should reinforce threatened
  cells, but in the first sweep it just dragged attack momentum back.
  Try smaller values (0.1–0.3) and see if the "lightning rod" effect
  emerges.
- **Compare against PPO.** Plug each solver in as opponents during PPO
  rollout (replace some seats' policy actions with `solver_actions`) and
  measure win rate. That's the real baseline.

## Board precondition

Both solvers require the v2 board-connectivity invariant: every non-DEAD
cell reachable from every other non-DEAD cell, every seat reachable from
every other seat, max seat-pair distance ≤ 4·R. Lots of dead cells is
fine — isolated live pockets are not. See [[decisions/v2-board-connectivity]]
for the rule, the two enforcement layers (sampler guard + runner retry/carve),
and the failure modes each solver would exhibit on a disconnected board.

`scripts/run_v2_solver.py --connect-mode retry` (default) regenerates boards
until the seat-distance check passes; `--connect-mode carve` revives a thin
bridge of dead cells along the shortest path between live components, which
preserves the spatial character of pure uniform-random dead-cell samples.

Related: [[v2-training-runs]] (PPO track), [[decisions/v2-edge-pressure-state]],
[[decisions/v2-three-term-reward]], [[decisions/v2-board-connectivity]].
