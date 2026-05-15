---
title: v2 — non-dead cells must form one connected component
kind: decision
first_seen: 2026-05-14
last_updated: 2026-05-14
status: active
---

## What

Every v2 board the BFS solver, the lightning solver, and the PPO trainer
operate on must satisfy two stacked invariants:

1. **Live-subgraph connectivity.** Every non-DEAD cell is reachable from every
   other non-DEAD cell by walking through non-DEAD neighbors.
2. **Seat reachability with bounded distance.** Every seat is reachable from
   every other seat through non-DEAD cells, and the maximum BFS hop distance
   between any pair of seats is `≤ 4 · radius` (≈ 2× the empty-hex diameter).

Lots of DEAD cells is fine — even up to ~50–70 % of the board. Isolated live
pockets are not, even when those pockets contain no seats: they sit dormant,
immune to capture, useless to the solvers, and they break assumptions the
distance-to-frontier BFS and the lightning potential field rely on.

## Why

The v2 [[v2-algorithmic-solvers|algorithmic solvers]] and the PPO policy all
assume *one* connected playfield:

- **BFS solver** ([[v2-algorithmic-solvers]]): `_frontier_distance` BFS only
  traverses owned cells, but `_ideal_outflow_for_cell` infers attack and relay
  candidates from cell-relative classifications that quietly assume the live
  subgraph is one piece. A live pocket with no enemy/neutral neighbor reachable
  through dead-free traversal would sit at the `BIG = 10_000` distance sentinel
  forever and emit only frontier attacks on its boundary — never relays.
- **Lightning solver** ([[v2-algorithmic-solvers]]): the potential field
  diffuses through non-DEAD neighbors. An isolated live pocket would never see
  the global attractor and would converge to its own local intrinsic field,
  losing the slime-mold property that makes lightning interesting.
- **PPO policy** ([[ppo-gnn]] / [[v2-trainer-displayer]]): the GCN aggregates
  3-hop neighborhoods through live edges. Disconnected pockets become hidden
  silos with no gradient flow between them.

The bounded-distance invariant is separate from raw connectivity. A snaking
50 %-dead board can be technically connected but have seats 25+ hops apart,
which is more than the pressure can propagate in a single 2000-tick game at
period 5. Effectively two parallel games on the same board. The `4·R` cap is
permissive enough to admit most random samples and tight enough to reject
the worst snake-corridor layouts.

## How

Two layers of enforcement, both in `python/flux_v2/graph.py`:

### Layer 1 — sampler (`random_seat_and_dead`)

When called with `neighbors=<board.neighbors>` (the default for both the PPO
trainer and the solver runner), the sampler is greedy: each candidate dead
cell is provisionally accepted and rolled back if `_live_subgraph_connected`
would fail. A postcondition assertion re-checks at the end. If the layout runs
out of safe candidates, the sampler returns however many dead cells it could
place (potentially fewer than requested).

This guarantees invariant (1) by construction.

### Layer 2 — runner-side mode (`run_v2_solver.py`)

The solver runner gates on a stricter contract via `--connect-mode`:

- **`retry`** (default). Generates connectivity-preserving boards as above,
  then runs `seats_mutually_reachable` and `max_seat_pair_distance` checks. If
  the max pair distance is `> 4 · radius`, the board is rejected and retried
  up to 200 times. The default mode for most runs — most uniform spatial
  character, rejects the snaky-corridor cases.
- **`carve`**. Samples dead cells *without* the live-subgraph guard
  (uniform random), then calls `carve_seat_connectors` to revive dead cells
  along the shortest bridge between every pair of live components — not just
  seat-bearing ones. Preserves the spatial character of pure uniform-random
  dead-cell distributions; a thin carved channel ties orphan islands back to
  the main island. Typically only a handful of cells get carved.

The `carve_seat_connectors` bridge-finder uses 0-1 BFS (cost 0 through live
cells, cost 1 through dead) so it always finds the minimum number of cells
that need reviving. Each non-main component picks its seat (if any) or the
smallest-id cell as the source for the bridging BFS; this tends to produce
seat-to-seat corridors when seats are spread across components.

### Trainer

`scripts/train_v2.py::make_batched_initial_state` calls `random_seat_and_dead`
with `neighbors=...`, so the trainer gets Layer 1 for free. Seat-distance
gating is not currently applied per-rollout because the trainer already
samples G boards in parallel and dilutes any single bad-shape board's
influence. See [[v2-training-runs]].

## Solvers as consumers

Both `flux_v2/solver.py` (BFS) and `flux_v2/solver_lightning.py` declare the
invariant as a documented precondition. They do not re-check it at runtime —
the BFS distance computation runs every AI tick and an additional connectivity
check would double its work. The runner / trainer is the right place to
enforce it.

## Tests

`python/tests/test_v2_solver_connectivity.py` covers:

- Retry mode produces connected boards across many seeds at multiple board
  sizes (radius 6 / 9, dead counts 10 / 40 / 80).
- Carve mode produces connected boards even when starting from uniform-random
  dead-cell samples that contain non-seat-bearing islands.
- `seats_mutually_reachable` correctly rejects an islanded seat.
- `carve_seat_connectors` bridges every live component (the regression case
  that motivated the v2-board-connectivity decision: a wall splitting the
  board where one side has no seats was previously left isolated).
- `max_seat_pair_distance` returns `-1` when seats are unreachable.
- BFS and lightning solvers both emit non-NOOP actions on connected boards.

## Related

- [[v2-algorithmic-solvers]] — the BFS + lightning consumers.
- [[v2-edge-pressure-state]] — the underlying state these solvers act on.
- [[v2-trainer-displayer]] / [[v2-training-runs]] — the PPO consumer.
