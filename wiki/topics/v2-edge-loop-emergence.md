---
title: v2 edge-loop emergence (lightning diffusion modes)
kind: topic
first_seen: 2026-05-14
last_updated: 2026-05-14
status: active
---

## The observation

Both algorithmic solvers in [[v2-algorithmic-solvers]] produce
tree-shaped flow networks. BFS picks one parent per cell by definition.
Lightning ([`solver_lightning.compute_potential`](../../python/flux_v2/solver_lightning.py)
in its original max-aggregation form) does the same thing by construction:

```
pot[c] = max(intrinsic[c], γ · max(pot[d] for d in non-dead nbrs))
```

A `max` is a single-parent operator. Each cell inherits from one steepest
neighbor — at most one — and the induced "uphill arrow" graph is therefore
a tree (technically a forest of trees rooted at intrinsic sources).

A cycle a→b→c→a is, mathematically, **forbidden** by this operator.

That's a missed feature: loops are a much stronger pressure generator
than single-cell outflows. Around a 3-cell ring, the same cell that
emits pressure also *receives* pressure from the cell behind it; total
edge throughput on the ring is 3× a single-direction outflow. If the
solver could form rings, it would.

## The fix (no rollout required)

The natural intuition is "we'd need to simulate forward to know which
cells the pressure will keep circulating through." It turns out you
don't — the closed-form fixed point of a discounted Markov chain
**already gives you exactly that**.

Two new modes alongside `max`:

### `mode="sum"` — uniform Bellman

```
pot[c] = intrinsic[c] + γ · Σ_d ( (1/deg(d)) · pot[d] )
```

Each non-dead neighbor `d` of `c` contributes `pot[d]` weighted by `d`'s
share-per-outgoing-edge (uniform over `d`'s non-dead neighbors). This is
value iteration on a uniform stochastic transition: row-sum ≤ 1, γ < 1
contracts.

Cycles self-reinforce as the geometric series Σ_{n=0}^∞ γⁿ — exactly the
"future residual" of pressure circulating through the loop, summed to
infinity. No rollout, just a fixed-point iteration with the same cost as
the existing max version.

### `mode="sum_pw"` — pressure-weighted (rich-get-richer)

```
pot[c] = intrinsic[c] + γ · Σ_d ( w[d→c] · pot[d] )
w[d→c] = edge_pressure[d→c] / total_outflow[d]   (uniform 1/deg fallback when zero)
```

Same operator but the weights are **read from the current edge_pressure
field**. A cell that is already pumping in direction `d→c` contributes
more of its pot to `c` than a cell that isn't. This is the lightning-style
positive feedback: once any tiny cycle starts circulating pressure, the
diffusion amplifies it, which makes the solver more likely to keep the
outflow slots on along the cycle, which strengthens the circulation, etc.

The uniform-fallback handles the cold-start case (no pressure flowing
yet) — early-game `sum_pw` collapses to `sum`, and the symmetry breaks
once any flow appears.

## How loops actually form in practice

Max mode: never. The action rule (`pot[d] > pot[c]` strictly) plus the
single-parent operator is a hard fence against cycles.

Sum / sum_pw: cycles emerge when the potential field is "flat enough"
that several mutually-pointing cells along a ring all see slightly-higher
neighbor pot than themselves. The action rule (still strictly uphill at
the same `fanout_eps` threshold) admits these once the field stops
being strictly monotone — which is the operator's natural state once
contributions are summed rather than maxed.

For `sum_pw` specifically, the key dynamic is symmetry-breaking by
ambient noise (random action choice, intrinsic-source asymmetry) followed
by exponential amplification through the pressure-weighted feedback.

## Solver registration

`python/scripts/run_v2_solver.py` registers all three:

| seat name           | mode      |
|---------------------|-----------|
| `lightning`         | `max`     |
| `lightning_sum`     | `sum`     |
| `lightning_sum_pw`  | `sum_pw`  |

Use the existing `--seats` arg, e.g.:

```bash
python scripts/run_v2_solver.py --radius 6 --num-players 6 --games 24 \
  --seats lightning,lightning_sum,lightning_sum_pw,lightning,lightning_sum,lightning_sum_pw \
  --write-replay
```

Replays land in `public/v2/replays/` with the seat → solver mapping in
the FLXR metadata; the v2 displayer renders them like any other v2
replay.

## Head-to-head (R=6, P=6, 4000 ticks, ai_period=5)

Four runs, all on the same board generator with seat-reachability guarantee
described in [[v2-algorithmic-solvers#board-generation-seat-reachability-guarantee]].

### Run 1 — all six seats `lightning_sum` (seed 700, 8 games)

| outcome | count | mean ticks |
|---------|------:|-----------:|
| decisive | 8 / 8 | 1501 |
| stalemate | 0 / 8 | — |

Self-play is **fully decisive at 4000 ticks**. Sum-mode breaks symmetry
fast enough to resolve — it does not deadlock the way one might fear from
a smoother field. Seat 4 won 2× early; final win distribution was a flat
2/1/0/2/2/1 — no seat-position artifact.

### Run 2 — all six seats `lightning_sum_pw` (seed 701, 8 games)

| outcome | count | mean ticks |
|---------|------:|-----------:|
| decisive | 0 / 8 | — |
| stalemate | 8 / 8 | 4000 |

Pure pressure-weighted self-play **deadlocks every game** at the tick
cap. Mean dominance 0.50 with 3–4 seats alive at end. This is the loop
mechanism working *too well* — once each seat establishes its own
circulating-pressure interior, the rich-get-richer feedback locks the
configuration in. Nobody can crack anybody else's loops, because the loops
sustain themselves from inside the territory. **Look at the
`solver_v2_lightning_sum_pw_*.flxr` replay** in the v2 displayer — the
visual signature is exactly the cycle structure the design predicted, and
it is in fact what's preventing decisive play.

### Run 3 — 3-way mix: `lightning, lightning_sum, lightning_sum_pw` × 2 seats each (seed 702, 24 games)

| solver              | seats | wins | win share |
|---------------------|------:|-----:|----------:|
| `lightning_sum`     | 2     | 15   | **62.5%** |
| `lightning`         | 2     | 9    | 37.5%     |
| `lightning_sum_pw`  | 2     | 0    | 0.0%      |

`lightning_sum` is the overall best solver — better than the original
max-mode lightning. The sum operator's diffuse "future residual" makes
relay routing more robust than max's strict steepest-uphill, and the
extra throughput shows up as a measurable win rate gap. `lightning_sum_pw`
is **dead last** in mixed company — its loops are defensive, not
offensive, so it cannot project force into opponents who attack with
directed pressure.

### Run 4 — alternating `lightning` vs `lightning_sum_pw`, 3 seats each (seed 703, 24 games)

| solver              | seats | wins | win share |
|---------------------|------:|-----:|----------:|
| `lightning`         | 3     | 23   | **95.8%** |
| `lightning_sum_pw`  | 3     | 0    | 0.0%      |
| (stalemate)         | —     | 1    | 4.2%      |

Direct refutation of pressure-weighted in adversarial play. Max-mode
sweeps. The 1 stalemate is the same dynamic as Run 2: when two adjacent
`sum_pw` seats happen to interlock loops before max can break in, the
position freezes.

### Summary

The hypothesis ("loops are stronger pressure generators") was correct
mechanically — `sum_pw` does form them, and the visual evidence is in
the replay. But the **strategic implication was inverted**: cycle
pressure is *defensive infrastructure*, not offensive throughput.
Against passive opponents the loops never get tested; against active
attackers the loops cost slots that would otherwise be on the frontier.

`lightning_sum` (the loop-permitting operator *without* the
edge-pressure feedback) ended up being the genuine improvement — it
beats the original max-mode by ~6 games out of 24, despite using the
same action rule. The geometric-series Bellman aggregation is just a
better local heuristic for relay routing.

See `wiki/log.md` (entry: lightning sum / sum_pw modes) for the run log
and replays:

- `public/v2/replays/solver_v2_lightning_sum_*.flxr` — Run 1 game 0
- `public/v2/replays/solver_v2_lightning_sum_pw_*.flxr` — Run 2 game 0 (the deadlock — watch the rings)
- `public/v2/replays/solver_v2_lightning+lightning_sum+lightning_sum_pw_*.flxr` — Run 3 game 0
- `public/v2/replays/solver_v2_lightning+lightning_sum_pw_*.flxr` — Run 4 game 0

## Tradeoffs

- **Sum dilutes focus on the weakest target.** Max's biggest visual win
  was the "channel everything toward the dying enemy" effect. Sum spreads
  the pull more broadly. `sum_pw` partially recovers focus via the
  pressure feedback, but only after some flow has already established.
- **Sum has higher pot magnitudes.** `intrinsic + γ/(1-γ)` rather than
  `intrinsic / (1 - γ^d)`. `fanout_eps` thresholds may need a separate
  tuning pass per mode.
- **Loops cost outflow slots.** A cell on a 3-loop has at least 2 of its
  6 outgoing slots committed to ring-maintenance. If that displaces an
  attack-frontier slot, the cell becomes a worse frontier piece. Whether
  this trade pays depends on board geometry — see results.
- **The action rule didn't change.** Loops emerge from the field shape
  alone. A future variant could explicitly bias the action rule toward
  cycle-closure (e.g., set an outflow toward `d` when `d` already sets
  an outflow toward a cell with potential > yours), at the cost of
  losing the "purely local field → greedy steepest-ascent" simplicity.

## Open questions

- **Loop-pressure → wins?** Empirically *no* for `sum_pw` — the loops
  hold their interior but can't project. The improvement was actually
  from `sum`, which doesn't need loops to win.
- **Hybrid: max-mode frontier, sum_pw interior?** Plausibly the best of
  both — keep the strict-uphill attack focus on the boundary, run the
  pressure-weighted feedback only on cells with ≥3 friendly neighbors.
  Untested.
- **Curriculum signal for PPO.** If `sum_pw` reliably forms loops the
  current PPO policy never discovers, that's a hint the policy is
  missing the inductive bias. See [[v2-training-runs]]; an auxiliary
  loss matching the policy's edge-flow output against a `sum_pw` field
  could seed the topology cheaply.
- **Does longer-game tuning save sum_pw?** The 4000-tick stalemates in
  Run 2 might resolve if loops gradually degrade under sustained
  pressure. Worth one larger run (12000 ticks) to confirm vs. genuinely
  stable equilibria.
- **Replace `lightning` with `lightning_sum` as the default baseline?**
  62.5% > 37.5% in mixed play is a clear gap. If a focused
  `fanout_eps` / `γ` tune on sum-mode widens it, switching the default
  baseline is justified.

Related: [[v2-algorithmic-solvers]],
[[decisions/v2-edge-pressure-state]], [[v2-edge-voting-policy]],
[[v2-training-runs]].
