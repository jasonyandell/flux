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

## How loops actually form in practice (post-experiment)

**The diffusion change was necessary but not sufficient.** Sum/sum_pw
smooth the field, but the strict-uphill action rule (`pot[d] > pot[c]`
gated by `fanout_eps`) is still tree-only by transitivity: you can't
have `pot[a] < pot[b] < pot[c] < pot[a]`. What `sum_pw` actually
produces in replay is *bidirectional feeding* between near-equal-pot
cells (a↔b 2-cycles) and Y-shaped confluences — not the directed
3-cycles the hypothesis predicted.

The fix is structural, not field-based. See `mode="loop"` below.

### `mode="loop"` — structural curl rule (no potential field)

Hex grid geometry: a cell c's neighbors at slots k and k+1 (mod 6) are
themselves mutually adjacent (this is what makes hex grids triangular at
every scale). Every such triangle has a fixed slot-parity — from each
of its three corners, the slot pair used is *either* both-even-k *or*
both-odd-k. So restricting the rule to k ∈ {0,2,4} fills the even-k
triangles uniformly with directed 3-cycles, and the odd-k triangles
stay empty.

Why even-only? The v2 reducer's "no friendly bidirectional flow"
invariant ([`step.apply_actions`](../../python/flux_v2/step.py)) clears
one side whenever both directions of an edge get set on the same tick.
Opposite slots are `(k, k+3)` — pairs of opposite parity. So if all
"loop relay" outflows live on even-k, every back-edge candidate is on
odd-k and is never set, the invariant never triggers, and clean directed
3-cycles survive.

Concretely (verified by sanity test on R=3 all-friendly board):

```
For cell c, for k in {0, 2, 4}:
    if nb[c,k] friendly AND nb[c,(k+1)%6] friendly:
        set outflow slot k on c
```

Plus the original frontier-attack rule on non-friendly neighbors.
Result: every interior cell has outflows on exactly slots `{0, 2, 4}`
(a "triskelion"), and every even-parity friendly triangle has a closed
a→b→c→a CCW 3-cycle. Visible in
`solver_v2_lightning_loop_20260515T015923.flxr` — the visual pattern is
dramatically different from `sum_pw`.

## Solver registration

`python/scripts/run_v2_solver.py` registers all four:

| seat name           | mode      | what it does                                       |
|---------------------|-----------|----------------------------------------------------|
| `lightning`         | `max`     | original tree-only diffusion + strict-uphill relay |
| `lightning_sum`     | `sum`     | uniform Bellman diffusion + strict-uphill relay    |
| `lightning_sum_pw`  | `sum_pw`  | edge-pressure-weighted Bellman + strict-uphill     |
| `lightning_loop`    | `loop`    | structural curl rule on hex triangles (no field)   |

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

### Run 5 — `lightning_loop` self-play (seed 800, 6 games)

| outcome   | count | mean ticks |
|-----------|------:|-----------:|
| decisive  | 4 / 6 | 1541       |
| stalemate | 2 / 6 | 4000       |

The triskelion pattern is unambiguously visible from frame 0 in
`solver_v2_lightning_loop_20260515T015923.flxr`. Loops resolve eventually
but with a longer mean tick count than `sum` (2361 vs 1501 overall) —
the committed slot budget for circulation slows down attack focus.

### Run 6 — alternating `lightning` vs `lightning_loop` (seed 801, 24 games)

| solver            | seats | wins | win share |
|-------------------|------:|-----:|----------:|
| `lightning`       | 3     | 17   | **70.8%** |
| `lightning_loop`  | 3     | 7    | 29.2%     |

Original max-mode beats the structural loop rule decisively.

### Run 7 — `lightning_sum` vs `lightning_loop` (seed 802, 24 games)

| solver            | seats | wins | win share |
|-------------------|------:|-----:|----------:|
| `lightning_sum`   | 3     | 18   | **75.0%** |
| `lightning_loop`  | 3     | 3    | 12.5%     |
| (stalemate)       | —     | 3    | 12.5%     |

`sum` keeps its top spot. `loop` is the weakest of the four.

### Summary

Three sequenced experiments, three sharpening lessons:

1. **Field smoothing alone doesn't make loops.** Sum/sum_pw smooth the
   potential field but the strict-uphill action rule is still tree-only
   by transitivity. `sum_pw` replays show 2-cycle feeding and
   Y-confluences, not the predicted directed 3-cycles.
2. **Structural rules do.** The even-k slot rule on hex triangles
   produces clean directed 3-loops with no field at all, no
   bidirectional invariant conflicts, and a striking triskelion visual
   signature. Hypothesis confirmed at the structural level.
3. **But loops aren't free.** Every interior cell commits 3 outflow
   slots to circulation. That's bandwidth diverted from attack focus.
   In head-to-head, `loop` loses to both `lightning` (29.2%) and
   `lightning_sum` (12.5%).

The genuine win of this experiment arc was `lightning_sum` —
non-edge-weighted Bellman aggregation, which beats original max-mode
by ~25 percentage points without any cycle structure. The geometric-
series field is just a better local relay heuristic.

The structural loop rule is the right tool *if* you can pay the slot
cost — e.g., late-game when the frontier is small and attack saturation
isn't the bottleneck. A frontier-aware hybrid (max-mode for cells with
≤3 friendly neighbors, loop rule for cells with ≥4) is the natural
next experiment.

See `wiki/log.md` (entry: lightning sum / sum_pw modes) for the run log
and replays:

- `public/v2/replays/solver_v2_lightning_sum_*.flxr` — Run 1 game 0
- `public/v2/replays/solver_v2_lightning_sum_pw_*.flxr` — Run 2 game 0 (2-cycle feeding / Y-confluences — *not* true 3-cycles)
- `public/v2/replays/solver_v2_lightning+lightning_sum+lightning_sum_pw_*.flxr` — Run 3 game 0
- `public/v2/replays/solver_v2_lightning+lightning_sum_pw_*.flxr` — Run 4 game 0
- `public/v2/replays/solver_v2_lightning_loop_*.flxr` — Run 5 game 0 (**the triskelion pattern — directed 3-loops on every even-parity triangle**)
- `public/v2/replays/solver_v2_lightning+lightning_loop_*.flxr` — Run 6 game 0 (loop vs max head-to-head)
- `public/v2/replays/solver_v2_lightning_loop+lightning_sum_*.flxr` — Run 7 game 0

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
- **Frontier-aware loop hybrid.** Apply the loop rule only when a cell
  has ≥4 friendly neighbors (deep interior, where the slot budget for
  circulation isn't displacing attack slots), and use `sum` relay
  elsewhere. Untested.
- **Test the structural-loop visual.** The `solver_v2_lightning_loop_*`
  replay should make the triskelion pattern obvious — a sanity check
  that the geometry argument matches what's drawn on screen, before
  trusting the hybrid idea above.

Related: [[v2-algorithmic-solvers]],
[[decisions/v2-edge-pressure-state]], [[v2-edge-voting-policy]],
[[v2-training-runs]].
