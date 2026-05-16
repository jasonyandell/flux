---
title: v2 edge-loop emergence (lightning diffusion modes)
kind: topic
first_seen: 2026-05-14
last_updated: 2026-05-15
status: active
---

## Note on rankings (2026-05-15)

This page's "final ranking" of `sum > attn ≈ sum > max > loop > sum_pw` was
the May-14 conclusion from single-direction tournaments. Overnight matched-
pair experiments the next day ([[v2-overnight-research|v2-overnight-research]])
established a 6pp seat-bias noise floor that invalidated several of the
close gaps here. The May-15 non-throttled ranking was:

> `wave_long` > `sum` > `bfs` ≈ `max` >> `attn` >> `pulse` / `pulse_stagger`

`attn` did *not* hold its tie with `sum` once seat-bias was removed —
matched-pair `sum` beat `attn` 87-6 over 100 games. Keep this page for the
mechanism story (uniform-Bellman vs max-aggregation, the structural 3-loop
curl, attention as reservoir-with-release) and the PPO reward-shape deep
dive; treat its head-to-head tables as historical.

## Update (2026-05-15 PM)

A new champion sits above the ranking on this page: `lightning_sum_throttled`
(sum-mode potential field with a top-1 desired-slot cap per cell). It beats
both vanilla `lightning_sum` (9/9 coherent, p≈0.004) and `bfs` (7/7, p≈0.016)
at R=25 P=12 40%-dead, 12000-tick matched-pair eval. The active mechanism is
likely waste-reduction plus loop-aware-potential, not the
"commitment-by-construction" that motivated it. See
[[v2-temporal-strategy]] for the framing, validation results, and
implementation. The ranking above describes the *non-throttled* family;
throttled sits above all of them in measured head-to-head.

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

`python/scripts/run_v2_solver.py` registers all five:

| seat name           | mode      | what it does                                              |
|---------------------|-----------|-----------------------------------------------------------|
| `lightning`         | `max`     | original tree-only diffusion + strict-uphill relay        |
| `lightning_sum`     | `sum`     | uniform Bellman diffusion + strict-uphill relay           |
| `lightning_sum_pw`  | `sum_pw`  | edge-pressure-weighted Bellman + strict-uphill            |
| `lightning_loop`    | `loop`    | structural curl rule on hex triangles (no field)          |
| `lightning_attn`    | `attn`    | 2-head: ATTACK (max-pot) + LOOP (curl) with frontier-tilt |

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

### Run 8 — `lightning_attn` self-play (seed 900, 6 games)

| outcome   | count | mean ticks |
|-----------|------:|-----------:|
| decisive  | 6 / 6 | 1485       |
| stalemate | 0 / 6 | —          |

**No deadlock.** Unlike `sum_pw` (8/8 stalemates) and `loop` (2/6
stalemates), the attention mixing breaks the storage trap — interior
cells maintain loops, but as the frontier shifts they tilt forward and
release pressure. Self-play resolves at sum-comparable tempo.

### Run 9 — alternating `lightning_sum` vs `lightning_attn` (seed 901, 24 games)

| solver            | seats | wins | win share |
|-------------------|------:|-----:|----------:|
| `lightning_sum`   | 3     | 12   | 50.0%     |
| `lightning_attn`  | 3     | 12   | 50.0%     |

**Dead even** with the previous champion on the first hand-designed
attempt. That's a meaningful signal — `sum` beat `lightning` by 25
points in mixed play and `loop` by 62 points; pulling even with `sum`
on the first try, without any tuning of `deep_threshold` or
`relay_thresh`, says the architectural shape is right.

### Run 10 — 3-way `lightning` / `lightning_sum` / `lightning_attn` × 2 seats (seed 902, 24 games)

| solver            | seats | wins | win share |
|-------------------|------:|-----:|----------:|
| `lightning_sum`   | 2     | 11   | **45.8%** |
| `lightning_attn`  | 2     | 8    | 33.3%     |
| `lightning`       | 2     | 5    | 20.8%     |

`sum` still leads in the 3-way, but `attn` decisively beats the
original `max` (33.3% vs 20.8%) — the attention solver is the second
best of the four. The full ordering across all experiments is now:

`sum` > `attn` ≈ `sum` > `max` > `loop` > `sum_pw`

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

## The attention head — reservoir *with* release

User intuition that drove this branch: *"the pressure should be a
reservoir rather than an ultimatum."* The structural-loop solver banks
pressure perfectly but never spends it; the gradient solvers spend it
the moment it arrives. Neither shape is "shots on goal from a stocked
backline."

The fix is to run *both* rules simultaneously and let them share each
cell's 6-slot budget. That's exactly what attention is for: soft
routing where multiple heads coexist with per-cell mixing weights.

### The architecture (hand-designed Q/K, no learning)

**Two heads** evaluated per cell c, per outgoing slot k:

```
attack_score[c,k] = max(0, pot[nb[c,k]] - pot[c])   # max-mode gradient
loop_score[c,k]   = 1 iff k ∈ {0,2,4} and slots k, k+1 both friendly
                                                    # even-k curl
```

**Per-cell mixing weight α(c)** from frontier-distance BFS through
friendlies to the nearest non-friendly-alive cell:

```
α(c) = clip(frontier_dist[c] / deep_threshold, 0, 1)
```

with `deep_threshold = 2.0` as a default.

  α = 0 at the frontier      →  pure ATTACK head (no loops, all gradient)
  α = 1 deep interior        →  pure LOOP head (triskelion, no leak)
  α intermediate             →  blend — loops *tilt* forward as the
                                frontier gets closer

**Combined score per slot**:

```
combined[c,k] = (1 - α(c)) · attack_score[c,k] + α(c) · loop_score[c,k]
```

Friendly relay activates when `combined ≥ max(0.5 · max_combined, 0.15)`.
Forced-attack on non-friendly non-dead neighbors stays unchanged.

### What this is really doing (the attention reading)

In transformer terms: each slot's score is a tiny single-head attention
weight, the Q is `α(c)` plus the local gradient/topology features, and
the K is "what does this slot *offer* — attack uphill or loop-edge".
The mixing across heads is a hard convex combination (no softmax over
heads, just a scalar `α`). It's not full attention in the multi-head
softmax sense, but it's the architectural shape: content-based routing
that lets two distinct routing patterns coexist on the same cell with a
data-dependent split.

The interesting bit is that **α is computed locally** (BFS through
friendlies, no global state) — it's a per-cell "where am I in the
shape of my own territory" signal. Frontier-adjacent cells release;
deep-interior cells store; intermediate cells leak. This is the
reservoir-with-release shape.

### Why "hand-designed attention pulls even with `sum`" is a strong
result

`sum` won by 25 points over `max` in mixed play and by 62 points over
`loop`. Pulling even with `sum` on the first try with no tuning of
`deep_threshold` or `relay_thresh` is the architectural-fit signal
you'd hope for: it doesn't beat the simpler heuristic at first attempt
*but it doesn't lose either*. Two implications:

1. The two-head shape is at least as expressive as `sum` on this game,
   which is consistent with "the right shape exists; the specific
   values are the question."
2. The next step shouldn't be more knob-tuning. It should be **learned
   Q/K vectors** — PPO with a small attention head that produces
   `(attack_score, loop_score)` per slot and a per-cell mixing weight,
   trained end-to-end on the existing v2 reward stack. The hand-designed
   version is the architectural test; the learned version is where it's
   meant to live.

### Visual sanity check

`solver_v2_lightning_attn_20260515T021549.flxr` is all-six-seats
`lightning_attn`. The signature to look for: triskelion patterns deep
in territory (loops) that lean forward toward any frontier — the
even-k slots are visible as the underlying loop substrate, with
additional odd-k slots appearing on intermediate cells as α drops below
1 and the attack head starts contributing. Cells right at the frontier
look like pure `lightning` (max-mode) attacks. The transition is
smooth, not banded.

## Phase II: learning Q/K via PPO (`--model attn`)

The hand-designed `lightning_attn` solver tied `lightning_sum` on first
attempt with no tuning. That's the architectural-fit signal — the
two-head shape is at least as expressive as `sum` on this game. The
natural next move is to learn the Q/K vectors end-to-end with PPO
rather than continue hand-tuning `deep_threshold` and `relay_thresh`.

### `AttnActorCritic` — structured policy parameterization

Same 3-layer GCN backbone as the existing `GNNActorCritic`
(`python/flux_v2/ppo.py`), differs only in the policy head:

```
SET_k logit     = (1 - α(c)) · attack_q[c, k] + α(c) · loop_q[c, k]
CLEAR_k logit   = clear_head(c, k)          # generic
NOOP logit      = noop_head(c)              # generic
```

where:

- `attack_q_head : H3 → (N, K)`  per-slot attack score
- `loop_q_head   : H3 → (N, K)`  per-slot loop score
- `alpha_head    : H3 → (N, 1)` → sigmoid
- `H3` is the 3rd GCN layer's per-cell embedding (32-dim)

The attack/loop split is a **structural prior**, not a hard constraint.
PPO can collapse `attack_q ≈ loop_q` if the split isn't useful, in
which case the policy reduces to a single SET-score head. The interesting
research question is whether the gradient finds a meaningful separation
that mirrors the hand-designed `lightning_attn`'s frontier-tilt
behavior.

Wired through `python/scripts/train_v2.py` via a new `--model {gnn,
attn}` flag (default unchanged: `gnn`).

### Why a structured head and not a free 13-action logit

`GNNActorCritic` already learns a 13-action softmax over (SET_k,
CLEAR_k, NOOP) freely. Two reasons to prefer the structured head for
this experiment:

1. **Parameter efficiency.** The attn head has ~13·6·2 + 13 + 1 ≈ 170
   parameters in the SET path vs the GNN head's 13·HIDDEN ≈ 416. With
   fewer parameters to train *and* a structural bias matching the
   hand-designed solver, the gradient has a much shorter path to
   competent play.
2. **Interpretability.** After training, you can read off the learned
   `α` field per cell to see whether the network discovered the
   "frontier-distance ramp" the hand-designed version uses, or
   something different. That answers the open question of whether the
   `(1-α)·attack + α·loop` decomposition is the *right* decomposition
   for this game, vs e.g. a `(1-α)·attack + α·defense` or
   `softmax([attack, loop, defense, retreat])` 4-way head.

### Training run (results pending)

_Filled in once the first 30-iter run lands. Comparison points:_

- Mean reward / explained-variance / entropy curves vs the same
  configuration with `--model gnn`.
- α distribution at end-of-training — does it cluster at 0 / 1 the way
  the hand-designed BFS-distance ramp does, or stay diffuse?
- Visual: does the learned policy produce the same triskelion-tilt
  signature as the hand-designed `lightning_attn` replay, or something
  unrecognizable?
- Head-to-head: trained-attn vs `lightning_sum` and vs the
  hand-designed `lightning_attn`. The bar to beat is the 12-12 tie.

See `wiki/log.md` (entry: lightning sum / sum_pw modes) for the run log
and replays:

- `public/v2/replays/solver_v2_lightning_sum_*.flxr` — Run 1 game 0
- `public/v2/replays/solver_v2_lightning_sum_pw_*.flxr` — Run 2 game 0 (2-cycle feeding / Y-confluences — *not* true 3-cycles)
- `public/v2/replays/solver_v2_lightning+lightning_sum+lightning_sum_pw_*.flxr` — Run 3 game 0
- `public/v2/replays/solver_v2_lightning+lightning_sum_pw_*.flxr` — Run 4 game 0
- `public/v2/replays/solver_v2_lightning_loop_*.flxr` — Run 5 game 0 (**the triskelion pattern — directed 3-loops on every even-parity triangle**)
- `public/v2/replays/solver_v2_lightning+lightning_loop_*.flxr` — Run 6 game 0 (loop vs max head-to-head)
- `public/v2/replays/solver_v2_lightning_loop+lightning_sum_*.flxr` — Run 7 game 0
- `public/v2/replays/solver_v2_lightning_attn_*.flxr` — Run 8 game 0 (**attention-mixed: triskelions in the back, tilted-uphill toward the frontier**)
- `public/v2/replays/solver_v2_lightning_attn+lightning_sum_*.flxr` — Run 9 game 0 (sum vs attn 12-12)
- `public/v2/replays/solver_v2_lightning+lightning_attn+lightning_sum_*.flxr` — Run 10 game 0 (3-way mix)

### Big-grid showcase (R=12, 70 dead cells, 8000-tick cap)

For the visual proof-of-concept on a phone-watchable board:

- `solver_v2_bfs+lightning+lightning_attn+lightning_loop+lightning_sum+lightning_sum_pw_*.flxr` —
  **all six solvers on one board, one seat each**. The clearest A/B
  comparison: each color's territory has the visual signature of a
  different solver, all under the same starting conditions.
- `solver_v2_lightning_sum+lightning_attn_*.flxr` — alternating 3v3 of
  the two strongest solvers on the same R=12 arena.
- `solver_v2_lightning_attn_*.flxr` — all-attn self-play on R=12. The
  triskelion-tilts get a lot of room to develop on the larger board.

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

## Phase III: PPO training, moonshot speedups, and the reward-shape bug

After landing the hand-designed `lightning_attn` at parity with `sum`,
the next move was learning Q/K via PPO end-to-end. Documented here
because every step exposed a different architectural or reward-shape
issue that's worth keeping recorded for the next round.

### Moonshot recipe (cheap iteration cycle)

The canonical `v2-killer-tuned` recipe takes ~75s per iteration on
R=9 P=12 max_ticks=10000. Iterating on policy architecture or reward
shape at that pace is dead — you need 1-2 hours per experiment. The
moonshot stack collapses iteration time to ~5 min for 40 iters:

- `--radius 5 --num-players 6 --num-dead-cells 10` — ~91 cells vs 271
- `--max-ticks 2000` — 5× shorter games
- `--games-per-rollout 12` — 3× gradient diversity per iter
- `--ai-period-ticks 5` — unchanged
- `--opponent-seats "0,2,4" --opponent-solver lightning_sum` —
  half the seats run a fixed solver, anchoring gradient direction

Tradeoffs: small-board dynamics differ from R=9 (less room for
loops, faster captures), but the architectural and reward-shape bugs
that need fixing show up at this scale too — and you can SEE them in
the displayer instead of waiting an hour.

### The slot-direction-bias bug

First PPO run at moonshot scale gave a stable plateau (R≈470, no
deadlock, no NaN) but **0-23 head-to-head** against `lightning_sum`.
Watching the replay made the bug obvious: the trained policy spread
*only east and southeast* (slots 0 and 5 in axial coords) and
ignored neutrals in other directions.

Diagnosis: `AttnActorCritic`'s SET head was
`Linear(HIDDEN, K)` — six per-slot scores from six **independent**
linear projections of the cell embedding. The slot index existed in
the parameters. Random init gave slot 0 slightly higher weights;
PPO's first iter amplified the asymmetry; entropy was sticky around
1.84 so the policy never broke out of the directional rut.

Fix: replace the per-slot independent heads with shared pair-functions
over `(cell_emb, neighbor-at-slot-k_emb)`:

```python
self.attack_pair = Linear(2 * HIDDEN, 1)
# applied per (c, k) to concat(H3[c], H3[nb[c, k]])
```

The slot index never appears as a learnable parameter. Slot 0 differs
from slot 3 *only* because they point at different neighbors. PPO
cannot develop a slot-0 fixation because slot 0 has no special
parameter to amplify.

Sanity check: on a uniform-state board (all NEUTRAL, equal strength),
an interior cell's 6 SET logits are identical (std = 0.0).

Param count drops too: old policy heads = ~660 params, new = ~261.
Smaller and slot-symmetric.

After the fix, training-time metrics improved across the board:
entropy held at 2.38 (vs 1.84), value EV climbed 0 → 0.45 (vs 0.34),
captures rose slightly. **But the head-to-head was still 0-23.**

### The missing-relay-reward bug

The equivariant trained policy looked structurally healthy but
visually had a different pathology: many cells with **zero outflows
at all** sitting idle in the interior. The model hadn't learned to
issue CLEAR (when outflows go stale) and hadn't learned to issue
SET-toward-relay (when interior cells should chain pressure forward).

Reading off the per-tick reward signals at `MAX_EDGE = 100`:

| action | per-tick reward signal |
|--------|-----------------------:|
| pump at enemy/neutral | `+10` |
| **pump at friendly relay (chain forward)** | **`0`** |
| pump at friendly sink | `-0.15` |
| NOOP | `-0.01` |

For an interior cell with all-friendly neighbors, **NOOP is literally
optimal**. There's no positive signal to chase. The trained policy
correctly learned to do nothing on interior cells.

`flux_v2.step.transit_credit_per_cell_for_tick` was already
implemented as "the positive twin of destination-terminated waste" —
pays a source cell when its outflow lands on a MAX-strength friendly
relay with active outflows. But it was marked "diagnostic" and never
wired into the trainer.

Fix (in `python/scripts/train_v2.py`): batched numpy implementation
`_batched_transit_credit()`, new `--transit-credit-coef` flag,
`r_transit` term added to total reward. With
`--transit-credit-coef 0.05` and `--waste-coef 0.1`, interior cells
now see:

| action | new per-tick signal |
|--------|--------------------:|
| pump at relay | `+1.5` |
| pump at sink | `-3` |
| NOOP | `-0.01` |

Interior cells now have a clear positive reason to chain pressure
forward and a clear negative reason to point at sinks.

### Reward-shape audit takeaway

The "ignores neutrals / sits idle / can't sustain pressure" failure
mode wasn't an under-training symptom — it was a **literally rational
response to the existing reward function**. The hand-designed
solvers in [[v2-algorithmic-solvers]] get the maintenance behavior
"for free" because they recompute desired outflows from scratch
every AI tick from local geometric rules. PPO has to *learn* both
the attack behavior AND the maintenance behavior, and the gradient
signal for maintenance was missing.

This is the kind of bug that's invisible without a UI. The training
metrics looked fine; the policy was visually broken. Replay-driven
iteration was what surfaced it.

### Head-to-head progression

All evaluations: R=5 P=6 dead=10, 24 games, alternating 3v3 with
`lightning_sum`. Trained policy uses categorical sampling (matches
training).

| version | model params | wins | stalemates | mean ticks | note |
|---------|------:|---:|---:|---:|------|
| phase 3 (slot-biased)            | 660 | 0 | 1 | 1426 | east/SE bias, ignored 4 directions |
| phase 4 (equivariant)            | 261 | 0 | 1 | 1426 | sat idle on interior cells |
| phase 5 (+ transit + 20× waste)  | 261 | 0 | **4** | **1865** | losing slower, more drawn games |
| (reference) hand-designed `lightning_attn` | n/a | 4 / 12 | 0 | — | hand-coded prior, smaller sample |

Every architectural and reward fix moved the trajectory in the right
direction but didn't yet cross 0 wins. Phase 5's 4 stalemates +
30% longer games suggest the structural healing is real (interior
cells now have outflows pointing at relays) but the **resulting
policy still isn't competitive with `sum`** at 60 iters and 261
policy parameters. Beating a well-tuned hand-coded heuristic on a
91-cell board with this little capacity is genuinely hard.

### What's stopping it

Three plausible causes in order of cost-to-test:

1. **Capacity.** 32 hidden units / 3 GCN layers / 261 policy params
   is small. Bump `HIDDEN` to 64 (2× capacity per layer) and retrain
   from fresh.
2. **Training time.** 60 iters is a sprint; the user's reference
   overnight runs go 1000+. Continue from the phase-5 checkpoint with
   `entropy_coef` annealed 0.01 → 0.001 to commit harder.
3. **Mixed-seat training biases toward coexistence, not domination.**
   The reward signal during training is "do well alongside three
   `lightning_sum` opponents", which is satisfied by parity. Head-to-
   head requires actively beating. A curriculum (start vs `bfs`, then
   `lightning`, then `lightning_sum`) might escape this trap.

This is now a real research direction, not a 5-minute moonshot. But
the architecture and reward shape are now correct — every future
iteration can build on these without re-litigating the slot-bias and
missing-relay-reward bugs.

Related: [[v2-algorithmic-solvers]],
[[decisions/v2-edge-pressure-state]], [[v2-edge-voting-policy]],
[[v2-training-runs]],
[[decisions/v2-three-term-reward]] (the reward stack's design).
