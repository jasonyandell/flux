---
title: v2 temporal strategy — throttle, targeting, and options as one ML problem
kind: topic
first_seen: 2026-05-15
last_updated: 2026-05-16
status: active
---

## Frame

Three pathologies have appeared in v2 solver and PPO work, and they are
the same pathology in three costumes:

1. **Threshold control** — too little pressure stalls the frontier; too
   much waste-bleeds the sender. See the pressure/waste trade-off.
2. **Target dithering** — with ≥3 surviving enemies, a greedy solver
   cycles between targets and finishes none. Observed live: dominant
   orange seat at R=30 P=21 oscillating across 5+ fragmented enemies,
   reaching one, regen restores another, focus flips.
3. **Temporal strategy** — the *real* high-level vocabulary is
   migrate, spread, flank, chase, corner. These are not steady states.
   They are trajectories.

The unifying diagnosis: **the policy is reacting to instantaneous state
when it should be committing to a multi-tick plan.** Throttle is
steady-state control *during* a commitment. Target is the *goal* of a
commitment. Migrate/spread/flank/chase/corner are *types* of commitment.

## Why the current architecture can't express this

The active [[ppo-gnn]] policy emits 13 logits per cell per AI tick —
`Set 0..5 / Clear 0..5 / No-op`. There is no temporal state in the
output: every tick the policy re-decides from scratch. There is no
notion of "I am 70% through executing a flank." There is no scalar
"how much sustained throughput do I want on this edge." There is no
slow variable that says "the current target is pink — do not retarget
unless something dramatic changes."

Without these handles, an unconstrained policy lands on one of the two
cliffs in the pressure/waste trade-off and dithers between targets
under multi-enemy conditions. Both failure modes are visible at 0%
win-rate but they have opposite fixes — which is what makes
single-scalar reward tuning ineffective.

## The formal problem class

This is the **options framework** (Sutton-Precup 1999) applied to a
graph-structured continuous-time control problem. Equivalently:
goal-conditioned RL (UVFA, Schaul 2015; HER, Andrychowicz 2017) with a
recurrent meta-policy (FeUdal Networks, Vezhnevets 2017). The
dithering failure under multi-enemy boards is **bandit with switching
costs** (Banks-Sundaram 1994) — the provably-optimal policy when
switching is expensive is piecewise constant: commit, ride to done,
then switch. Greedy is suboptimal at K≥3 targets, which is exactly
where flux's dithering shows up.

Precedent: AlphaStar (DeepMind 2019) and OpenAI Five (2019) solved
harder versions of this exact shape — long horizon, graph-like map,
partial decentralization, sparse terminal reward. Flux is a cleaner
instance: pure step ([[pure-step-function]]), no fog of war beyond
ownership, no camera tricks. The hard part is engineering, not
research.

## Strategies as measurable graph quantities

Each item in the strategic vocabulary maps to a concrete delta on the
edge-activation field. All can be computed from `outflow[c,k]`,
`edge_pressure[c,k]`, and `owner` without rollouts:

| strategy | graph signature |
|---|---|
| migrate | shift of active-outflow centroid in graph distance |
| spread | variance of active-outflow distribution (or # connected components of active edges) |
| flank | angle between current pressure-on-enemy vector and new edge-activation direction |
| chase | edge-pressure-on-enemy maintained while enemy cell-set translates |
| corner | reduction in target seat's neutral-frontier perimeter |

These quantities have three uses simultaneously:

1. **Auxiliary prediction tasks** — the encoder learns representations
   that surface them, even when not directly rewarded.
2. **Intrinsic reward signals** — shaping bonuses for "reducing
   target's perimeter while committed to corner."
3. **Diagnostic dashboards** — eval-time signatures that distinguish
   "this solver dithers" from "this solver commits," independent of
   win rate.

## Architectural answer

The same hierarchical split handles all three axes (throttle,
targeting, temporal):

### Manager (slow, recurrent)

- Picks a *goal* from a continuous goal-space — e.g.
  `(target_seat, intent_vector, throttle_level)` — where
  `intent_vector` is a low-dim embedding learned to span
  migrate/spread/flank/chase/corner without being hand-enumerated.
- Updates every 50–100 ticks, not every AI tick.
- Recurrent state (LSTM or transformer over recent ticks) tracks
  "phase within current option."
- **KL-regularized against its own previous output.** Commitment by
  construction; switching is expensive in the loss, mirroring the
  in-game cost of rerouting in-flight pressure.

### Worker (fast, can be a heuristic)

- Goal-conditioned. Given the manager's goal, shapes the outflow
  field to execute.
- Can be a learned per-cell policy *or* `lightning_sum` parameterized
  by the goal — boundary conditions on the potential field set by
  the goal's target location and throttle level.
- Throttle is encoded by capping per-cell open-slot count, not by
  learning a continuous flow-rate variable directly. See
  the pressure/waste trade-off — bfs's success is precisely this
  structural one-slot-per-cell prior.

### Why this earns its keep

- **Throttle** becomes an explicit head, not an emergent side effect.
- **Targeting** lives at manager cadence with hysteresis built in —
  the dithering pathology disappears by construction.
- **Temporal strategies** emerge as clusters in the learned
  goal-embedding space. No need to enumerate "migrate, flank,
  chase" as code branches.

## Why pulse_stagger and attn variants failed in this frame

Pulse-and-release is fundamentally a temporal option — charge phase
closes outflows; release phase opens them. `pulse_stagger` tried to
express that as a *steady-state* algorithm and scored 0% (see
[[v2-overnight-research]]) because the right action depends on
*which phase* the cell is in, which is a hidden temporal variable a
steady-state algorithm has no place to store. An options policy
stores it naturally — the option *is* the phase.

`lightning_attn` (see [[v2-edge-loop-emergence]]) tied `sum` on first
attempt with hand-designed Q/K. That tie is significant: it shows the
*spatial* attention structure is right. What's missing is the
*temporal* scaffolding — `lightning_attn` still re-decides every
tick. Add a slow head choosing target and throttle, and the same
spatial machinery has somewhere to express commitment.

## Concrete first moves (in order of cheapness)

1. **Diagnostic only** — log per-tick *modal target* (enemy receiving
   most of a seat's outgoing pressure) and count switches per 100
   ticks. Cycling solvers will show 5–20; bfs will show 0–2. One
   number that separates dithering from committed play. Zero model
   changes required.
2. **Throttle prior** — hard-cap `lightning_sum` to one open slot per
   cell. Tests the hypothesis that bfs's edge is throttling, not
   targeting. If it closes most of the gap to bfs, the structural
   prior is confirmed and the throttle head becomes a clear lever.
3. **Behavior-clone bfs** — pretrain a policy to imitate bfs
   rollouts. bfs has both commitment (geographic targeting) and
   throttle (one parent per cell) by construction. Clone *that*, not
   `lightning_sum`. Replaces the prior recommendation in
   [[ppo-gnn]] / [[v2-edge-voting-policy]] context.
4. **KL-regularized fine-tune** — PPO from the BC warm start with a
   KL penalty back to the bfs prior. Lets the policy deviate to
   flood when the gradient is strong, snaps back to the throttled
   committed baseline when it isn't.
5. **Manager/worker split** — only after 1–4 land. Slow recurrent
   head over `(target_seat, throttle)`, fast worker is
   goal-parameterized `lightning_sum`. This is the AlphaStar pattern
   minus the league; league training is a later add.

## Validation experiments (2026-05-15)

The first two steps from the implementation plan ran the same day the
framing was written. Tooling:
[`python/scripts/eval_solvers.py`](../../python/scripts/eval_solvers.py)
(matched-pair tournament with Wilson 95% CI + sign test),
[`python/scripts/switch_rate.py`](../../python/scripts/switch_rate.py)
(per-seat modal-target switch rate per 100 ticks), and a `throttle`
kwarg on `lightning_solver_actions` that caps `desired` slots to top-N
by `pot[d]` (attack-tier ranked above relay-tier).

### Switch-rate diagnostic — null at self-play

Configurations R=20 P=8 dead=30 max_ticks=4000 and R=25 P=12 dead=80
max_ticks=12000, 3–4 games per solver:

- `bfs`: 0.58 / 0.66 switches per 100 ticks
- `lightning_sum`: 0.55 / 0.61
- `lightning_sum_throttled`: 0.88 / 0.93

No bfs-vs-sum gap. Throttle is *higher*, not lower. Both findings
contradict the naive hypothesis that bfs's edge is sub-AI-tick
commitment.

**Re-interpretation:** at AI-tick scale (sample period 25 ticks), the
"modal target" metric is noise-dominated. The actual dithering the
user observed plays out at 500–1000-tick timescales (kill A → switch
to B → A regens, switch back). The diagnostic needs a coarser sample
period and a *target-spell-completion* metric (does this seat's modal
target survive past its first commitment, or does it get abandoned?).
Also: symmetric self-play between identical solvers cannot produce
the asymmetric "dominant seat vs many fragments" condition where
dithering was observed. The pathology is real but invisible to
self-play.

### Win-rate eval — the throttle hypothesis is validated

Configuration R=25 P=12 dead=200 (~40% dead) max_ticks=12000, 20
matched pairs (40 games each):

| matchup | coherent | sign-test p | signed coherent advantage |
|---|---|---|---|
| `lightning_sum_throttled` vs `lightning_sum` | **9 / 9** | **0.0039** | +100pp |
| `lightning_sum_throttled` vs `bfs` | **7 / 7** | **0.0156** | +100pp |
| `lightning_sum` vs `bfs` | 5 / 5 | 0.0625 | +100pp (borderline) |

Raw win rates over 40 games: throttled-sum 67–72% vs both
alternatives; vanilla sum 62% vs bfs.

**Interpretation.** Throttling (one slot per cell on top of sum-mode
potential field) is a substantial, statistically significant
improvement over both bfs and vanilla sum at this config. The
"commitment by construction" intuition was directionally correct, but
the active mechanism is not what the switch-rate diagnostic
measures — likely a waste-reduction effect (fewer active outflow
slots → less overflow at maxed cells → sustained per-edge
throughput). The structural prior of *≤1 outflow per cell* combined
with sum-mode's *loop-aware potential* outperforms either component
alone.

**Implications for the framing:**

- **Throttle is a real lever.** Any learned policy with a throttle
  head can encode this directly. Step 2 of the implementation plan
  is confirmed.
- **bfs is *not* the right BC target.** It loses to throttled-sum at
  this config. Behavior-clone from `lightning_sum_throttled` instead.
  This revises the prior recommendation in the "concrete first
  moves" section above.
- **Switch-rate at AI-tick scale is not the right dithering proxy.**
  Build a coarser-grained "target-spell-completion" metric for the
  next diagnostic pass, especially under mixed-strategy FFAs where
  the original observation occurred.
- **Waste/throughput rather than commitment may be the dominant
  effect.** Test by measuring per-tick mean waste in throttled vs
  vanilla sum games — if throttle wins by reducing waste rather than
  reducing switches, the framing should shift toward
  "waste-bound-vs-pressure-shy" as the primary axis and
  target-commitment as the secondary axis.

### Open follow-ups

- **Target-spell diagnostic exists.** `python/scripts/target_spell.py`
  samples modal enemy target at a coarse cadence and reports spell
  completion vs live-target abandonment. In a small mixed R=20 P=6 sample,
  hard sticky targeting pushed completion up (75% vs 25%) and abandonment
  down (0% vs 33%) but still lost badly in head-to-head. The diagnostic
  is useful; the hard intervention is not.
- **Target-hysteresis needs a softer form.** A hard prototype
  (`lightning_sum_throttled_sticky`, registered in
  `python/scripts/run_v2_solver.py`) latches the first modal enemy target
  receiving outgoing attack pressure, then treats other enemy seats as
  blocked until the target has no cells. It is a useful negative control,
  not a win: at R=20 P=8 dead=80 max_ticks=6000, 12 matched pairs,
  `lightning_sum_throttled` beat it 8/8 coherent pairs (p≈0.0078,
  20-4 raw games). The failure suggests all-or-nothing target commitment
  starves opportunistic conversion and local defense. If target commitment
  is still worth testing, make it a bias/switching cost, not a hard mask.
- **Multi-strategy FFA test.** Run `lightning_sum_throttled` in the
  R=30 P=21 big-zoo where bfs was previously the only converter.
  Predict: throttled-sum becomes the new converter, beating bfs at
  the conversion job.
- **Throttle sweep.** Evaluate throttle ∈ {1, 2, 3, 6}. If throttle=2
  closes most of the gap and 3 erases it, the cliff is sharp and
  one-slot is the structural sweet spot; if it's smooth, throttle is
  a tunable knob worth treating as a learned scalar.
- **Waste-attribution diagnostic.** Per-tick mean waste, per-cell
  active-slot histogram — separate the throttle effect into its
  components.

## Cross-references

- [[v2-edge-voting-policy]] — edge-centric representation; this
  page's manager/worker split presupposes its output factorization.
- [[v2-edge-loop-emergence]] — attention as the spatial primitive;
  temporal scaffolding is the missing layer.
- [[v2-overnight-research]] — the methodology base (matched pairs,
  seat-bias quantification) needed to actually measure whether any
  of this works.
- [[ppo-gnn]] — the current node-centric policy this proposes to
  replace.
