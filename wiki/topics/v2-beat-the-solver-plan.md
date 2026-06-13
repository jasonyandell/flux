---
title: v2 beat-the-solver plan — superset ML over the champion
kind: topic
first_seen: 2026-06-12
last_updated: 2026-06-13
status: active
---

## What this page is

The concrete program for producing a learned (or hybrid) agent that beats
`lightning_sum_throttled` under Todd's matched-pair protocol
([[v2-todd-measurement-lab]]). It refines [[v2-grand-research-plan]]
phases 2–3 in light of the 2026-05-16 BC failures ([[v2-training-runs]]):
the central move is no longer "distill, then RL" but **build policy
classes that strictly contain the champion, initialize there, and
hill-climb matched-pair win rate.** Imitation becomes a one-day science
probe (Gate 0), not a pipeline stage.

## Diagnosis: the BC/PPO failures are an information gap

The champion factorizes into a global computation followed by a nearly
trivial local readout (all in
[`python/flux_v2/solver_vec.py`](../../python/flux_v2/solver_vec.py)):

1. **Intrinsic field** — per-cell reward: `weak_bonus·(1−strength/MAX)`
   on enemy cells, `expand_bonus` on neutrals, zero elsewhere
   (`compute_potential`).
2. **Global propagation** — up to 32 Bellman value-iteration sweeps over
   the whole graph at `gamma=0.85`. Effective receptive field: tens of
   hops.
3. **Local readout** — attack every non-friendly slot; relay uphill to
   friendly neighbors within `fanout_eps` of the best friendly pot
   (`_gradient_relay_core`).
4. **Throttle** — keep the top-1 desired slot, attack tier first, then
   by `pot[d]` (`_throttle_top_k_core`).
5. **Picker** — diff desired vs current outflows; emit one SET
   (preferred) else one stale CLEAR per cell per AI tick, rotated
   tie-break (`_actions_from_desired`).

Steps 3–5 are almost linear *given the field*. The hard part is step 2.

The students that failed were 3-hop GCNs over 9 purely local channels
(`build_features` in
[`python/flux_v2/ppo.py`](../../python/flux_v2/ppo.py)); the edge-aware
variant adds local edge features but no extra reach. The original
hypothesis was that none of them could represent the champion's decision
*even in principle* — that consequential actions depend on field
structure 10–30 hops away. **Gate 0 (run 2026-06-13, below) partially
refuted that.** The corrected reading of the 2026-05-16 failure, in order
of contribution:

- **Error compounding** (dominant). A 3-hop net predicts the champion's
  per-edge decision at ~0.93 AUC (Gate 0) — *most* of the function is
  locally representable. But ~7–10% per-edge error compounds: one wrong
  relay breaks a transport chain, and over a full board × thousands of
  ticks the structure degrades into unplayable. Good per-edge accuracy
  is not a playable policy.
- **Serialized-action label** (major). BC supervised the 13-way
  per-cell action (NOOP 87–93%), a harder and sparser target than the
  per-edge desired-mask bit Gate 0 probes. Acting in desired-mask space
  removes this.
- **Covariate shift** (real). The clone's own errors lead off the
  champion's state distribution; DAgger territory.
- **Optimization pathologies** (documented). Entropy collapse and KL
  spikes from cloning a near-deterministic teacher ([[v2-training-runs]]).

So the residual representability gap is real but *modest* — pot adds
~+0.01–0.02 AUC overall and more on the relay/throttle half (Gate 0).
The deeper lessons are that the action interface and exactness matter
more than raw reach. Both point at the same fix as the original plan
(hand the policy the field; act in mask space), but for corrected
reasons — see the Ring 1 re-basing below.

### Gate 0 — RESULT (run 2026-06-13)

Built as a training-free probe instead of the originally-specified
BC rerun — cleaner, because it isolates *representability* from BC's
optimization dynamics.
[`python/scripts/gate0_probe.py`](../../python/scripts/gate0_probe.py)
runs champion self-play, labels each owned cell's valid outflow slot
with the champion's post-throttle desired-mask bit, and fits a numpy
logistic probe on three nested edge-feature sets: **L0** (the exact 9
GNN node channels for source+dest, 0-hop), **L3** (+ 1/2/3-hop
mean-pooled channels — the *same* receptive field the failed 3-hop GNN
had), **POT** (+ the champion's potential field: pot[c], pot[d], gap,
neighbor rank, margin-to-max-friendly). Train/test split by game.

Result, two seeds, ~130k edges per radius:

| R | L0 AUC | L3 AUC | POT AUC | gap L3→POT | relay L3→POT |
|---|---|---|---|---|---|
| 5 | 0.90 | 0.93 | 0.94–0.95 | +0.020 | 0.91→0.95 |
| 7 | 0.90 | 0.93 | 0.95 | +0.017 | 0.91→0.95 |
| 12 | 0.90 | 0.93–0.94 | 0.95 | +0.012 | 0.93→0.96 |
| 20 | 0.91 | 0.94 | 0.94–0.95 | +0.010 | 0.93→0.96 |

Two findings, both robust across seeds:

1. **3 hops already capture ~0.93 AUC.** The per-edge decision is mostly
   *local-gradient-following*: relay points at the friendly neighbor of
   highest pot above you — that needs the local pot *gradient*, not its
   distant structure, and 3 hops of message passing approximate a local
   gradient fine. Attack is purely local; throttle is a 6-way local
   compare. The "needs 10–30 hops" claim was wrong.
2. **The gap SHRINKS with board size** (+0.020 at R=5 → +0.010 at R=20),
   the opposite of the receptive-field-scaling prediction. Bigger boards
   have *smoother* fields per hop, so the local gradient is *easier* to
   follow. Pot's residual value is the exact gradient + exact throttle
   rank (visible in the larger relay-AUC lift), not distant reach.

**Verdict:** the information-gap framing is refined, not confirmed. Pot
carries real but modest non-redundant signal; the dominant cause of the
BC failure is error-compounding + the action interface, not
unrepresentability. This is why Ring 1 is still right but for corrected
reasons (next section).

**Capstone — deploy the probes as solvers** (`--deploy-pairs`, R=7, 30
matched pairs vs the champion on fresh boards): a hard-threshold of each
probe's per-edge prediction, serialized through the champion's own picker.

| probe | per-edge AUC | decisive win rate vs champion |
|---|---|---|
| L3 (3-hop local) | 0.93 | **0%** (0W 38L 22T) |
| POT (+global field) | 0.94 | **27%** (7W 19L 34T) |
| champion vs champion | — | ~50% baseline |

Three things, all decisive:

1. **0.93 per-edge AUC → 0% wins.** Per-edge accuracy does not transfer
   to play. This is the 2026-05-16 BC failure reproduced in a clean
   setting and explained: it was compounding, not representability.
2. **The global field is worth far more to *play* than to *predict*.**
   +0.016 AUC became +27pp of win rate — small per-edge gains compound
   *favorably* once deployed. "Hand the policy the field" survives the
   correction.
3. **Even with pot, an independent-edge threshold is unplayable** (27% ≪
   50%) because the throttle is a *joint* per-cell constraint ("keep
   exactly the top-1 slot"), not six independent edge decisions. This is
   exactly why Ring 1 keeps the champion's *exact* throttle + picker and
   only learns the field's inputs/readout — and why Ring 0/1, which retain
   the whole operator, actually beat the champion while clones cannot.

## Principle: the champion is a point in policy space — start there

Every stage builds a policy class that **strictly contains** the
current champion, initializes at it (analytic init = perfect clone, no
imitation loss needed), and optimizes matched-pair win rate from there.
Search never starts from a losing point, and each ring contains the
previous one:

| ring | class | new degrees of freedom | timescale |
|---|---|---|---|
| 0 | the champion's own scalar knobs | ~6 scalars | days |
| 1 | neural intrinsic + readout around the frozen Bellman operator | O(10²–10³) params | weeks |
| 2 | opponent fields, temporal manager, league | new inputs + slow state | after Ring 1 |

A corollary: behavior cloning is unnecessary as a pipeline stage on
this path. Initialization *is* a perfect clone. (DAgger / intent-mask
supervision stay relevant only if a pure-net distillation of a hybrid
champion is ever wanted for its own sake.)

**Why this beats BC, re-based on the Gate 0 result.** Gate 0 showed a
3-hop net *can* mostly represent the champion (~0.93 AUC per edge), so
"only Ring 1 can represent it" is the wrong justification. The right one:
Ring 1 (i) computes the field *exactly* — zero per-edge approximation
error, so nothing compounds; (ii) acts in *desired-mask space* through
the solver's own picker — no serialized-action bottleneck, idempotent,
no SET-spam attractor; (iii) is trained by ES on *game outcomes* — there
is no teacher state-distribution to shift away from. It starts at exact
champion fidelity and moves only where the games reward it. The three
dominant BC failure causes (compounding, action interface, covariate
shift) are all structurally absent, which is what the Ring 0/1 evolution
results below empirically confirm.

## Ring 0 — evolve the champion's own knobs (days)

The champion ships with hand-picked constants that have never been
swept jointly (registration:
[`python/scripts/run_v2_solver.py`](../../python/scripts/run_v2_solver.py),
defaults: `lightning_solver_actions` in
[`python/flux_v2/solver_lightning.py`](../../python/flux_v2/solver_lightning.py)):

| knob | champion value | note |
|---|---|---|
| `gamma` | 0.85 (default) | γ=0.94 helped vanilla sum pre-vec; **never tested under throttle** |
| `weak_bonus` | 1.0 | enemy-weakness intrinsic |
| `expand_bonus` | 0.6 | neutral intrinsic |
| `defense_bonus` | **0.0** | the champion has no defense term at all |
| `fanout_eps` | 0.05 | relay near-max tolerance |
| `throttle` | 1 | {1,2,3,6} sweep queued in [[v2-temporal-strategy]], not run |

**Method.** CMA-ES (or simple (μ,λ)-ES; SPSA as fallback) over these
scalars, population ~16.

**Fitness.** Primary: matched-pair win rate vs the frozen champion on a
**fixed deterministic board set** with paired seat rotation and pinned
RNG seeds — common random numbers via Pete factory boards
([[v2-pete-factory]]). CRN is what makes ES workable against the 6pp
seat-bias noise floor. Secondary tie-break: mean dominance margin —
the two opposite stalemate pathologies (waste-bound and pressure-shy)
both present as 0% wins, so win rate alone is flat exactly where the
search needs gradient.

**Ladder.** Screen at R=10 P=6, ~32 boards (64 games) per candidate;
confirm survivors at R=25 P=12 40%-dead. Rough budget: at Pete speeds
a screen generation is minutes; an overnight run is ~100 generations.
Run **per world** (default edge physics and `EDGE_ALPHA=0.05` fluid):
one champion per world. Promotion happens only through Todd's official
protocol (Wilson CI, sign test, seat rotation) — never on ES internal
fitness.

**Output.** A Ring-0 champion, plus a knob-sensitivity map that tells
Ring 1 where to spend capacity (e.g. if `defense_bonus>0` wins,
defense-aware intrinsic is a confirmed-value feature).

## Ring 1 — neural boundary conditions on a frozen Bellman operator

Keep the global propagation and the picker; learn what feeds the field
and what reads it:

```
actions = picker ∘ throttle_θ ∘ readout_θ ∘ Bellman ∘ intrinsic_θ
```

- **`intrinsic_θ`** — tiny net (start linear, then 1 hidden layer) over
  per-cell local features: ownership one-hot, strength, inbound enemy
  pressure (`_inbound_enemy_pressure`), frontier distance
  (`_frontier_distance`), neutral adjacency, game-phase scalar.
  Champion-init: reproduce `weak_bonus·(1−strength)` + `expand_bonus`
  exactly (in-class).
- **`readout_θ`** — per-edge scorer over (`pot[c]`, `pot[d]`, pot gap,
  edge category, `edge_pressure`, strength pair) → desired logit.
  Champion-init: attack ∪ uphill-relay-within-eps.
- **`throttle_θ`** — per-cell cap from local features (frontier cells
  may want 1 while interior relays want 2). Champion-init: constant 1.

**Action surface.** Emit the (N, K) desired mask and reuse
`_actions_from_desired`. Idempotent (no SET-spam attractor), matches
the solver's native interface, and gives dense supervision if ever
needed. This is the missing half of [[v2-edge-voting-policy]]: that
design had the edge scorer but only local inputs — here the pot field
does the global work by construction.

**Optimizers.** Primary: the same CRN-ES loop as Ring 0 — no credit
assignment, robust to the noise floor, embarrassingly parallel under
Pete. Secondary: PPO with a KL anchor to the champion-init policy (not
an entropy bonus) if gradient finesse is needed. A differentiable path
exists if wanted: `_mlx_bellman_sum_32` is a compiled pure-MLX graph,
so backprop-through-operator is available — pin to the cold 32-iter
variant for gradients (the warm-start 4-iter inference path is a known
semantic delta; the vectorization episode taught us to respect those).

**Gates.**

- *Gate 2 (reproduction):* at champion-init, the Ring-1 policy must
  match champion actions on a probe state set (same RNG draw schedule
  for picker tie-breaks) and tie ~50/50 in matched pairs. Failure here
  is by definition an implementation bug.
- *Gate 3 (the win):* beat the Ring-0 champion above the noise floor
  under Todd's protocol.

**What Ring 1 can express that the champion can't** (hypothesis list
for where the win comes from): defense-aware intrinsic on threatened
own cells; per-enemy-seat weakness weighting; per-cell adaptive
throttle (frontier vs interior); overkill suppression near
already-doomed targets; stale-route clearing readout;
phase-dependent expand/attack balance.

## Ring 2 — capabilities outside the solver family

Staged only after Gate 3, since each adds inputs or state to the
Ring-1 class (still superset-and-warm-start at every step).

1. **Opponent fields.** Pot is seat-myopic. We have full observability,
   and the batched path already computes every seat's field each AI
   tick (`_lightning_sum_batched_core`) — feed *rival* fields into
   `intrinsic_θ`/`readout_θ`. Genuinely new strategic information no
   hand mode uses: contested-target avoidance, kill-steals, letting
   two rivals grind each other.
2. **Temporal manager.** The [[v2-temporal-strategy]] manager/worker
   split, with the goal space now concrete: the manager is a slow
   recurrent head emitting modulations of Ring-1 parameters
   (intrinsic weights, per-seat target weights, throttle) every 50–100
   ticks with a switching cost (KL-to-previous-goal). The worker is
   the Ring-1 pipeline itself. Small recurrent state keeps it
   ES-trainable. This is also the spectacle ring: visible
   charge/hold/release and focused campaigns instead of tick-reactive
   shimmer.
3. **League.** Frozen lineage (hand champion, Ring-0, Ring-1 best),
   exploiters trained specifically against the current best, mixed
   FFA boards — [[v2-grand-research-plan]] Phase 4. Guards against
   overfitting to one frozen opponent. Todd gates every promotion.

## Standing changes (worth making for any trainer)

- **Desired-mask interface** for every learned policy; reuse the
  solver's picker. The 13-way per-cell edit surface forces policies to
  serialize a configuration into single edits and feeds the SET-spam
  attractor documented in [[v2-training-runs]].
- **Train against the champion**, not self-play parity — fixed solver
  seats already exist in `train_v2.py`; the ladder in
  [[v2-ml-gameplay-opportunities]] still applies.
- **Dense win-aligned proxy:** per-tick Δdominance, since terminal
  bonuses are invisible at γ=0.99 over multi-thousand-tick games
  (cross-run lesson in [[v2-training-runs]]).
- **Same `EDGE_ALPHA` in train and eval**; champions are per-world.

## Gates and sequencing

| # | gate | test | status (2026-06-13) |
|---|---|---|---|
| 0 | information-gap probe | local vs +pot per-edge prediction | **done — refined the diagnosis** (gap modest, ~0.93 local; compounding+interface dominate) |
| 1 | Ring-0 champion | ES knobs beat champion above noise floor (Todd) | **passed at R=7** (7–1 coherent, holdout 7–0 p≈0.016); **wash at R=20** (transfer gap) |
| 2 | Ring-1 reproduction | champion-init matches champion; ~50/50 pairs | **passed** (bit-exact, `test_v2_ring1_parity.py`) |
| 3 | Ring-1 win | ES from champion-init beats Ring-0 champion (Todd) | **failed at R=7** — v1 holdout 47%, v2 (40 boards+anchor) holdout 42% (0–5 coherent against). Dead end at this scale; retry at R≥12 / harder regime or treat the gap as temporal → Ring 2 |
| 4 | Ring-2 staging | each addition re-passes Gate-2 reproduction, then beats previous best | not started |

Original note (kept): Gate 0 and Ring 0 were meant to run in parallel —
one validates the theory, the other mints the first ML champion. That
happened: Ring 0 produced a real R=7 win the same session Gate 0 showed
the theory needed correcting. The two are independent, which is why a
partly-wrong diagnosis still yielded a working champion.

## Failure modes specific to this plan

- **ES fitness noise.** The 6pp floor eats small populations. CRN board
  sets, rank-based selection, enough games per candidate — and never
  promote on ES internal fitness, only on Todd's protocol.
- **Operator drift.** Training against cold 32-iter Bellman while
  inferring with warm-start 4-iter (or vice versa) is a semantic delta
  of exactly the kind that made the pre-vec rankings provisional. Pin
  one variant per experiment.
- **Stagnation at init.** Warm-starting at the champion with timid
  mutation scales can sit at the champion forever. Track
  distance-from-init; use restarts and mutation-scale schedules.
- **Frozen-opponent overfitting.** A Ring-1 win vs one frozen champion
  may be an exploit, not strength. The league (Ring 2c) is the
  structural fix; until then, validate wins across both worlds and
  multiple board distributions.
- **Metric drift.** Wins via degenerate physics exploits read as
  progress on the scoreboard. Keep Todd's "why it won" artifacts
  (waste, active-slot histograms, target-spell completion, replay
  review) attached to every promotion.

## Results so far (2026-06-13)

First evolution + probe pass, R=7 P=6 dead=15 screen world unless noted.
All numbers are provisional lab signal; official promotion is Todd's
`eval_solvers.py`.

**UPDATE (autonomous campaign): transfer is solved at the knob level; the
current champion is `evolve_r0_r20` (score 0.746).** Two transfer-robust
champions emerged, and the winner is counter-intuitive:

- `evolve_r0_ms` (multi-scale R=7/12/20): 67.8/71.2/79.6%, score 0.703 —
  low-gamma + aggressive expansion.
- `evolve_r0_r20` (**single-scale R=20** tuning): 77.6/83.3/71.7%, score
  **0.746**, significant at R=7 and R=12 (coherent 6-0, p=0.031) — defense-heavy
  (`defense_bonus=0.777`). **This is the best champion.**

**Methodology: train on the hardest case, not the average.** Tuning purely at
the largest/richest scale (R=20) generalized DOWN to small boards *better* than
multi-scale averaging — the hard regime forces robust mechanics (defense,
moderate gamma) that also work at easier scales. This vindicates the defense
lever (the single-scale R=7 finding) — decisive once surfaced at large scale.
The open goal is a *learned* (Ring 1/2) policy beating the **0.746** knob bar;
live board [[v2-champion-lab]]. Single-scale results below are the original
record.

**Ring 0 — a real win at training scale.** ~230 generations of CMA-style
ES over the champion's six knobs reached ~74% CRN win rate vs
`lightning_sum_throttled`. Out-of-sample it holds:

- holdout (fresh boards, `--confirm-pairs 40`): candidate **61.6%**
  [50.2%, 71.9%], coherent pairs **7–0**, sign-test p≈0.016.
- Todd `eval_solvers.py` (R=7, 40 pairs, fresh seeds): evolve_r0 wins
  coherent pairs **7–1** (+75pp), sign-test p≈0.07.

So evolved knobs beat the hand champion at the scale they trained on.
The evolved genome moves `gamma` ~0.85→0.85 (≈unchanged), nudges
`weak_bonus` down and `expand_bonus` down, and — notably — turns on a
small `defense_bonus` (the hand champion's is 0.0). Registered as
`evolve_r0` in `run_v2_solver.py`.

**Transfer is unsolved.** At R=20 the same genome is a wash:
Todd reports 2–0 coherent but 28/30 pairs *split* (p≈0.5) — the edge
mostly vanishes on 4× boards, where seat bias dominates the signal. The
R=7-tuned `gamma`/`throttle` don't carry to a larger diameter. Fix:
multi-scale board curriculum during evolution (train across radii at
once). This is the gap to the north-star's "transfers across board
sizes" clause.

**Ring 1 — a clean negative result at R=7.** The 19-param genome hit
~64% CRN but only **46.7%** [35.8%, 57.8%] on the holdout (coherent 1–5
*against*) — overfitting from the extra degrees of freedom. The retry
`ring1_v2` (40 CRN boards + `--anchor-coef 0.03` champion regularizer)
plateaued at only ~57% in-sample (far below Ring 0's ~74%) and held out
at **42.1%** [31.6%, 53.3%], coherent **0–5 against** (p≈0.06) — *worse*
than the original. More boards + regularization did not help: at R=7 the
19-param field policy is simply a worse policy than the champion. Ring 1
at this scale is a **dead end** — do not re-run it expecting a win.

**Why, and the convergent lever.** Both rings independently drove
`defense_bonus` / `w_defense` to ≈0.37 from the champion's 0.0 — the one
real improvement either found is *a small defense term*, which Ring 0
already captures in a single knob. Ring 1's other 18 parameters add
overfitting room, not signal, because at R=7 the champion is near-optimal
(Gate 0: the per-edge decision is ~93% local and well-structured — little
headroom for extra capacity to exploit). Ring 1's capacity can only earn
its keep in a regime where the champion is genuinely *weak* — bigger
boards, more seats, the multi-enemy dithering condition of
[[v2-temporal-strategy]] — not the tiny screen arena.

**Takeaway.** The simplest hypothesis class (Ring 0, 6 physical knobs)
won and generalized; the richer one (Ring 1) lost. Start simple; add
capacity only where a measured weakness gives it something to bite on.
The champion's real exploitable flaw, found twice independently, is its
zero defense term.

**Highest-value next experiments** (replacing "let Ring 1 cook"):

1. **Multi-scale Ring 0.** Evolve the six knobs against a board
   *distribution* spanning radii (e.g. R∈{7,12,20}) at once, directly
   targeting the transfer gap — the actual north-star miss. Cheapest path
   to a champion that holds at R=20.
2. **Ring 1 where the champion is weak.** Re-run Ring 1 at R≥12 P≥12
   (and/or the dithering FFA), no anchor, where its defense/opponent-aware
   degrees of freedom have real headroom. If it can't win *there*, the
   learnable gap is temporal (jump to the Ring 2 manager), not spatial.
3. **Defense-augmented champion as the new baseline.** Promote the Ring 0
   `evolve_r0` (defense on) via Todd, then make *it* the opponent — Ring 1
   must beat the stronger baseline, not the zero-defense original.

## Runbook (landed 2026-06-13)

Rings 0 and 1 are implemented and runnable fully offline:

- [`python/flux_v2/ring1.py`](../../python/flux_v2/ring1.py) — the Ring 1
  field-policy class (19-param genome: linear intrinsic, relay/attack
  readout, slot ranking, per-cell throttle around the frozen
  `_compute_potential_core`). `champion_vector()` reproduces
  `lightning_sum_throttled` bit-for-bit.
- [`python/scripts/evolve_champion.py`](../../python/scripts/evolve_champion.py)
  — dependency-free (μ,λ)-ES for both rings: CRN matched pairs vs a frozen
  opponent, fitness = win score + 0.2·cell-share margin (margin supplies
  gradient where both stalemate pathologies read 0% wins), best-ever
  anchored recombination, JSON checkpoints with resume, best-vs-champion
  FLXR replays dropped into `public/v2/replays/` for the viewer.
- [`python/tests/test_v2_ring1_parity.py`](../../python/tests/test_v2_ring1_parity.py)
  — **Gate 2 passes**: champion-init matches the champion action-for-action
  over full games (3 seeds × 60 AI ticks × 4 seats); perturbed genomes
  diverge (knobs are live).
- [`python/scripts/gate0_probe.py`](../../python/scripts/gate0_probe.py) —
  the Gate 0 receptive-field probe (numpy logistic, no sklearn). Sweeps
  radii, prints the L0/L3/POT AUC table above.
- `evolve_r0` / `evolve_r1` are **registered solvers** in
  `run_v2_solver.py` (lazy-load the checkpoint genome), so Todd's
  `eval_solvers.py` runs them directly.

```bash
cd python
# Ring 0 (champion's own knobs):
.venv/bin/python scripts/evolve_champion.py --ring 0 \
    --generations 400 --pop 12 --boards 16 --workers 6 --replay-every 10
# Ring 1 (field policy), with more boards + champion anchor (overfit fix):
.venv/bin/python scripts/evolve_champion.py --ring 1 \
    --generations 600 --pop 16 --boards 40 --workers 8 \
    --anchor-coef 0.03 --replay-every 10 --out checkpoints/evolve/ring1_v2.json
# Watch:   tail -F /tmp/flux-evolve-r*.log   (mirror-match baseline = +0.50)
# Holdout: evolve_champion.py --ring 0 --resume checkpoints/evolve/ring0.json --confirm-pairs 40
# Official promotion via Todd:
.venv/bin/python scripts/eval_solvers.py evolve_r0 lightning_sum_throttled \
    --pairs 40 --radius 7 --num-players 6 --num-dead-cells 15 --max-ticks 3000
# Gate 0 probe (+ capstone: deploy probes as solvers vs champion):
.venv/bin/python scripts/gate0_probe.py --radii 5,7,12,20 --games 14
.venv/bin/python scripts/gate0_probe.py --radii 7 --games 14 --deploy-pairs 30
```

Checkpoints land in `python/checkpoints/evolve/ring*.json`; replays are
named `evolve_r{ring}_gen*.flxr` and show up in the `/index-v2.html` drawer.
ES fitness is CRN-internal — treat any number here as provisional until the
fresh-board holdout and Todd's protocol agree.

### Discovery: the champion's attack ranking is vestigial

Found while proving Gate 2: `_throttle_top_k_core` scores attack slots as
`pot[d] + 1e30` **in float32, which saturates** — every attack slot ties at
exactly 1e30, so the intended pot-ranking never happens and the scan always
keeps the lowest slot index. The reigning champion has a fixed eastward
attack bias and effectively unranked attack throttling. The Ring 1 genome
splits this into `rank_pot_attack` (champion-equivalent 0.0) and
`rank_pot_relay` (1.0), so evolution can now learn the ranking the original
code intended but never computed.

Related: [[v2-grand-research-plan]], [[v2-ml-gameplay-opportunities]],
[[v2-temporal-strategy]], [[v2-training-runs]], [[v2-edge-voting-policy]],
[[v2-vectorized]], [[v2-todd-measurement-lab]], [[v2-pete-factory]],
[[v2-overnight-research]].
