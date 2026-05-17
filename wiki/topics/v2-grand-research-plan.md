---
title: v2 grand research plan
kind: topic
first_seen: 2026-05-16
last_updated: 2026-05-16
status: active
---

## Frame

If flux v2 had a serious research grant, the mandate should not be
"buy enough compute for PPO to discover physics." The current frontier says
the opposite: the solver stack already exposes strong mechanics
([[v2-vectorized]], [[v2-overnight-research]], [[v2-temporal-strategy]]).
The grant should buy a disciplined factory for turning those mechanics into
learned, evaluated, and inspectable policies.

The north-star result is a learned or hybrid agent that beats the current
best solver champion under matched-pair evaluation, transfers across board
sizes and dead densities, and produces replays that look strategically
intentional rather than merely numerically strong.

## Compute principle

Compute is not infinite; it is a burn-rate instrument. Spend it in this order:

1. **Cheap truth first.** Use the vectorized solver harness to make every
   claim pass matched-pair tests before launching long training.
2. **Distill before RL.** Behavior-clone `lightning_sum_throttled` /
   `sum_long` / `wave_long` rather than making PPO rediscover pressure flow.
3. **Learn residuals and timing.** Use RL for the parts local solvers cannot
   express: throttle, target persistence, defense, finishing, stale-route
   clearing, and when to deviate from the long-field prior.
4. **Scale only after evidence.** Increase board size, population, model
   capacity, and cloud budget only when smaller matched-pair tests predict
   value.

## Program Shape

### Phase 0: Measurement Lab

- Rerun the official solver hierarchy on the vectorized code, because
  [[v2-vectorized]] changed solver semantics enough that the pre-vectorize
  ranking is provisional.
- Standardize every leaderboard around matched pairs, Wilson intervals, sign
  tests, and seat-rotation sanity checks.
- Add diagnostics for waste, active-slot histograms, target-spell completion,
  stale slots, follow-through after capture, and defense saves.

Output: a trusted scoreboard and a replay corpus that can say *why* one agent
won, not just that it won.

### Phase 1: Solver Factory

- Treat `lightning_sum_throttled` as the first champion unless the vectorized
  rerun dethrones it.
- Run systematic sweeps over throttle, gamma, dead density, radius,
  player count, and mixed-strategy FFA conditions.
- Store close same-board divergences as preference examples.

Output: a family of high-quality teachers plus hard states where teachers
disagree.

### Phase 2: Distillation And Residuals

- Behavior-clone the solver champions into the node-centric and edge-aware
  policy surfaces.
- Add a residual-policy path: solver proposes the default edge intent; the
  model learns open/clear/hold corrections.
- Prefer small interpretable heads until the residual beats the teacher on
  named arenas.

Output: a policy that preserves solver expansion speed while improving late
game conversion, defense, and cleanup.

### Phase 3: Temporal Manager

- Add the slow recurrent manager described in [[v2-temporal-strategy]]:
  target, throttle, and low-dimensional intent update every 50-100 ticks.
- Keep the worker either solver-parameterized or solver-warm-started at first.
- Penalize gratuitous switching through KL-to-previous-goal or explicit
  switching-cost terms.

Output: commitment as an architecture feature, not a reward-tuning accident.

### Phase 4: League And Scaling

- Build a league of solver champions, stale learned checkpoints, exploiters,
  and mixed FFA boards.
- Use active data collection: spend expensive training on states where current
  agents disagree, collapse, or fail to finish.
- Scale to cloud GPU/CPU fleets only after the local vectorized lab can predict
  which experiment is worth the burn.

Output: robust agents, not one lucky checkpoint that beats one solver on one
board distribution.

## Budget Use

Most of a 10 million dollar grant should buy people/time/compute operations,
not one monolithic training run:

- reliable experiment orchestration, replay storage, and dashboards;
- large CPU fleets for vectorized tournaments and preference mining;
- GPU/accelerator budget for distillation, residual RL, and league training;
- human review of strange replays and failure clusters;
- publication-quality analysis artifacts so lessons do not vanish into logs.

## Failure Modes

- **Compute theater:** spending money on bigger PPO before the evaluation lab
  can catch seat bias and stale rankings.
- **Teacher lock-in:** cloning a solver so hard that the learner cannot
  deviate on defense, finishing, or target switching.
- **Metric drift:** optimizing raw reward while replays get worse.
- **Temporal blindness:** improving edge logits while still re-deciding from
  scratch every tick.
- **Unbounded league sprawl:** adding opponents faster than the analysis can
  explain regressions.

## First Million-Dollar Move

Do not start with a giant run. Start with a ruthless 2-4 week campaign:

1. rerun vectorized matched-pair rankings;
2. confirm or replace `lightning_sum_throttled` as champion;
3. generate teacher/preference corpora from champion and close challengers;
4. build solver-residual policy prototype;
5. evaluate residual-vs-teacher under matched pairs;
6. only then decide whether the next dollar buys model scale, league scale, or
   better diagnostics.

If that campaign cannot produce a residual that beats its teacher somewhere
real, the correct grant move is not "more PPO"; it is to improve the game
formulation, diagnostics, or solver family until there is a learnable gap.

Related: [[v2-ml-gameplay-opportunities]], [[v2-temporal-strategy]],
[[v2-vectorized]], [[v2-overnight-research]], [[v2-edge-voting-policy]],
[[v2-training-runs]].
