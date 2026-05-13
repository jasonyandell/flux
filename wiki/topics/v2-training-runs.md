---
title: v2 training runs
kind: topic
first_seen: 2026-05-13
last_updated: 2026-05-13
status: active
---

## What this page is

A running record of meaningful v2 PPO training runs so future sessions can
pick up the thread without rerunning. Not every run is recorded — only
those that produced configuration insights, a useful checkpoint, or a
named result. Single-iteration smoke tests are not.

For the design itself, start at [[v2-edge-pressure-state]],
[[v2-set-clear-actions]], [[v2-three-term-reward]], [[v2-trainer-displayer]].

## Launching a run

Operational runbook so a new run lands in the right places — wandb,
the standard log path, and the displayer's replay drop — without
having to rediscover the conventions each time.

**Working directory:** always `python/`. The trainer's hardcoded
`DEFAULT_OUT_DIR` is `REPO_ROOT/public/v2/replays/`, derived relative
to the script, so any cwd technically works, but the venv lives at
`python/.venv` so it's least friction.

**Standard log path:** `/tmp/flux-train-v2.log`. The 10-min check-in
prompt (and the Monitor armings in this skill) all read from that
path — using anything else means the watchers go blind.

**Background launch template:**

```bash
nohup .venv/bin/python scripts/train_v2.py \
  --radius 9 --num-players 12 --games-per-rollout 4 \
  --max-ticks 10000 --num-dead-cells 40 \
  --power-coef 0.20 --power-damage-coef 0.1 --capture-coef 2.0 \
  --waste-coef 0.005 --kill-pressure-coef 0.3 --entropy-coef 0.003 \
  --win-bonus 500 \
  --fresh --wandb --wandb-run-name <run-name> \
  > /tmp/flux-train-v2.log 2>&1 &
```

The board / capacity / reward block above is the known-good
`v2-killer-tuned` recipe — start from it and add the one knob the
experiment is testing. Don't drift the unrelated flags between runs;
that's how comparisons get muddied.

Flags that matter for routing:
- **`--wandb`** + **`--wandb-run-name <name>`** — without `--wandb`
  metrics live only in stdout; without an explicit name wandb
  auto-generates one and you'll have to grep for it later. Project
  is hardcoded to `flux-v2` via `--wandb-project` (default).
- **`--fresh`** — start from scratch. Required after any reward-shape
  or state-representation change, otherwise the value head learned
  under the old semantics will mislead PPO. Resume with
  `--checkpoint python/checkpoints/v2/latest.npz` only when the
  reward signal hasn't moved.
- **stdout redirect to `/tmp/flux-train-v2.log`** — Python buffers
  stdout when redirected to a file, so don't expect the first iter
  line for ~1-3 min after launch. The wandb API has live data
  immediately; lean on it for early-iter status, not the log.

**Replay output is automatic:** `.flxr` files land in
`public/v2/replays/` plus an `index.json` the displayer polls. The
Vite dev server (`npm run dev`) serves them at `/v2/replays/` and
the `index-v2.html` displayer auto-discovers new replays — no
configuration needed at the UI side. **`.gitignore`** already
excludes `public/v2/replays/*.flxr` and the index, so a long run
won't pollute git.

**Iter cadence in the displayer:** the trainer drops a replay
roughly every iter at `record-stride=25`. Each is ~1.5 MB; long
runs accumulate but stay manageable. Force-add a notable replay
with `git add -f public/v2/replays/<file>.flxr` if you want it
preserved as a demo artifact.

**Watching a run:**
- **wandb API:** the canonical live source. Most-recent run in the
  project is `api.runs('jasonyandell-forge42/flux-v2',
  order='-created_at')[0]`; the summary dict has the per-iter
  metrics. Use this for cron-driven check-ins, not the log tail.
- **`tail -F /tmp/flux-train-v2.log`** — useful for spotting
  policy-loss thrash or KL spikes; cosmetically lags wandb by the
  buffer flush.
- **`ps -p <pid> -o stat,etime`** — `RN` / `SN` are healthy, `Z` is
  bad. Memory leaks here have been rare; the run-killer is usually
  KL or reward collapse, not OOM.
- **Displayer**: `npm run dev` (from repo root) and open the URL it
  prints with `/index-v2.html` appended. Existing replays show in
  the drip bar; new ones land within the poll interval.

**Stopping cleanly:** `kill -TERM <pid>`. The trainer handles SIGTERM
by writing a checkpoint (`python/checkpoints/v2/latest.npz`) and
flushing wandb. Don't `kill -9` unless it's truly stuck — you'll
lose the last 20-something iters of gradient state. Confirm
shutdown with `ps -p <pid>` (should exit within ~5 s).

**Cron / Monitor patterns this session has used:**
- The `/loop` skill armed a 10-minute cron job that issues a
  structured check-in prompt; cancel with `CronDelete` when done.
- The `Monitor` tool with
  `tail -F /tmp/flux-train-v2.log | grep -E "^iter ..." | head -3`
  is the right shape for "wake me at iter N or on crash" —
  alternation must include `Traceback|Error|Killed|OOM` so silent
  failures don't look the same as still-running.

## Overnight 2026-05-13 — summary

Tested seven configurations end-to-end. Headline result: the **best v2 PPO
config so far** is `v2-patient`'s recipe (HIDDEN=32, 3-layer GCN,
entropy_coef=0.003, power=0.20 / waste=0.015 / time=0.01 / win=200, 40
connected dead cells, 2000-AI-tick games). Achieved R30avg≈−215 at peak
(vs baseline −2725), routinely decisive games (alive_seats_end avg ~1.8),
single-iter peaks down to R=−38 (98% improvement from baseline).

**Run timeline:**

| # | run | trigger | what changed | outcome |
|---|-----|---------|--------------|---------|
| 1 | `v2-overnight` | initial | 2-layer GCN, ent=0.01, balanced power/waste | plateau iter ~70, entropy stuck at 2.31, waste flat |
| 2 | `v2-deeper-3hop` | breakout attempt | + 3-layer GCN, ent 0.01→0.003 | clear progress, plateau iter ~120 at ~R=−2400 |
| 3 | `v2-rebalanced` | reward shape | pwr 0.05→0.20, wst 0.05→0.015, win 50→200 | best plateau yet (R≈−530), decisive games, plateau iter ~40 |
| 4 | `v2-bigger-hidden64` | capacity | HIDDEN 32→64 | degenerate collapse: SET-everything in 4 iters |
| 5 | `v2-h64-ent01` | regularize | ent_coef 0.003→0.01 | same degenerate collapse, slower |
| 6 | `v2-patient` | patience | revert to v2-rebalanced recipe, commit to 500-iter floor | broke prior plateau, best operating point |
| 7 | (still running) | — | — | — |

**Big lessons (also in cross-run lessons below):**

1. **Reward-magnitude balance is the dominant lever.** Going from
   waste-dominant (power 0.05 / waste 0.05 / win 50) to growth-dominant
   (power 0.20 / waste 0.015 / win 200) flipped the policy from pacifist
   stalemate to decisive aggressor with no other change. The single
   biggest improvement of the night.

2. **Plateaus break with patience.** Killing at iter 40 was throwing away
   gradient progress. `v2-patient` ran iter 50–185 through what looked
   like a plateau and then surfaced new peaks (R=−97, −67, −38 single
   iters).

3. **Bigger isn't better here.** HIDDEN=64 with this action design
   collapsed into a degenerate SET-everything policy in <5 iters,
   regardless of entropy_coef. The 32-dim hidden + 3-layer GCN is the
   right capacity envelope.

4. **Trust rolling averages, not single iters.** Per-iter R swings ±200
   from board-luck. The 30-iter rolling averages are what told us the
   `v2-patient` plateau wasn't terminal.

5. **Dominance + alive_seats_end track policy capability better than R.**
   R is noisy and reward-coef-dependent. Dominance and alive_seats_end
   are direct game-outcome measures and improved monotonically through
   most of the night.

6. **3-hop GCN was a real unlock.** The 2-layer ceiling was structural —
   deep upstream source cells couldn't see wasteful terminators 3+ hops
   away. Adding the third MP layer let entropy actually collapse and the
   policy start discriminating between productive and wasteful outflows.

**Still open at end of overnight:**

- `v2-patient` showed a sustained regression iter 200–235 (R30avg slid
  −215 → −308 over 55 iters, action distribution drifting toward
  over-clearing). Restarted from the iter-235 checkpoint as
  `v2-patient-recover` with `entropy_coef 0.003 → 0.006`. **This broke
  the policy in 1 iter** — the new loss landscape pushed PPO into a
  pacifist over-clear equilibrium (alive_seats_end 1.5→11, dominance
  0.91→0.21, set 40%→22%, clear 57%→68%) and it stayed there.
  Counter-intuitively, the higher entropy_coef found a *lower*-entropy
  solution (1.88 vs 2.30) that happened to satisfy the new loss shape.
  Killed `v2-patient-recover` after 23 iters and fresh-restarted as
  `v2-patient-2` with v2-patient's exact recipe.

  **Lesson:** changing hyperparameters mid-flight is risky — the new
  loss landscape can have its own local mins that the optimizer falls
  into immediately. If you want to perturb a trained policy, do it from
  fresh weights, not from a near-converged checkpoint.
- Per-cell value head was not tried — currently value head is mean-
  pooled per-seat, which dilutes credit assignment. Likely next
  intervention if `v2-patient-recover` doesn't recover.
- **Default `record_stride` raised 1 → 25** in `train_v2.py` for future
  runs. Tick-by-tick replays were 32 MB each; stride=25 drops to ~1.5 MB
  with no loss of trend info. Pass `--record-stride 1` to opt back into
  play-by-play resolution.

## Cross-run lessons (frontier knowledge)

Updated as runs accumulate. These claims are accountable to current code +
the most recent run that exercised them.

- **Value head locks in fast.** `explained_variance` reaches ~0.97 by iter
  ~15 on this scale (radius=9, 12 seats, G=4). Long before the policy is
  coherent. Useful early sanity check: if `ev` is still near zero past
  iter ~25, something is wrong.
- **Iter cadence at radius=9, max_ticks=10000:** ~15 s/iter for the 2-layer
  GCN, ~17 s/iter for the 3-layer GCN (extra MP layer adds ~2-3 s mostly to
  the update step). `minibatch_size=512`, `update_epochs=4`. The first iter
  is ~50 s due to MLX kernel compilation — extrapolating from iter 1 alone
  is misleading.
- **`max_ticks` matters for stalemate exposure.** At 5000 game ticks
  (1000 AI ticks), seats survive the full game on average. Doubling to
  10000 doesn't shorten games meaningfully — they keep running to the
  cap — but it gives the policy 2× the exposure to long-game
  waste-minimization scenarios.
- **Pressure features changed the input dim** (`IN_DIM` 6 → 9). The new
  channels are `pressure_in_friendly`, `pressure_in_enemy`,
  `pressure_out`, all seat-relative, normalized by `MAX_EDGE`. Adds ~5 s
  to the rollout step (single MLX gather + sum); checkpoint-incompatible
  with the 6-channel weights.
- **Connectivity-guaranteed dead cells.** `random_seat_and_dead` now
  rejects any dead-cell candidate that would disconnect the live
  subgraph. At radius=9 the algorithm can place up to ~67 % dead before
  the connectivity guard starts rejecting candidates. 50 % is freely
  achievable.
- **State-only reward terms are dead weight under PPO.** A reward
  proportional to a state quantity that barely changes per AI tick
  (`power_held_coef * cells_owned`) is absorbed entirely by the value
  baseline — `EV` saturates near 1, advantages collapse to ~0, and the
  policy stops learning. Every reward term has to be *action-conditioned*
  (Δ of state, or directly produced by the action like `damage_dealt`).
  Killed `v2-composite-power-2` at iter 52 for this exact reason
  (entropy frozen at 2.55, KL shrinking each iter).
- **`entropy_coef` ≥ policy gradient = no learning.** At
  `entropy_coef=0.01`, the entropy bonus `0.01 × log(13) ≈ 0.0256` per
  state is ~5× the observed `policy_loss` magnitude (~0.005) for fresh
  policies on this game. The entropy term dominates → policy stays at
  uniform → no commitment. Known-good value is **`entropy_coef=0.003`**
  (used by `v2-patient`, which actually learned). Diagnostic: if
  `policy_loss / (entropy_coef × log(NUM_ACTIONS))` ≲ 0.5 after iter ~10
  on a fresh init, lower `entropy_coef`. Caused successive failure of
  `v2-composite-power-2` AND `v2-action-driven` (both used 0.01).
- **Terminal rewards are mostly invisible at γ=0.99 over multi-thousand-tick
  games.** `gamma=0.99` gives effective horizon ~100 AI ticks (γ^100 ≈ 0.37).
  Games run 400 AI ticks (2000 game ticks / period 5). A terminal `win_bonus=500`
  arrives at tick 0 with weight ≈ 500 × 0.99^400 ≈ 9. By tick T-200 it's
  ≈ 60. The policy almost never sees terminal rewards in early-game decisions.
  Fix: convert anything you want optimized toward into a per-tick signal that
  GAE can backpropagate. Diagnostic: if won-game rollouts have lower R than
  stalemates, the terminal reward isn't doing work — promote it to per-tick.
  Spotted in `v2-stacked-actions` at iter 158 (won game R < iter 157 stalemate
  R); addressed by adding `kill_pressure_coef` in [[v2-killer-instinct]].
- **Event bonuses need attribution.** A "thing happens, everyone gets +reward"
  signal (like the *symmetric* kill_pressure in `v2-killer-instinct-3`) acts
  as a state-value signal: PPO's value head absorbs the predictable bump,
  advantages collapse, policy gradient flattens. The fix is to point the
  reward at the specific seat whose actions caused the event (in this case,
  whoever owns the most of the dying seat's previous-tick cells). Attribution
  turns a state-value signal into an action-conditioned one. See
  `v2-killer-attributed` for the corrected version.
- **Strong "kill" reward without a "waste" penalty drives set-spam in
  symmetric multi-agent.** `v2-killer-attributed` (kill_pressure_coef=1.0,
  waste_coef=0.001) looked great through iter 15 (dominance 0.81, games
  ending early) then over-rotated: by iter 47, action mix had collapsed to
  set 60% / clear 37% / noop 2%, dominance crashed to 0.27, alive_seats=10.
  All 12 seats playing hyper-aggressive → no one wins → kill_pressure stays
  near zero → policy stuck in low-R local min. Need to *balance* the kill
  carrot with a waste penalty so wasted aggression has cost. Sweet spot for
  this game scale: kill_pressure_coef ~0.3 + waste_coef ~0.005 (tested in
  `v2-killer-tuned`).

## Runs

### v2-overnight — 2026-05-13 (killed @ iter ~70, plateaued)

**wandb:** `jasonyandell-forge42/flux-v2/6xjpm2ld` (name: `v2-overnight`).

**Config (frozen):** radius=9, num_players=12, G=4, max_ticks=10000,
num_dead_cells=40 (connectivity-guaranteed), record_stride=1,
minibatch_size=512, update_epochs=4, lr=3e-4. 9-channel pressure-aware
policy, **2-layer GCN**, entropy_coef=0.01,
power=0.05/waste=0.05/time=0.01/win=50.

**Final state (iter 69):** R=−2610, pwr=93.5, wst=−2683, ev=0.98,
ent=2.36, noop 3.3% / set 56% / clear 41%, dominance 0.56.

**Outcome:** plateaued by iter ~30 and didn't break out. Value head locked
in fast (`ev` reached 0.98 by iter 15 and held), but the policy never
committed — entropy floored at ~2.31 (uniform = 2.56) and waste hung
around −2700 with no downward trend.

**Diagnosis:** credit assignment from a wasteful chain terminator back to
the upstream cells generating the pressure is weak under a 2-hop GCN +
mean-pooled value head. The waste signal reaches the seat's value
estimate but doesn't sharpen the policy logits at the offending source
cells. Plus `entropy_coef=0.01` was too high — kept the policy near
uniform.

**Lessons (added to cross-run lessons above):**
- A 2-hop GCN may be structurally too shallow for long-chain coordination
  on a 271-cell board.
- Reward magnitude balance matters: at the run's defaults, waste (~−2800)
  totally dominated power (~+94). Policy mostly saw "waste is bad" and
  hardly saw "growth is good."

---

### v2-deeper-3hop — 2026-05-13 (killed @ iter ~122, second plateau)

**wandb:** archived `jasonyandell-forge42/flux-v2` run name `v2-deeper-3hop`.

**Changes vs v2-overnight:** 3-layer GCN (added `w3_self` + `w3_neigh` in
`ppo.py`, receptive field 3 hops); `entropy_coef 0.01 → 0.003`.

**Final state (iter 122):** R=−2409, pwr=91, wst=−2481, ev=0.98, ent=1.87,
noop 3% / set 31% / clear 66%, dominance 0.29, alive_seats_end 10.75/12.

**Outcome:** clear improvement over v2-overnight on every axis — R came
down ~18%, waste down ~17%, entropy actually collapsed (2.31 → 1.87),
action distribution flipped to SET 31% / CLEAR 66% (was 56/41), some
seats start eliminating others (alive ~10 vs ~3). Then plateaued by
iter ~90: incremental drift but no breakout.

**Diagnosis:** the 3-hop GCN + commit-friendly entropy were necessary but
not sufficient. The reward shape now bites: waste term still dominates
power magnitude-wise (pwr ~91 vs wst ~−2300, 25× larger). Policy
converges to "play defensively, minimize my contribution to global
waste" instead of "win by attacking" — because the win bonus (50) is
dwarfed by the wst differential between aggressive and defensive play.

**Lessons (added to cross-run lessons above):**
- Going from 2-hop to 3-hop GCN was a real unlock — entropy could
  actually collapse, alive_seats_end dropped from ~12 to ~10. Worth keeping.
- `entropy_coef 0.003` is fine — entropy floored at ~1.87 (not collapsed
  to a single action). Below 0.001 risks single-action lock-in.
- Reward magnitudes need to be comparable across terms or PPO will optimize
  the biggest one and ignore the rest. waste ≫ power ≫ time made the
  policy a pacifist.

---

### v2-rebalanced — 2026-05-13 (killed @ iter ~40, third plateau)

**wandb:** archived `v2-rebalanced`.

**Final state (iter 40):** R=−532, pwr=377, wst=−889, ev=0.94, ent=2.19,
noop 7% / set 60% / clear 33%, dominance 0.85, alive_seats_end 3.25/12.

**Outcome:** the rebalance worked — best run yet. Games went decisive
(alive ~3 vs ~10 in v2-deeper-3hop), dominance climbed to 0.85, the
policy actively engages (set 60% vs the prior pacifist 31%). R improved
from baseline −2725 to −500 range — ~80 % of the headroom we had to
play with on this reward scale.

**Outcome (cont):** then plateaued by iter ~20. Waste stuck at −900,
entropy stuck at 2.20, all action fractions frozen. The policy found a
working equilibrium (be aggressive, accept some waste) but didn't refine
further. Same plateau pattern as previous runs, just at a much better
operating point.

**Diagnosis:** with the reward shape now balanced, the bottleneck is
probably **representational capacity**. HIDDEN=32 + 3-layer GCN can't
distinguish the productive vs wasteful outflows finely enough to learn
to prune. The signal that "this particular outflow leads to a wasteful
terminator vs that one leads to an enemy capture" requires per-cell
discrimination the current MLP head can't sustain.

**Lessons (added to cross-run lessons above):**
- Reward magnitude balance is the single biggest lever I've seen so far.
  Going from waste-dominant to growth-dominant (with proper waste-as-
  guardrail) flipped the policy's equilibrium from "pacifist stalemate"
  to "decisive aggressor" with no other change.
- `power_coef=0.20, waste_coef=0.015, win_bonus=200` looks like a workable
  baseline for v2 self-play at radius=9, 12 seats. Keep for follow-up runs.

---

### v2-bigger-hidden64 — 2026-05-13 (killed @ iter ~21, degenerate collapse)

**wandb:** archived `v2-bigger-hidden64`.

**Final state (iter 20):** R=−339, pwr=372, **wst=−691**, ev=0.93,
**ent=0.60**, **noop 0.5% / set 93% / clear 6%**, dominance 0.34,
alive_seats_end 11.

**Outcome:** the bigger network collapsed entropy hard (2.56 → 0.56 by
iter 4) and locked into "SET almost every action, never CLEAR." That
single-strategy policy is pacifist — everyone sets all outflows in all
directions, attacks cancel via friendly bidirectional resolution, no
one gets captured (alive ≈ 11). Worse than v2-rebalanced on the
"actually playing the game" axis.

**Diagnosis:** with HIDDEN=64 the policy logits can be sharpened twice
as easily. `entropy_coef=0.003` was sufficient for the 32-dim network
but not the 64-dim one — same regularization, half the relative pressure
against extreme logits.

**Lessons (added to cross-run lessons above):**
- Entropy regularization must scale with network capacity. Doubling
  HIDDEN without doubling `entropy_coef` is asking for premature commitment.
- "SET-everything" is a strong attractor: with 6 SET + 6 CLEAR + 1 NOOP
  and a single-action-per-tick budget, SET dominates because each cell
  starts unset and SET is irreversible until a future CLEAR — so the
  policy that just hits SET on every cell accumulates 6/6 outflows
  quickly. Worth watching for in future runs.

---

### v2-h64-ent01 — 2026-05-13 (killed @ iter ~22, same collapse)

**wandb:** archived `v2-h64-ent01`.

**Final state (iter 22):** R=−430, pwr=280, wst=−690, ev=0.95,
**ent=0.59**, noop 0.5% / **set 94%** / clear 5%, dominance 0.34,
alive_seats_end 9.

**Outcome:** same SET-everything collapse as v2-bigger-hidden64. Bumping
`entropy_coef` 0.003 → 0.01 didn't prevent it. Power got *worse* over
training (375 → 276) as the policy over-spilled. HIDDEN=64 is wrong-
direction for this problem.

**Lessons (added to cross-run lessons above):**
- HIDDEN=64 with the current action design is a worse operating point
  than HIDDEN=32, regardless of entropy_coef. The SET-everything
  attractor is faster to find with more capacity.
- I've been killing runs too aggressively at first sign of plateau —
  reverting to the best variant (`v2-rebalanced`'s recipe) and letting
  it cook with patience.

---

### v2-patient — 2026-05-13 (in flight, breakthrough @ iter ~90)

**wandb:** `v2-patient` (newest in flux-v2 project).

**Config:** identical to `v2-rebalanced` — HIDDEN=32, 3-layer GCN,
entropy_coef=0.003, power_coef=0.20, waste_coef=0.015, win_bonus=200,
40 dead cells, 2000 AI ticks/game, radius=9, 12 seats.

**Commitment:** at least 500 iters before any intervention — wanted to
test whether prior plateaus were patience-bound vs architecture-bound.

**Progress (30-iter rolling averages):**

| iter | R30avg | waste30avg | dom30avg | alive30avg | ent30avg |
|------|--------|------------|----------|------------|----------|
| 20   | −400   | −720       | 0.74     | 2.75       | 2.28     |
| 50   | −365   | −720       | 0.79     | 2.13       | 2.34     |
| 75   | −364   | −720       | 0.78     | 2.18       | 2.35     |
| 90   | −345   | −702       | 0.78     | 2.22       | 2.36     |
| 95   | −330   | −688       | 0.79     | 2.17       | 2.36     |
| 106  | −313   | −672       | 0.84     | 2.09       | 2.36     |

**Outcome so far:** the *patience-bound* hypothesis held. The first plateau
(R≈−350) wasn't terminal — it broke around iter ~85, and rolling averages
have moved on every metric since. By iter 106 the policy has reached the
strongest operating point of any run yet: dominance averaging 0.84,
alive_seats_end averaging 2.09 (vs v2-rebalanced's iter-40 best of
dominance 0.85, alive 3.25). Games are routinely ending early.

**Lessons (added to cross-run lessons above):**
- **Plateaus break with patience.** Killing runs at iter 40 and starting
  over was throwing away gradient progress. The same recipe ran 100+
  iters reaches a meaningfully better operating point.
- **Trust rolling averages, not single iters.** Per-iter R bounces ±100
  on different board seeds; 30-iter rolling averages cut through to the
  signal.
- **Dominance + alive_seats_end are better progress signals than R.**
  R fluctuates with board luck; dominance and alive_seats_end track the
  policy's actual competitive ability and they were already improving
  during the iter 50-85 "plateau" that single-iter R hid.

---

### v2-no-waste — 2026-05-13 (peak R=+1290, healthy aggression)

**Config:** waste_coef=0 (the constraint removed entirely), power_coef=0.20,
win_bonus=200, otherwise v2-patient recipe.

**Outcome:** dominance 0.66, alive_seats_end 2.25, interior_max_frac=1.0
(back-line loops fully developed). First v2 run with positive rewards.
Demonstrated that the policy *can* learn aggressive play once the waste
constraint is lifted — but waste was the term distinguishing wasteful
spam from purposeful pressure, and removing it sacrificed signal quality.

---

### v2-regen-waste — 2026-05-13 (R=1405, dominance 0.76)

**Config:** waste_coef=0.001 (waste as a tiny guide, not a constraint),
new waste rule: "any regen you didn't send is waste" (only no_spill
counts; cap-bound regen is not penalized — see [[v2-three-term-reward]]).

**Outcome:** R=1405, cap=526 (single-iter highs), dominance 0.76. Best
overall when blending the no-waste's aggression with a faint penalty
against pure dead-end chains. Established the operational point for
subsequent reward-shape experiments.

---

### v2-composite-power-2 — 2026-05-13 (killed @ iter 52, advantage signal vanished)

**wandb:** `jasonyandell-forge42/flux-v2/r2su225c` (name `v2-composite-power-2`).

**Config:** introduced composite power: `r_power = power_held_coef * cells_owned + power_damage_coef * damage_dealt`. Coefs: `power_held_coef=0.02`,
`power_damage_coef=0.01`, `waste_coef=0.001`, `win_bonus=500`. Hypothesis: in
prior runs `Δ(strength_owned)` saturated at MAX so steady-state policies got
zero power signal; composite power should keep the gradient alive after
territory is developed.

**Final state (iter 52):** R=692, pwr=767, wst=-56, t=-20, **entropy=2.551**
(uniform=2.565 → ~zero policy commitment), KL=0.016, pol_loss=0.010 (both
shrinking monotonically), EV=0.99, action mix flat at noop 8% / set 47% /
clear 45% since iter ~5.

**Diagnosis:** the `power_held_coef * cells_owned` term is a **state value
not action value** — cells held barely change per AI tick, so the value head
absorbs the ~770 baseline perfectly (EV=0.99) and per-state advantages
collapse to ~0. Policy gradient vanishes. Action mix never specializes.
The damage term *was* action-conditioned but dwarfed by held; the run never
escaped baseline.

**Lessons (added to cross-run lessons above):**
- State-only reward terms (raw quantities, not deltas) are PPO poison.
- Diagnostic for a dead run: entropy near `log(NUM_ACTIONS)` ± 0.01 after
  20+ iters, KL strictly shrinking, EV ≥ 0.98. That triple = no learning.
- The 97%-noop trap from earlier `capture_coef=50` runs was the *opposite*
  failure: too much action signal, too punctuated, drove the policy into
  a "wait for free captures" local min. Both extremes break.

---

### v2-action-driven — 2026-05-13 (killed @ iter 13, same freeze)

**wandb:** `jasonyandell-forge42/flux-v2/iqfcr5ic` (name `v2-action-driven`).

**Config:** all action-conditioned terms — `power_damage_coef=0.1`,
`capture_coef=2.0`, `waste_coef=0.001`, `time_coef=0.01`, `win_bonus=500`,
**`entropy_coef=0.01`**. `power_held_coef=0.0`.

**Final state (iter 13):** R=325, pwr=309, cap=92, wst=-56, t=-20.
entropy=2.558, KL=0.008 (tiny), pol_loss=0.005 (tinier), EV=0.95,
dominance 0.63, alive_seats_end 2.75. Action mix noop 8.5/set 45/clear 46
(uniform).

**Diagnosis:** even with every reward term properly action-conditioned,
the policy still froze. Root cause turned out NOT to be reward shape but
**`entropy_coef=0.01`**: the entropy bonus magnitude (`0.01 × log(13) ≈
0.026`) was ~5× the policy_loss (0.005), so the entropy term *dominated*
the loss and the policy stayed at uniform. The successful `v2-patient`
run used `entropy_coef=0.003`. Lowering this is the missing piece.

**Lesson (added to cross-run lessons above):** check
`policy_loss / (entropy_coef × log(NUM_ACTIONS))`; if it's much less than
1, entropy term is suppressing learning.

---

### v2-stacked-actions — 2026-05-13 (killed @ iter ~160, learning but lacking killer instinct)

**wandb:** `jasonyandell-forge42/flux-v2/vejwyjxv` (name `v2-stacked-actions`).

**Config:** all known-good action-conditioned signals stacked + lower
entropy:
- `power_coef=0.20` (legacy Δ(Σ strength_owned) — v2-patient's winning
  signal; action-conditioned via outflow setting cells to grow)
- `power_damage_coef=0.1` (continuous damage signal)
- `capture_coef=2.0` (event credit on cells gained)
- `power_held_coef=0.0` (the dead state-only term — stays off)
- `waste_coef=0.001` (light guide; "any regen not sent" rule)
- `time_coef=0.01`, `win_bonus=500`
- **`entropy_coef=0.003`** (v2-patient's value, fixes the freeze)

**Final state (iter ~160):** R peak 881 (iter 110, new highs throughout
run from baseline ~700). Captures peaked 140. Entropy 2.554 → 2.495
over 150 iters. KL/pol_loss healthy. Dominance volatile (0.6-0.9).

**Outcome:** **first v2 run that actually learned end-to-end.** The diagnosis
chain (state-value terms → frozen entropy → entropy_coef too high) was
correct. Stacking three action-conditioned signals + entropy_coef=0.003
broke out of the iter-13 freeze in [[v2-action-driven]].

**But three failure modes spotted in visualization (Jason, ~iter 160):**
1. **No killer instinct.** Policy lets weak enemies survive; they come
   back later to harass while it fights elsewhere. Finishing a wounded
   seat does not pay off in the current reward.
2. **Pipes pressure to friends.** Visualization shows outflows on
   interior friendly cells pointing into other friendly cells.
   `damage_dealt` only *rewards* non-friendly outflow; friend-to-friend
   is reward-neutral, so the policy doesn't bother clearing them.
3. **R not aligned with winning.** Iter 158 *won* its game but had
   lower R than iter 157 which stalemated — terminal `win_bonus=500`
   is washed out by GAE discount: with γ=0.99 over a 2000-tick game,
   the terminal bonus reaches tick 0 with weight ≈ 1e-9. Effectively
   invisible. The policy chases per-tick power, not victory.

**Lesson (added to cross-run lessons above):** terminal rewards are
mostly invisible to PPO at γ=0.99 over multi-thousand-tick games.
Anything you want the policy to optimize toward must be a per-tick
signal that GAE can backpropagate.

---

### v2-killer-instinct — 2026-05-13 (killed @ iter 3, kill term too hot)

**wandb:** `jasonyandell-forge42/flux-v2/gt8zftxo` (name `v2-killer-instinct`).

**Config change:** `--kill-pressure-coef 1.0` (all else like v2-stacked-actions).

**Outcome:** killed after 3 iters. `kill=11047` was 95% of `R=11777`,
swamping power (672) and capture (100). KL spiked to 0.32 at iter 2 —
policy thrashing under huge advantage variance.

**Lesson:** for `num_players=12`, peak kill contribution scales as
`P × kill_pressure_coef × T_remaining`. Sanity-check against power
magnitude when picking the coef. `0.05` gives ~575/rollout (balanced
with power ~700) — that's the operating point.

---

### v2-killer-instinct-2 — 2026-05-13 (killed @ iter 2, restart for metrics)

**wandb:** `jasonyandell-forge42/flux-v2/5uo5x3us`.

**Config change:** `--kill-pressure-coef 0.05`. Confirmed magnitudes
reasonable (R=1271, pwr=673, kill=575). Killed only to incorporate the
total-edge-pressure metric — same recipe restarted as `-3`.

---

### v2-killer-instinct-3 — 2026-05-13 (killed @ iter 54, attribution problem)

**wandb:** `jasonyandell-forge42/flux-v2/apw0hhtb` (name `v2-killer-instinct-3`).

**Config:** identical to `v2-stacked-actions` + symmetric kill pressure
(`--kill-pressure-coef 0.05`) + new structural metrics
`total_edge_pressure_end`, `max_edge_pressure_end`.

**Reward term:** per AI tick, every alive seat received
`kill_pressure_coef × num_dead_seats_in_game` reward — same to all
surviving seats. Magnitudes balanced: R=1280 with kill=575, pwr=669
at iter 1.

**Final state (iter 54):** R=1386, peak iter 51 hit 1471. Kill term
533-712 across iters. **Entropy STUCK at 2.550-2.551 for 27 iters**
(iter 28→54 no movement). KL=0.011 shrinking, pol_loss=0.009.
total_edge_pressure_end declined 593→491 (policy uses fewer edges
modestly vs random's piled-up edges).

**Diagnosis:** symmetric kill_pressure is a **state-value signal**, not
action-conditioned. When seat q dies, every surviving seat p gets the
same +reward — so PPO can't tell which of p's actions caused the kill.
Value head absorbs the predictable bump (EV=0.91), advantage collapses
to ~0 again. Same failure mode as `v2-composite-power-2`'s
`power_held_coef × cells_owned`. Diagnostic ratio `pol_loss / (entropy_coef
× log NUM_ACTIONS) = 0.009/0.0077 = 1.17` (vs `v2-stacked-actions`'s
3.0 at the same iter) confirmed the gradient was sub-threshold for
overcoming entropy regularization.

**Lesson (added to cross-run lessons above):** event-based bonuses
need *attribution* to drive learning. "Everyone benefits when X
happens" creates a state-value signal that disappears into the
baseline. Credit must point at the specific seat whose actions caused
the event.

---

### v2-killer-attributed — 2026-05-13 (killed @ iter 47, set-spam collapse)

**wandb:** `jasonyandell-forge42/flux-v2/dxwmrb4v` (name `v2-killer-attributed`).

**Reward term — attributed kill pressure:** when seat q dies, the seat
that ended up owning the most of q's previous-tick cells is credited
with the kill. Per-seat counter `kills_per_seat[g, p]` increments, and
that seat receives `kill_pressure_coef × kills_per_seat[g, p]` per AI
tick thereafter. Now action-conditioned.

**Config:** `--kill-pressure-coef 1.0` (20× the symmetric version since
attribution concentrates signal on one seat per kill).

**Phase 1 (iter 1-15) — looked great:**
- R 1672→1725, kill 960→1011 (dominant signal, 57% of R)
- Diagnostic ratio `pol_loss / (entropy_coef × log13)` = 9.6 (vs the
  symmetric run's stuck-at-1.17). Gradient was strong.
- By iter 15: dominance 0.81 (up from 0.74), alive_seats_end=2 (down
  from 2.5), time penalty -18.8 (games ending sooner on ~25% of
  rollouts). **Aggression metrics all improving.**

**Phase 2 (iter 16-47) — over-rotation collapse:**
- R 1663→544 (-67%), pwr 649→290, cap 93→22, kill 993→275
- entropy collapsed 2.55→2.08 (Δ=-0.47 in 30 iters)
- **Action mix flipped to noop 2% / set 60% / clear 37%** — committed
  to set-spam local min
- **dominance crashed 0.81→0.27, alive_seats_end ballooned 2→10**
  Games end with 10/12 seats still alive
- KL settled near 0 (0.004), pol_loss → 0 — policy locked in

**Diagnosis:** kill_pressure_coef=1.0 was too aggressive. Early
gradient correctly pushed toward elimination (Phase 1), but then over-
rotated into "spam SET, hope someone dies" — and with all 12 seats
playing this way, **no one actually kills anyone**, so kill_pressure
stays near zero and the policy is stuck in a low-R local min. Same
shape as `v2-bigger-hidden64`'s set-everything collapse (different
cause though — there it was capacity; here it's reward magnitude).

**Lesson (added to cross-run lessons above):** rewarding "kill the
enemy" too strongly in a symmetric multi-agent setting drives every
seat into hyper-aggression. Without enough penalty for *wasted*
aggression (waste_coef was 0.001 — basically off), set-spam becomes
the equilibrium and no one wins. Either: (a) lower kill_pressure_coef
to balance with power, or (b) raise waste_coef so spam carries cost.
Both for next run.

---

### v2-no-dead-ends — 2026-05-13 (killed @ iter ~107, combo signal lit)

**wandb:** `jasonyandell-forge42/flux-v2/ll0s4j8j` (name
`v2-no-dead-ends`).

**Config:** identical to `v2-killer-tuned` (kill_pressure_coef=0.3,
waste_coef=0.005, entropy_coef=0.003, power_coef=0.20,
power_damage_coef=0.1, capture_coef=2.0, win_bonus=500) **plus** the
new destination-terminated waste category in
`python/flux_v2/state.py`: `WASTE_WEIGHT_DEST_TERMINATED=0.3`. Fresh
start.

**Hypothesis:** v2-killer-tuned learned everything except "don't ship
pressure into a dead-end MAX cell." Added explicit waste attribution
to source whenever its active outflow lands on a friendly cell that
is BOTH at MAX strength AND has zero active outflows (a pure sink).
Pass-through MAX cells with outflows are combo relays and stay
un-penalized — the rule is careful not to break the combo pattern.

**Outcome — combo signal lights up, structural metric crosses random
baseline.** Stopped at iter 107 (~52 min) per plan to compact + think.

| metric | iter 13 | iter 34 | iter 55 | iter 70 | iter 92 | iter 106 |
|---|---|---|---|---|---|---|
| R (mean) | 1468 | 1465 | 1498 | 1390 | 1541 | 1504 |
| pwr | 1469 | 1468 | 1446 | 1350 | 1431 | 1414 |
| waste | −314 | −318 | −333 | −343 | −349 | −351 |
| kill | 236 | 235 | 302 | 305 | 368 | 355 |
| cap | 97 | 99 | 103 | 97 | 111 | 106 |
| dominance | 0.54 | 0.70 | 0.65 | 0.80 | 0.76 | 0.69 |
| alive_end | 3.50 | 3.25 | 2.50 | 2.00 | 2.00 | 2.50 |
| **total_edge** | 488 | 460 | 470 | 599 | **756** | 666 |
| entropy | 2.558 | 2.558 | 2.557 | 2.556 | 2.555 | 2.554 |
| action mix (n/s/c) | 8/47/45 | 8/47/45 | 8/47/45 | 8/48/44 | 8/49/44 | 8/49/44 |

**Run-wide peaks** (across 107 iters, sampled history): R=1587,
kill=368, capture=121, total_edge_pressure=805, dominance=0.95.

**Highlights:**
- **total_edge_pressure crossed random baseline 638 at iter ~70 and
  peaked at 805** — the combo signal Jason flagged is firing. The
  dest-terminated rule didn't break combos; structural flow improved.
- **No collapse, no entropy decay** — entropy stayed at 2.55+ for 107
  iters under the new term. The 0.3 weight is safe; could go higher.
- **Kill rate landed**: kill reward 236 → 368 (+56 %) over the run;
  alive_seats trended 3.5 → 2.0 → 2.5 (kills happening, then policy
  defending; not pacifist).
- **Waste at steady-state ~−330 to −350** — much larger negative
  magnitude than v2-killer-tuned (~−80 in same range), proving the
  new term is firing meaningfully. R climbs *anyway* — the policy
  trades the penalty for the larger positive.

**The "leftmost cram" pathology** that motivated this run — pressure
crammed into dead-end MAX edges — should be visibly reduced in
late-iter replays; verify in the displayer before the next run.

**Killed:** at iter 107 to compact and think about the next change.
Idea queued: [slime-mold transit credit](#slime-mold-transit-credit)
in the section below — a *positive* local signal to mirror the
dest-terminated *negative*, so the policy has somewhere to flow TO
rather than just away from.

**Lessons (added to cross-run lessons above):**
- Dest-terminated waste is a safe, productive addition at weight 0.3.
  Combo signal (`total_edge_pressure_end`) is the right structural
  metric to verify the rule isn't suppressing relays.
- One-sided penalties teach what *not* to do but leave the policy
  without a clear local positive alternative. Future reward additions
  should pair penalties with symmetric positives where possible
  (slime-mold principle — see queued ideas).

### v2-killer-tuned — 2026-05-13 (killed @ iter ~260, best run of session)

**wandb:** `jasonyandell-forge42/flux-v2/oyfcui8g` (name `v2-killer-tuned`).

**Config:** `--kill-pressure-coef 0.3` + `--waste-coef 0.005` (3.3× less
kill aggression, 5× more waste than v2-killer-attributed). Everything
else: entropy_coef=0.003, power_coef=0.20, power_damage_coef=0.1,
capture_coef=2.0, win_bonus=500.

**Outcome — the breakthrough run of the session.** Stable trajectory
from iter 1, no collapse, multiple R>1000 breakthroughs after iter ~200.

| metric | iter 30 | iter 100 | iter 200 | iter 250 |
|---|---|---|---|---|
| R (mean) | 875 | 880 | 950 | 970 |
| R peak | 882 | 950 | 1089 | 1089 |
| pwr peak | 728 | 760 | 849 | 849 |
| cap peak | 117 | 121 | 130 | 130 |
| kill peak | 340 | 391 | 419 | 419 |
| entropy | 2.553 | 2.540 | 2.515 | 2.508 |
| dominance | 0.79 | 0.81 | 0.90 | 0.90 |
| alive_end | 2.25 | 2.25 | 1.50 | 1.50 |
| total_edge | — | 638 | 1014 | 1014 |

**Highlights:**
- **R > 1000 in 7+ iters** (216 = 1089, 230 = 1042, 242 = 1033, 248 =
  1029, 250 = 1039, 252 = 1045, 255 = 1032, 248). R floor moved from
  ~800 (early) → ~970 (late).
- **All components peaked**: power 849, capture 130, kill 419.
- **total_edge_pressure crossed 1000** (1014, +59 % over random
  baseline ~638) — the combo signal Jason's metric was designed to
  detect.
- **Action mix stayed balanced** through 260 iters (no spam collapse;
  noop drifted 6 % → 11 %, set 51 % → 44 %, clear 43 % → 45 %).
- **Entropy decayed past 2.51** — slowest decay of any successful
  signal, but monotonic over the run.

**Pathology spotted (Jason, ~iter 253 viz):** dominant red player crams
pressure leftward into MAX cells that have no outflows. Pressure lands,
gets clipped on absorption, dies. Current waste rule weights this at 0.

**Killed:** to apply the destination-terminated waste rule (see Queued
for next restart). Saved checkpoint `python/checkpoints/v2/latest.npz`
at the kill point.

**Lessons (added to cross-run lessons above):**
- `kill_pressure_coef=0.3 + waste_coef=0.005` is the working operating
  point for the v2 reward stack — preserves aggression without
  spam-collapsing the policy.
- Per-tick attribution + small magnitude beat per-tick symmetric large
  magnitude (vs v2-killer-instinct-3 / v2-killer-attributed).
- Plateau-then-breakthrough pattern: the run looked stuck at ~950 for
  ~30 iters before breaking past 1000 cleanly. Patience pays.

## Queued ideas

### Slime-mold transit credit (implemented, next run)

**Observation:** the dest-terminated waste rule tells the policy
*not to ship pressure into a dead end* — a clear local penalty.
But that's a one-sided gradient: the policy learns the cost without
seeing the alternative. Slime molds find efficient global structure
from purely *positive* local rules (pheromone reinforcement on
successful paths); they never compute "don't do X." Symmetry suggests
v2 should have a local positive for *the productive thing* — the
combo relay — to give the policy a "ship to relay" gradient instead
of just "don't ship to sink."

**Implemented rule:** *transit credit*. Pay the source cell when its
active outflow lands on a friendly cell **that has active outflows
of its own** (so the inbound pressure can propagate). Same
observation window as dest-terminated, opposite sign:

| Destination state | Today | With transit credit |
|---|---|---|
| Friendly, MAX, no outflows | dest-terminated waste (−) | unchanged |
| Friendly, MAX, has outflows | unchanged (no waste) | **NEW positive** |
| Friendly, < MAX, has outflows | absorbed productively | unchanged in strict mode |

Decision for first run: strict "relay" case only. The destination must be
friendly, have active outflows, and already be at MAX strength. This is closer
to the combo-detection signal Jason cares about and avoids rewarding ordinary
fill traffic.

**Implementation:**

- `python/flux_v2/state.py`: `TRANSIT_CREDIT_STRICT=True`.
- `python/flux_v2/step.py`: pure diagnostic
  `transit_credit_per_cell_for_tick`.
- `python/flux_v2/mlx_step.py`: `tick_batched` now returns
  `transit_credit_per_cell` alongside `waste_per_cell`.
- `python/scripts/train_v2.py`: `--transit-coef` (default off) adds
  `r_transit` and logs `reward_transit_iter`; first planned value is `0.1`.
- Tests cover strict relay credit and MLX/pure parity for waste + transit.

**Risk:** another dense reward term increases PPO cross-talk and
makes it harder to attribute future changes. Start small (0.1
weight, below dest-terminated 0.3) and watch entropy / KL
trajectories. If signals stay clean, ramp; if entropy collapses or
KL thrashes, dial down.

**Why this might matter beyond the dead-end pattern:** the slime
mold analogy generalizes. Self-organizing structures emerge from
*reinforcement on success* + cheap retries, not from foresight.
Right now v2 has penalties for visible failure (waste) and large
rewards for delayed success (kills, captures), but no
fine-grained positive for the *intermediate productive step* —
shipping pressure that propagates. Transit credit is the
missing-middle signal.

Related: [[v2-three-term-reward]] for the current reward stack.



## Structural metrics (Phase 2+)

After Phase 1 reached the "decisive games" milestone but failed to produce
*structured* play, added five end-of-rollout metrics that describe the
shape of territory the policy is producing. Wired into wandb in
`train_v2.py::_structural_metrics`. The dream scenario (per Jason): back
lines = max-strength loops, frontlines = pressure exchange, expansion
priority throughout. Metrics designed to make that legible:

- **`frontier_ratio_end`** — per-seat fraction of owned cells adjacent
  to a non-friendly cell. Pure-loop policy ≈ 0 (everything interior),
  spread-out aggressor ≈ 1. Healthy "established territory" expected to
  sit ~0.3–0.5.
- **`interior_max_frac_end`** — per-seat fraction of *non-frontier* owned
  cells at MAX strength. The dream scenario has this ≈ 1.0: back-lines
  pumping max-strength reserves. Low = back-lines are still being filled.
- **`enemy_pressure_per_frontier_end`** — average `edge_pressure` carried
  across non-friendly outflows on frontier cells. The "frontline pressure"
  signal. Higher = more active engagement.
- **`expansion_rate`** — per-AI-tick rate of positive cell-count delta
  per seat. Captures both neutral and enemy captures.
- **`neutral_capture_rate`** — per-AI-tick rate of total neutral-cell
  consumption (any seat). The user flagged neutrals being abandoned in
  v2-attack-reward; this makes that visible.

Together these turn "is the policy doing the right thing structurally?"
from a visual-replay question into a wandb-graph question.

## Run-archival convention

When a run finishes (or is killed) and is worth preserving, add:
- **Final state:** iter count, last R, ev, entropy, action dist.
- **Checkpoint:** path under `python/checkpoints/v2/` if kept.
- **What changed in the codebase as a result:** decisions promoted,
  thresholds tuned.
- **Failure modes observed:** what to *not* do next time.

If nothing memorable came out of a run, leave it un-archived. Wiki
should compound on signal, not noise.
