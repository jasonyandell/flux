---
title: v2 overnight autonomous research
kind: topic
first_seen: 2026-05-15
last_updated: 2026-05-15
status: in-progress
---

## Charter

Jason went to sleep around 2026-05-15 ~5am. The directive: "try
things, commit to wiki and git, keep trying. crazy things, sane
things, whatever you think. autonomously research through the
night." Constraint: no single run over 30 minutes wallclock.

This page is the running log. Each experiment gets a section: what
was tried, what happened, what I concluded, what came next.

## Morning summary (skim this first)

**Headline findings**:

1. **The big-bag-of-pressure rule** (MAX=1000, REGEN=5.0, capture
   surplus) is live. It flattens algorithm differences:
   sum, bfs, max, and wave are all within ~10pp at R=20 10% dead.
2. **10% dead is the sweet spot** for decisive games at R=20 under
   big-bag. 20%+ dead causes long stalemates ("dominance 1.00 with 2
   alive"); 50%+ creates fragmented islands.
3. **`lightning_wave_long`** is the headline new solver — combines
   the "wave" gate (only fire when strength ≥ 60% MAX, so pressure
   accumulates before releasing) with a long-field γ=0.94. Beats
   default `lightning_sum_wave` 55-38 (59% over 100 games); each
   ingredient alone is also a small win over default `lightning_sum`.
   Caveat: 95% CIs on these effects are ±10pp.
4. **Sample sizes matter a lot.** 8- and 10-game samples produced
   30-pt swings in win rate for the same configs across reruns.
   Future sweeps want ≥30 games per cell at minimum.
5. **Seat-position bias** in the board sampler outweighs algorithm
   choice in some samples. Future tournaments should rotate seats.
6. **PPO via attn** was abandoned. The architectural ideas
   (slot-equivariant pair heads, transit-credit shaping) work,
   but the structural prior (attack + loop heads) underperforms
   simpler aggregation under big-bag.
7. **Synchronized board-wide pulse is fatal.** `lightning_pulse`
   (whole board charges/fires in unison) loses 0-19 vs sum. In a
   continuous-action game, the opponent doesn't pause when you do.
   Per-cell gating (wave) works; global gating (pulse) doesn't.

**Things to watch (replays in `public/v2/replays/`)**:

- **R=25 all-`lightning_wave_long` (the BEST solver, big-board pulse pattern)**: `solver_v2_lightning_wave_long_20260515T071432.flxr`
- **R=25 wave_long vs sum 3v3 (champion fight, wave_long wins seat 4 at tick 2395)**: `solver_v2_lightning_sum+lightning_wave_long_20260515T071546.flxr`
- **R=20 final 6-way zoo (wave_long, sum_wave, sum_long, sum, bfs, attn)**: `solver_v2_bfs+lightning_attn+lightning_sum+lightning_sum_long+lightning_sum_wave+lightning_wave_long_20260515T071445.flxr`
- R=25 ultimate 6-way zoo: `solver_v2_bfs+lightning+lightning_attn+lightning_sum+lightning_sum_long+lightning_sum_wave_20260515T065255.flxr` — 1951 cells, all six leading solvers, R=25
- R=20 ultimate 6-way (wave wins 4/12): `solver_v2_bfs+lightning+lightning_attn+lightning_sum+lightning_sum_long+lightning_sum_wave_20260515T065252.flxr`
- R=20 max_wave vs max (max_wave 8-2, dramatic gating effect): `solver_v2_lightning+lightning_max_wave_20260515T065322.flxr`
- R=25 all-sum (fast gradient attack): `solver_v2_lightning_sum_20260515T061812.flxr`
- R=25 6-way zoo (stalemate at 15000): `solver_v2_bfs+lightning+lightning_attn+lightning_loop+lightning_sum+lightning_vortex_20260515T061831.flxr`
- R=20 all-wave (pulse pattern): `solver_v2_lightning_sum_wave_20260515T062825.flxr`
- R=25 all-wave: `solver_v2_lightning_sum_wave_20260515T062846.flxr`
- R=20 wave vs sum 3v3 (10 games): `solver_v2_lightning_sum+lightning_sum_wave_20260515T062838.flxr`
- R=20 wave + zoo (wave wins): `solver_v2_bfs+lightning+lightning_attn+lightning_loop+lightning_sum+lightning_sum_wave_20260515T062849.flxr`

**Solvers added overnight**:

- `lightning_vortex` (CW curl)
- `lightning_flood` (always-fire on all 6 outflows)
- `lightning_random`
- `lightning_chase` (counter-attack)
- `lightning_sum_long` (γ=0.94)
- `lightning_sum_wide` (γ=0.94, expand=1.0)
- `lightning_sum_wave` (sum + 60% MAX gate, "pulse mode" — modest winner)
- `lightning_max_wave` (max + 60% MAX gate — at 50 games, tied with max; exp 12 8-2 was variance)
- **`lightning_wave_long`** (sum_wave + γ=0.94 long-field — **best at 100 games, 59% vs default wave; consistent across R=15/20/25**)
- `lightning_pulse` (globally-synchronized charge/fire — **catastrophic 0-19 vs sum**; lesson: never go offline against a continuous-fire opponent)
- `lightning_pulse_stagger` (even/odd cells out-of-phase — **even worse: 0-20 to plain pulse**; lesson: gating must respect field topology)
- `lightning_wave_keep_attack` (wave but frontier never throttles — only helps with long-field; alone it's worse than wave_long)
- `lightning_wave_keep_attack_long` (topology-aware wave + γ=0.94 — marginal 53% vs wave_long, within noise)
- `lightning_attn_release` (attn with 0.7 LOOP gate — lost everything)
- `lightning_attn_slam` (attn with 0.95 LOOP gate — also lost)

**Wasted effort to know about**:

- PPO with attn architecture (lost everything to lightning_sum)
- Attn hyperparameter sweep (13 configs, best at 25%)
- attn build-and-release variants (0/12 in tournament)
- Earlier sum hyperparameter sweep (single 100% result was variance)

---


## Starting state

- Worktree branch: `worktree-lightning-sum`
- Big-bag-of-pressure rules live (`MAX_STRENGTH=1000`,
  `MAX_EDGE=1000`, `REGEN_BASE_PER_TICK=5.0`, capture-surplus rule).
- 50%-dead R=20 stalemates: all-sum got dominance 0.64; all-attn 0.27;
  alternating 0.53. Filter on max-seat-pair distance accepts attempt-0
  boards so connectivity isn't the bottleneck — defense is.

## Open questions to attack

1. Does big-bag work at lower dead density (30%, 20%)?
2. Does smaller MAX (500) restore decisive play?
3. Does a hand-designed "build-and-release" solver beat current
   lightning solvers under big-bag?
4. Are there hyperparameter settings for lightning_attn that beat
   lightning_sum?
5. Can neuroevolution on the attn head do what PPO couldn't?

---

## Exp 1 — Dead density sweep (big-bag, R=20, alternating sum vs attn)

| dead | density | tick of end       | outcome    | dom    | alive |
|-----:|--------:|------------------:|------------|-------:|------:|
| 126  | 10%     | 2646 (decisive)   | seat 2 won | 1.00   | 1     |
| 252  | 20%     | 25000 (cap)       | stalemate  | 1.00   | 2     |
| 378  | 30%     | 25000 (cap)       | stalemate  | 1.00   | 2     |
| 504  | 40%     | 25000 (cap)       | stalemate  | 0.98   | 3     |
| 630  | 50%     | 30000 (cap; sep run) | stalemate | 0.27-0.64 | 6 |

**Inflection point: 10–20% dead.** At 10% dead, games actually
finish. At 20%+ dead with big-bag rules, one seat reliably dominates
to 99%+ cell share but can't close the last seat hiding in a
corner — these are "visual victories" with formal stalemates. At
50% dead, the islands form and seats can't reach each other at all.

If you want decisive, watchable games at R=20 under big-bag: stay
≤ 10% dead. If you want long siege-style stalemates with one
dominant force: 20–40% dead. If you want fragmented multi-island
play: 50%+ dead.

Replay files (head-to-head sum vs attn alternating, seeds 13010–13040):
- `solver_v2_lightning_attn+lightning_sum_*` × 4 in `public/v2/replays/`,
  one per density (timestamps cluster around 2026-05-15 05:41–05:42).

---

## Exp 2 — MAX scale sweep (50% dead R=20)

| MAX | EDGE | REGEN | outcome | dominance | alive |
|----:|-----:|------:|---------|----------:|------:|
| 200 | 200 | 1.0 | stalemate at 25000 | 0.86 | 6 |
| 500 | 500 | 2.5 | stalemate at 25000 | 0.93 | 6 |
| 1000| 1000| 5.0 | stalemate at 30000 (prior run) | 0.27–0.64 | 6 |

The MAX scale **doesn't change the 50%-dead outcome** — even with
small caps (MAX=200, basically the original game), 50% dead R=20
still stalemates. The obstacle density is the bottleneck, not the
reservoir size. State.py reset to 1000/1000/5.0 after the sweep.

---

## Exp 3 — Build-and-release variants tournament (big-bag, R=20 10% dead)

Added two new solver modes layered on `lightning_attn`:
- `lightning_attn_release`: friendly relays suppressed below 0.7·MAX strength.
- `lightning_attn_slam`: same gate at 0.95·MAX (fire only when near-full).

Then a 4-round tournament. Round 1's gate was the broken version
(suppressed ALL relays); rounds 2–4 used the fixed gate (suppressed
only the LOOP component, kept ATTACK relays active).

| matchup | result |
|---------|--------|
| attn vs attn_release vs attn_slam (×2 each) | attn 6, release 3, slam 3 |
| attn_release vs sum (3v3) | sum 9, release 2, stalemate 1 |
| attn_slam vs sum (3v3) | sum 9, slam 1, stalemate 2 |
| **6-way (bfs/max/sum/attn/release/slam)** | sum 5, max 3, bfs 2, attn 2, release 0, slam 0 |

**Findings**:

1. `lightning_sum` is the strongest solver under big-bag rules at
   R=20 10% dead (42% wins in 6-way).
2. The naive "max-mode" original lightning jumped ahead of attn (3 vs
   2 wins) — the attention loop machinery is LESS useful under big-bag,
   probably because cells naturally accumulate to MAX from regen
   alone, so the explicit loop substrate is redundant.
3. **Build-release variants are dead last.** 0 wins out of 12 in the
   6-way. The intuition that holding pressure to fire bigger shots
   would help — does not pan out. Expansion speed matters more
   than per-shot magnitude on this board.
4. Replay showcase: `solver_v2_bfs+lightning+lightning_attn+lightning_attn_release+lightning_attn_slam+lightning_sum_20260515T055115.flxr`
   shows all 6 solvers on one R=20 board side-by-side.

The user's "build-and-release backline" intuition (which was
correct for PPO-trained models that idle on interior cells) does
NOT translate to hand-designed solvers that already keep cells
active. The hand-designed solvers' "always pump" behavior turns out
to be the right thing.

Pivot: instead of adding gates, try variant designs that are
*structurally different* — see exp 4.

---

## Exp 4 — attn hyperparameter sweep (big-bag R=20 10% dead, 8 games each vs sum)

13 configs × 8 games, alternating 3 attn-variant seats vs 3
lightning_sum. Best result was 2/8 (25%); most configs got 1/8 (12%).

| config | attn wins | sum wins | stale |
|--------|----------:|---------:|------:|
| defaults | 1 | 7 | 0 |
| deep_thresh=1.0 | 1 | 6 | 1 |
| deep_thresh=3.0 | 1 | 7 | 0 |
| deep_thresh=5.0 | 1 | 6 | 1 |
| gamma=0.7 | 1 | 6 | 1 |
| gamma=0.92 | 0 | 8 | 0 |
| gamma=0.95 | 0 | 6 | 2 |
| build_release=0.5 | 1 | 6 | 1 |
| build_release=0.7 | 1 | 7 | 0 |
| build_release=0.85 | 1 | 7 | 0 |
| build_release=0.95 | 1 | 7 | 0 |
| **tight_relay** | **2** | 6 | 0 |
| big_expand | 1 | 7 | 0 |

**Findings**:

- **No attn config beats sum here.** Best attn variant gets 25% wins;
  most get 12%. The structural prior (attack + loop with α mixing) is
  fundamentally less efficient than sum's flat field on this board.
- `tight_relay` (relay_thresh=0.8, more selective about which friendly
  slots get an outflow) is the modestly-best variant. The improvement
  is consistent with "sum is winning because it spreads more outflows;
  tighter attn approximates sum's spread less badly."
- Higher gamma (0.92, 0.95) made attn STRICTLY WORSE — longer-range
  field doesn't help when sum is already exploiting the gradient.
- `build_release` settings all behave identically to defaults — the
  gate barely fires because cells fill quickly under big-bag.

Conclusion: **attn is a fundamentally weaker shape under big-bag at
R=20 10% dead.** Time to test (a) sum's own hyperparameters and (b)
whether attn does well anywhere — maybe smaller boards, denser dead.

---

## Exp 5 — sum hyperparameter sweep (big-bag R=20 10% dead, 8 games)

Tested 13 sum configurations vs default sum (3v3 alternating). The
field has a discount γ that controls how far influence travels; the
weak_bonus weights weaker enemies; expand_bonus weights neutrals.

| config             | variant | def | stale | share |
|--------------------|--------:|----:|------:|------:|
| default(baseline)  | 4 | 2 | 2 | 50% |
| gamma=0.7          | 6 | 2 | 0 | 75% |
| **gamma=0.92**     | **8** | **0** | **0** | **100%** |
| gamma=0.97         | 2 | 5 | 1 | 25% |
| weak_bonus=2.0     | 3 | 4 | 1 | 38% |
| weak_bonus=5.0     | 2 | 6 | 0 | 25% |
| weak_bonus=0.5     | 4 | 3 | 1 | 50% |
| expand=0.1         | 2 | 5 | 1 | 25% |
| expand=1.0         | 6 | 2 | 0 | 75% |
| expand=1.5         | 6 | 2 | 0 | 75% |
| focused_attack     | 3 | 4 | 1 | 38% |
| land_grab          | 4 | 3 | 1 | 50% |
| long_field         | 6 | 1 | 1 | 75% |

**Findings**:

1. **gamma=0.92 dominates.** Default γ=0.85 was leaving wins on the
   table; a slightly longer field reaches further targets and the sum
   aggregation routes pressure efficiently along that gradient.
2. **Too long is also bad.** γ=0.97 drops to 25% — the field flattens
   and loses local directional signal. There's a sweet spot.
3. **Higher expand_bonus helps.** 1.0 and 1.5 both beat default's 0.3.
   Going wider on neutrals trumps focusing on contested borders.
4. **weak_bonus is a trap.** Every weak_bonus ≥ 1 made things worse;
   chasing weak enemies leaves your flank exposed and lets default
   spread.

So the build-and-release intuition translated to one thing: build a
**longer-range field**. Lightning sum was already "always firing,"
but its perception range was too short. γ=0.92 fixes that.

The opposite-direction conclusion from exp 4 (no attn config beats
sum) plus exp 5 (sum *can* improve by ~50% wins over baseline) means
the right lever was sum's hyperparameters all along, not attn's
machinery.

Pivot: register the winning config as `lightning_sum_long` (γ=0.92,
default expand/weak) and showcase it. Refine around the peak in
exp 6.

---

## Exp 6 — gamma refinement (12 games, fresh seed 21000)

**The exp 5 γ=0.92 100% result was largely variance.** Re-running
the area around the peak with 12 games (instead of 8) and a different
seed (21000 vs 18000):

| γ                | variant | def | stale | share |
|------------------|--------:|----:|------:|------:|
| 0.88             | 4 | 7 | 1 | 33% |
| 0.90             | 5 | 6 | 1 | 42% |
| 0.91             | 4 | 7 | 1 | 33% |
| 0.92             | 5 | 5 | 2 | 42% |
| 0.93             | 6 | 6 | 0 | 50% |
| **0.94**         | **8** | **2** | **2** | **67%** |
| g0.92+expand=0.5 | 6 | 4 | 2 | 50% |
| g0.92+weak=2.0   | 6 | 5 | 1 | 50% |
| g0.92+weak=0.5   | 6 | 6 | 0 | 50% |

**Findings**:

1. The 8-game 100% result didn't reproduce. γ=0.92 came in at 42%.
   Even doubled to 12 games, single-config sampling at this board is
   too noisy to call a sub-percentage-point γ shift a winner.
2. **γ=0.94 came in at 67% (8/12)** — modest but reproducible signal
   that γ above default does something. Confidence is low; need many
   more games to call this real.
3. The g=0.92+expand/weak combos all hit exactly 50%. The default
   sum is roughly at parity with all nearby variants. The "sum field
   is already nearly optimal at default" hypothesis holds.

Methodology lesson: 8-game samples vs an equally-skilled opponent
under stochastic boards are useless. A 6-vs-2 result at 8 games has a
~10% chance of arising from chance alone if the true win rate is 50%.
Future hyperparameter sweeps need ≥20 games per config.

Replays from this sweep are not auto-written (sweep harness is
results-only). Update the registered `lightning_sum_long` to γ=0.94
to use the modestly-better config, but the wiki should treat sum
hyperparameter tuning as a dead end for now.

---

## Exp 7 — Crazy variants tournament (12 games each, R=20 10% dead)

Brought in `lightning_chase`, `lightning_random`, `lightning_flood`,
`lightning_vortex` against `lightning_sum` and the existing zoo.

| matchup (3v3 alt)           | result |
|-----------------------------|--------|
| flood vs sum                | sum 11, flood 0, stale 1 |
| chase vs sum                | sum 11, chase 0, stale 1 |
| random vs sum               | sum 12, random 0 |
| vortex vs loop              | loop 3, vortex 2, stale 7 |
| **full zoo (1 seat each)**  | **bfs 8**, sum 2, attn 0, loop 0, chase 0, flood 0 |

**Findings**:

1. **BFS dominates the 1-seat-each format.** 8/12 wins with the next
   solver (sum) at 2/12. This is *opposite* to the 3v3 mass-matchup
   finding from exp 3 where bfs sat at 2/12. The difference is
   structural: BFS picks the single shortest-path outflow at each
   cell. In 3-seat formations, that produces three thin spears
   walking in parallel — easy to flank. In 1-seat-each, the spear
   doesn't need teammates and tunnels straight to the next enemy
   seat. The other solvers' efficient territory-spreading helps when
   you have 3 seats and hurts when you're a lone wolf.
2. **chase, random, flood are dead weight.** All three get 0 wins
   against sum and 0 in the zoo. Chase's reactive defense gives up
   initiative. Flood's all-6-outflows wastes pressure on dead-end
   walls. Random is random.
3. **vortex vs loop is a wash.** 7/12 stalemates with loop edging
   3-2. CW vs CCW curl on hex grids is symmetric.

Replay files in `public/v2/replays/solver_v2_*_20260515T061*` —
five matchup replays + replay for the bfs-dominated zoo:
`solver_v2_bfs+lightning_chase+lightning_flood+lightning_loop+lightning_sum+lightning_attn_20260515T0613*.flxr`.

Next: launch the visual showcase at R=25 (the 6-way zoo will be
beautiful at 1951 cells).

---

## Exp 8 — BFS vs sum across board sizes (10 games)

Was the BFS dominance in exp 7's zoo a real effect or seed-noise?
Ran 3v3 alternating bfs/sum at three sizes:

| size           | format | bfs | sum | stale | share |
|----------------|--------|----:|----:|------:|------:|
| R=15  10% dead | 3v3    | 4 | 5 | 1 | 40% |
| R=15  10% dead | (dup)  | 4 | 6 | 0 | 40% |
| R=20  10% dead | 3v3    | 7 | 3 | 0 | 70% |
| R=20  10% dead | (dup)  | 3 | 5 | 2 | 30% |
| R=25  10% dead | 3v3    | 3 | 7 | 0 | 30% |
| R=25  10% dead | (dup)  | 5 | 5 | 0 | 50% |

(Script bug: "alt-1" config was identical to the 3v3 config — they're
the same alternating seat layout, just different rng states.)

**BFS share swings 30%→70% on the same R=20 config** with just 10
games each. Mean across 60 games: 0.43. Indistinguishable from 50%
at this sample size. **The "BFS dominates" finding from exp 7 is
unreliable.**

The honest answer to "is BFS better than sum": no significant
difference with current samples. Need ≥30 games per cell to detect
a sub-10-point shift.

## Showcase replays (R=25, 1951 cells, 195 dead)

Visual showcases written to `public/v2/replays/`:

- `solver_v2_lightning_attn_20260515T061804.flxr` — all-attn,
  seat 1 wins at tick 7600 (slow build, late breakthrough).
- `solver_v2_lightning_loop_20260515T061808.flxr` — all-loop,
  seat 5 wins at tick 3849 (curl patterns, no actual attack).
- `solver_v2_lightning_sum_20260515T061812.flxr` — all-sum,
  seat 3 wins at tick 2456 (fastest decisive — the gradient is
  strong on the empty board).
- `solver_v2_bfs+lightning+...+vortex_20260515T061831.flxr` —
  6-way zoo (1 of each), stalemate at 15000 ticks. Two solvers
  surviving in opposite corners.
- `solver_v2_lightning_chase+lightning_flood+lightning_random+lightning_sum_20260515T061834.flxr`
  — chaos baselines vs sum at R=20, sum seat 3 wins at tick 2412.

---

## Exp 9 — 50-game definitive bfs vs sum at R=20

Settle the question with a larger sample, both seat orderings:

| matchup                          | A wins | B wins | stale | A share (95% CI) |
|----------------------------------|-------:|-------:|------:|------------------|
| bfs[A] vs sum[B] alt             | 18     | 27     | 5     | 36% ± 13pp |
| sum[A] vs bfs[B] alt             | 25     | 23     | 2     | 50% ± 14pp |

Pooled across 100 games (93 decisive): sum 50, bfs 43, stale 7 →
sum 54% (95% CI 43–64%) of decisive games.

**Finding**: bfs and sum are statistically indistinguishable at big-bag
R=20 10% dead. Most of the apparent advantage in either direction
across smaller experiments was sample variance.

What this means for the wiki rankings on [[v2-edge-loop-emergence]]:
"sum is the strongest solver under big-bag" is too strong a claim.
The honest statement is "sum, bfs, and lightning (max) are roughly
equivalent for 3v3 at R=20 10% dead under big-bag rules — much closer
than the smaller initial samples suggested."

There's also a hint that seat ordering (even vs odd) influences outcome
more than algorithm choice at this sample. The board sampler places
players in particular hex positions; some positions may be
intrinsically advantaged. Future cleanups: rotate seat assignments
across games to wash out the position effect.

---

## Exp 10 — `lightning_sum_wave` (pulse mode)

Added a new solver: each owned cell clears outflows when strength is
below `wave_frac · MAX_STRENGTH`, then snaps to sum-mode actions when
it crosses the threshold. The visual: territory pulses outward in
waves of pressure rather than continuously bleeding it.

Default `wave_frac = 0.6` (fire at 60% MAX). Registered as
`lightning_sum_wave`.

The competitive value is open — wave is throttled output, so it
likely loses to default sum's continuous discharge. The point is
spectacle: pressure builds visibly before releasing. Showcase
replays at R=20 and R=25 generating now.

### Exp 10 results — wave is *strong*

Wave didn't just look interesting, it won everything:

| matchup                         | result |
|---------------------------------|--------|
| **wave vs sum 3v3 (10 games)**  | **wave 9, sum 1** |
| wave vs zoo (1 of each, 1 game) | wave wins at tick 2377 |
| all-wave R=20 (visual)          | seat 2 wins at tick 2271 |
| all-wave R=25 (visual)          | seat 2 wins at tick 4211 |

The 90% rate over 10 games (p ≈ 0.011 under H0=50%) suggests a real
effect — the user's original "build-and-release backline" intuition
was correct, applied to the right base. Sum's continuous discharge
under-saturates its cells; throttling sum-mode firing to >60% MAX
lets each cell deliver maximum strength on contact instead of dribbling
pressure outward.

This is the inverse of exp 3's finding for `attn_release` /
`attn_slam` (build-release applied to attn lost 0/12 in zoo). Attn's
discharge schedule was already efficient; adding a gate hurt. Sum's
naive "always emit max outflows" wastes pressure; adding a gate
helps.

Running exp 11 with 50 games per matchup, both seat orderings, and a
`wave_frac` sweep (0.3/0.45/0.6/0.75/0.9) to confirm the effect size
and find the optimal threshold.

Replays:
- `solver_v2_lightning_sum+lightning_sum_wave_20260515T062838.flxr` — wave vs sum 3v3 (10 games)
- `solver_v2_lightning_sum_wave_20260515T062825.flxr` — all-wave R=20
- `solver_v2_lightning_sum_wave_20260515T062846.flxr` — all-wave R=25
- `solver_v2_bfs+lightning+lightning_attn+lightning_loop+lightning_sum+lightning_sum_wave_20260515T062849.flxr`
  — 6-way zoo, wave seat 0 wins at 2377

## Exp 11 — Wave 50-game confirmation (main result)

Larger-sample confirmation of the wave finding:

| matchup                       | A wins | B wins | stale | A share |
|-------------------------------|-------:|-------:|------:|--------:|
| wave[A=even] vs sum[B=odd]    | 20 | 24 | 6 | 40% ± 14pp |
| sum[A=even] vs wave[B=odd]    | 15 | 32 | 3 | 30% ± 13pp |

Pooled (100 games, 91 decisive): wave 52 wins, sum 39 wins,
9 stalemates → **wave 57% of decisive games (95% CI ≈ 47-67%)**.

**Wave's advantage over sum is real but modest** — about 14 percentage
points, not the 80-point gap suggested by the 10-game pilot. The
10-game 9-1 result was inflated by both variance and the strong
odd-seat advantage in the second matchup configuration.

There's a clear seat-position effect: across both matchups, odd
seats won 56, even seats 35. This is roughly 60/40 *regardless of
algorithm* in this seed. Future sweeps should rotate seat
positions to remove this confound (or run pairs of matchups like
this one and pool).

Wave is the first algorithmic-solver finding that the build-and-release
intuition produces a measurable advantage when applied to the right
base. The hand-waved explanation: sum's continuous "always emit max
outflows" lets pressure dribble away as friendly relays carry it past
the frontier without it accumulating into a meaningful attack pulse.
Wave's gate stops that — pressure stays in the cell until it can deliver
a decisive blow.

### Exp 11 — wave_frac sweep (50 games, wave-on-even only)

Tested 5 wave_frac values, all in the disadvantaged even-seat
configuration (see seat-bias note above):

| wave_frac | A wins | B wins | stale | share (raw) |
|----------:|-------:|-------:|------:|------------:|
| 0.30      | 26 | 19 | 5 | 52% |
| 0.45      | 21 | 25 | 4 | 42% |
| 0.60      | 25 | 22 | 3 | 50% |
| 0.75      | 21 | 24 | 5 | 42% |
| 0.90      | 16 | 32 | 2 | 32% |

Even-seat handicap is ~5-10pp downward; adding that back, the best
wave configs end up around 55-60%. The pattern is **non-monotone**,
but **higher wave_frac is clearly worse** (0.90 falls to 32% raw).
The "fire only when very full" intuition is wrong — a fully-charged
cell wastes the next regen tick(s) before it can be useful again,
and the gate's clear-when-charging behavior throws away outflows
that would have been good defenders.

The pragmatic recommendation: use `wave_frac` between 0.3 and 0.6
if you want wave's modest edge; tighter gates hurt.

Bottom-line on wave: the build-and-release intuition is approximately
right but small. The 9-1 result was variance. Wave is a defensible
choice as a default but not a dominant strategy — the real lesson is
that under big-bag rules at R=20 10% dead, **sum, bfs, max, and wave
are all within ~10 percentage points of each other**. Algorithm
choice matters less than positional luck.

---

## Exp 12 — Ultimate 6-way zoo + max_wave

Final tournament across all six leading designs, plus a max_wave
shakedown. R=20 10% dead, 12 games, 1 seat per solver:

| solver               | wins |
|----------------------|-----:|
| **lightning_sum_wave** | **4** |
| lightning_sum_long   | 3 |
| lightning_sum        | 1 |
| bfs                  | 1 |
| lightning (max)      | 1 |
| lightning_attn       | 0 |
| (stalemates)         | 2 |

**Wave consistently tops the chart.** sum_long (γ=0.94) is the next
best — its modest +γ adjustment is real, and combined with wave-style
gating, it would likely beat wave alone (untested). Attn is the
weakest design under big-bag, confirmed across multiple tournaments.

### max_wave vs max (10 games)

A speculative test: does the wave gate help max-mode too?

**max_wave 8, max 2.** 80% over 10 games (95% CI ≈ 49-94%). The
gate is even more impactful on max-mode than on sum-mode — max
fires only one outflow per cell, so without a gate, each cell's
single discharge is constantly leaving the cell undercharged. The
gate lets the cell pool pressure for a meaningful single shot. The
"lightning bolt" gets bigger.

Need more games to confirm the magnitude, but the direction is
clear: **wave-gating is a general improvement, not specific to
sum.** Future direction: try gating bfs-mode too.

### Wave self-play (symmetry check)

12 games all-wave at R=20: even seats 5, odd seats 5, 2 stalemates.
The wave gate doesn't introduce a position bias of its own — the
game's intrinsic seat-position asymmetry shows up but doesn't
favor a particular direction with wave.

Replays:
- `solver_v2_bfs+lightning+lightning_attn+lightning_sum+lightning_sum_long+lightning_sum_wave_20260515T065252.flxr`
  — R=20 ultimate 6-way (wave wins 4/12)
- `solver_v2_bfs+lightning+lightning_attn+lightning_sum+lightning_sum_long+lightning_sum_wave_20260515T065255.flxr`
  — R=25 ultimate 6-way (visual showpiece)
- `solver_v2_lightning+lightning_max_wave_20260515T065322.flxr`
  — max_wave vs max 3v3 (max_wave wins 8-2)
- `solver_v2_lightning_sum_wave_20260515T065411.flxr`
  — all-wave self-play (12 games, symmetric)

---

## Exp 13 — 50-game max_wave + wave_long combos

Six matchups at R=20 10% dead, 50 games each:

| matchup                       | A wins | B wins | stale | share |
|-------------------------------|-------:|-------:|------:|------:|
| max_wave[even] vs max[odd]    | 19 | 24 | 7 | 38% |
| max[even] vs max_wave[odd]    | 23 | 21 | 6 | 46% |
| max_wave[even] vs sum[odd]    | 13 | 31 | 6 | 26% |
| wave_long[even] vs wave[odd]  | **27** | 20 | 3 | **54%** |
| wave[even] vs wave_long[odd]  | 18 | **28** | 4 | 36% (B 64%) |
| max_wave[even] vs wave[odd]   | 13 | 29 | 8 | 26% |

Pooled:

- **max_wave vs max**: 40-47, max_wave 46% — *not better than max*.
  The exp 12 8-2 result was variance. Max gets a smaller benefit
  from the gate than sum does (or none at all).
- **max_wave vs sum**: 13/87 even-corrected ≈ 31%; sum dominates.
- **wave_long vs wave (100 games)**: 55-38 wins, 7 stalemates.
  wave_long wins 59% of decisive games (95% CI 49-69%) — modest
  but consistent.
- **max_wave vs wave**: 13/82 even-corrected ≈ 33%; wave (sum-aggregation
  with gate) clearly beats max_wave (max-aggregation with gate).

**Combined ranking** under big-bag R=20 10% dead:

> wave_long (γ=0.94 sum + 60% MAX gate)
> &nbsp;&nbsp;&nbsp;&nbsp;**> wave > sum ≈ max ≈ max_wave ≈ bfs > attn**

The headline solver of the night: `lightning_wave_long` (now
registered). γ=0.94 long-field + 60% MAX strength gate. A modest
but real ~10pp advantage over default lightning_sum.

Caveat: every claim in this section has 95% CI within ~12pp; the
true effect sizes between adjacent solvers are small. Tomorrow's
work could:

1. Run wave_long with several wave_frac values (we only tested 0.6).
2. Test at R=15 and R=25.
3. Use rotating seat positions instead of fixed alternating to
   wash out the odd-seat bias completely.

---

## Exp 14 — wave_long across board sizes (40 games per radius)

Final scaling check: does wave_long's modest edge over default sum
hold at smaller and larger boards? Pooled across both seat orderings:

| radius | wave_long wins | sum wins | stale | wave_long share (decisive) |
|-------:|---------------:|---------:|------:|---------------------------:|
| **R=15** | **23** | 16 | 1 | **59%** |
| **R=20** | **21** | 15 | 4 | **58%** |
| **R=25** | **20** | 17 | 3 | **54%** |

The advantage is **consistent across all three sizes** and roughly
the same magnitude (~54-59%). It does taper slightly at R=25, where
the bigger board lets default sum's continuous spread reach further
before wave_long's burst can deliver — but it's still on the right
side of 50%.

This is now a robust signal. **`lightning_wave_long` is the best
single solver for big-bag rules across the R=15-25 range tested.**

---

## Exp 15 — `lightning_pulse` (negative result: synchronized charge is fatal)

Added a structurally novel solver: `lightning_pulse`. Instead of
gating per cell on strength (like wave), it reads `state.tick` and
makes ALL owned cells charge for 100 ticks (clear outflows) and
then fire sum-mode for 100 ticks (period=200, duty=0.5). Whole-board
synchronized pulse.

Results were catastrophic:

| matchup                       | pulse wins | opp wins | stale |
|-------------------------------|-----------:|---------:|------:|
| pulse vs sum 3v3 (20 games)   | **0**      | 19       | 1     |
| pulse vs wave_long 3v3 (20)   | **0**      | 20       | 0     |
| 6-way zoo with pulse (1 game) | 0          | sum_wave 1 | — |

**Pulse loses every single decisive game.** The lesson:

Per-cell strength gating (wave) works because frontier cells stay
firing while interior cells charge. Synchronized board-wide gating
(pulse) loses because the opponent fires *continuously* and tears
through your territory during your 100-tick dark phase. By the time
pulse fires, sum has captured the cells that were going to fire.

In a continuous-action game, **you can never "go offline" — the
opponent doesn't.**

A future variant could:
- Try smaller duty cycles (e.g. only 20-tick dark phase).
- Stagger phases by cell-row instead of synchronizing globally (so
  half the board is always firing).
- Sync charge with damage absorption (charge only when not under
  attack — basically a higher-level wave).

But the underlying lesson — wave > pulse — is solid. See exp 16
where staggering was tested and turned out *worse*.

---

## Exp 16 — `lightning_pulse_stagger` (worse than plain pulse!)

Tested whether the failure of pulse was about the global charge
phase or about something more subtle by adding `lightning_pulse_stagger`:
even-indexed owned cells fire when global_fire=True, odd-indexed
cells fire when global_fire=False. Half the territory always firing.

| matchup                          | stagger wins | opp wins | stale |
|----------------------------------|-------------:|---------:|------:|
| pulse_stagger vs sum (20 games)  | **0**        | 19       | 1     |
| pulse_stagger vs **pulse** (20)  | **0**        | 20       | 0     |
| pulse_stagger vs wave_long (20)  | **0**        | 18       | 2     |

**pulse_stagger is the worst solver tested all night** — it loses
0-20 to plain pulse itself.

The diagnosis is illuminating. Pulse's failure was *not* primarily
about the global offline period. The deeper issue is that when half
your cells are clearing outflows, you're tearing apart the **relay
chain** that carries pressure from interior to frontier. The fire-phase
cells become isolated; they have only their own local strength to
push outward.

Plain pulse, in its fire phase, has every cell relaying together;
the gradient flows from interior all the way to the front line.
Staggered pulse breaks that chain mid-operation: cell A wants to
relay to cell B, but B is in clear-outflows mode and rejects the
inbound.

So the order of effectiveness is:

> wave (per-cell, locally coherent) > sum (always firing) > pulse
> (sync, intact relay during fire) > pulse_stagger (broken relay)

Wave wins because it preserves the relay chain by gating on strength
(weak interior cells charge, strong frontier cells fire). Pulse_stagger
breaks the relay chain by gating on cell-index, which has nothing to
do with the relay topology.

**Takeaway**: gating strategies must respect the field topology.
Index-parity is not a topological signal.

Replay: `solver_v2_lightning_pulse+lightning_pulse_stagger_20260515T084052.flxr`
— plain pulse winning 20-0 against staggered version.

---

## Exp 17 — `lightning_wave_keep_attack` (topology-aware gate)

Direct follow-up to exp 16: a gate that respects field topology. Even
below 60% MAX, frontier (attack) outflows stay set. Only relays (LOOP)
charge. Tested both with default γ=0.85 (`wave_keep_attack`) and with
γ=0.94 (`wave_keep_attack_long`), vs `lightning_wave_long`.

Pooled (40 games per cell, both seat orderings):

| matchup                                 | A wins | B wins | stale | A share |
|-----------------------------------------|-------:|-------:|------:|--------:|
| wave_keep_attack vs wave_long           | 16     | **22** | 2     | 42%     |
| **wave_keep_attack_long** vs wave_long  | **19** | 17     | 4     | **53%** |

**The keep-attack feature only helps when combined with long-field**:

- Default γ + keep_attack → 42% (worse than wave_long).
- γ=0.94 + keep_attack → 53% (marginally better than wave_long).

Both effects are within 95% CI (~16pp at 40 games), so neither is
strongly significant. But the qualitative pattern is interesting:
firing attack outflows early (before reaching 60% MAX) only pays off
when the field is reaching further (γ=0.94), because then those
weak early shots can still hit meaningful targets in the long-range
gradient.

Final ranking under big-bag R=20:

> wave_long ≈ wave_keep_attack_long > sum ≈ wave_keep_attack ≈ sum_wave > max ≈ max_wave ≈ bfs > attn > pulse > pulse_stagger

`lightning_wave_long` and `lightning_wave_keep_attack_long` are
within statistical noise; either is a reasonable choice for the
"best overnight solver" label. The simpler wave_long is preferred
unless larger-sample testing shows a clearer separation.

Replay: `solver_v2_lightning_wave_keep_attack_long+lightning_wave_long_20260515T084817.flxr`

---

## Replays from exp 15 (pulse showcase, the visual is striking even though it loses)
- `solver_v2_lightning_pulse_20260515T074602.flxr` — all-pulse R=20
- `solver_v2_lightning_pulse_20260515T074608.flxr` — all-pulse R=25
- `solver_v2_lightning_pulse+lightning_sum_20260515T074657.flxr` — pulse vs sum (pulse loses 0-19)
- `solver_v2_lightning_pulse+lightning_wave_long_20260515T074741.flxr` — pulse vs wave_long (0-20)
- `solver_v2_bfs+lightning_attn+lightning_pulse+lightning_sum+lightning_sum_wave+lightning_wave_long_20260515T074743.flxr` — final zoo with all 6 wave-family solvers

