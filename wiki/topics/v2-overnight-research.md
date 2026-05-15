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

