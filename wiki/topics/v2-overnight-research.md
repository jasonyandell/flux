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

