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

