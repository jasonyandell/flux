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

