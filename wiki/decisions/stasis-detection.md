---
title: Stasis detection
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Decision

The browser declares **stasis** when per-player cell counts stop moving for a window of recent samples, and shows a `STASIS` banner alongside the existing winner banner. Gates `step` and AI advancement the same way `winner` does. Detector is pure and lives outside `src/game/`.

## Where

- `src/sim/stasis.ts` — `detectStasis(samples, epsilon, windowSize) → boolean`. Pure. No DOM, no `three`, no globals. Sits in `src/sim/` so it can stay pure without being part of the game core (which is reserved for the `step`/`applyAction` contract — see [[pure-step-function]]).
- `src/main.ts` — owns the ring buffer and tick counter. Calls `detectStasis` after each sample push.
- `src/render/gameui.ts:showStasisBanner` — sibling to `showBanner`; reuses the same banner element and `onPlayAgain` handler.

## Method

Ring buffer of the last `STASIS_WINDOW` samples; each sample is a `number[]` of per-player cell counts. Three early-return suppressions before the variance check:

1. **Buffer not yet full** — still accumulating samples.
2. **1v1 endgame** — only one or two players have any cells. The 1v1 finishing moves matter and `getWinner` will fire when the last opponent dies; no need to short-circuit.
3. **Cleanup phase** — the runner-up player holds fewer than 5 cells. The leader has genuinely earned dominance and is just mopping up scraps. Absolute count, not a percentage (a percentage-based threshold was rejected as too lenient — early-game dominance must still be earned).

After the suppressions, declare stasis if `max(per-player variance) < STASIS_EPSILON` across the full window.

## Defaults

In `src/main.ts`:

- `STASIS_SAMPLE_PERIOD_TICKS = 5` — sample every 5 game ticks.
- `STASIS_WINDOW = 50` — 50 samples covers 250 sim ticks (~5 seconds at `SPEED = 5`).
- `STASIS_EPSILON = 1.0` cell². Perfectly frozen counts give variance 0. Any per-player count swinging ±1 cell repeatedly across the window gives variance near 1.0, which we still treat as movement; the front-line oscillations from [[../questions/open|open]] settle into amplitudes well above ±1 over hundreds of ticks, so the threshold separates "stuck" from "twitching".

## Reset

`respawn()` clears `winner`, `stasis`, the ring buffer, and `lastStasisSampleTick`. The `play again` button, the lil-gui `respawn` action, and the `numPlayers` change handler all route through `respawn()`.

## Limits

This is a simple statistical test, not a proof. It fires on long oscillation-free plateaus and misses rare slow drifts. False-positive cases observed and fixed:

- *Cleanup-phase plateaus* (one player owns everything, a handful of stragglers stable for ages) — fixed by the runner-up-cell threshold suppression.
- *1v1 endgame oscillations* (two players trading micro-territory in tricky moves) — fixed by the alive-count suppression.

Tune `STASIS_EPSILON` or the window if new false positives appear.

## Why

The default mode is [[multi-player-free-for-all]] at 5× speed, and four of the six AIs in the [[ai-zoo]] stalemate against each other (see [[../questions/open|open]]). A spectator session that wedges into an oscillation should declare itself done rather than spin forever.
