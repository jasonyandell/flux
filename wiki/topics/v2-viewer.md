---
title: v2 viewer (trainer-displayer)
kind: topic
first_seen: 2026-05-14
last_updated: 2026-05-14
status: active
---

## What this page is

The single feature catalog for the v2 trainer-displayer at
`/index-v2.html`. The *why* of a separate UI entrypoint lives in
[[v2-trainer-displayer]]; this page collects what the viewer **does**
for someone watching v2 PPO replays land.

The viewer plays back `.flxr` v2 replays from `public/v2/replays/`. It
never runs the simulation in-browser — the trainer is the source of
truth, the viewer is the lens. The list below stays accountable to
`src_v2/` as it ships.

## Why anyone reaches for it

- Live training monitor — the trainer drops a `.flxr` plus an
  `index.json` entry each `--record-stride` iters; the viewer polls and
  auto-swaps to the newest replay on a 3 s cadence (`POLL_INTERVAL_MS`).
- Post-hoc inspection — by default the index keeps the last **50** runs
  (`append_index(cap=50)` in [`python/flux_v2/replay.py`](../../python/flux_v2/replay.py)),
  so you can scroll back through whatever's still on disk.
- Mixed-experiment streams — radius/seat changes are normal between
  runs; the viewer rebuilds geometry on board-signature changes and
  doesn't need a reload.

## Recent-runs list (most recent first)

The top bar carries a horizontally scrollable strip of every indexed
replay, newest at the left. Each entry is a clickable `i<iteration>`
chip; the currently-playing one is outlined. Clicking jumps to that
replay and unpauses.

- **Order:** index is sorted `saved_at` descending on every write
  (`python/flux_v2/replay.py::append_index`), so on reload the freshest
  run is what plays first and the list reads recent → less recent left
  to right.
- **Depth:** all 50 indexed entries are reachable from the strip. There
  is no separate "older runs" menu — horizontal scroll is the affordance.
- **Hover tooltip** shows the full `.flxr` filename, relative
  `saved_at`, radius, seat count, and `kind` if recorded.
- **Highlighting:** the active replay is bordered `#4a90e2`; others sit
  at 0.78 opacity. The whole strip dims to 0.6 when the pointer leaves.
- **Pause survival:** explicit selection unpauses and bypasses the
  auto-cycle gate (`forceLoad` in
  [`src_v2/replay/player.ts`](../../src_v2/replay/player.ts)). Auto-poll
  preemption still respects paused state for genuinely-new arrivals
  (only newest fresh drops can preempt; static newest entries cannot).

## Transport bar (bottom)

```
[⏮]  [⏪]  [⏯]  [⏩]  [⏭]   ━━━●━━━━━   12 / 240   1×
 prev  step  play  step  next   scrubber    counter   speed
```

- **Prev / Next replay** walk the index round-robin
  (`Shift+←` / `Shift+→`). Skipped (unplayable) entries are filtered
  out, same set the auto-cycle uses.
- **Step back / forward one frame** (`←` / `→`) auto-pause and jog ±1.
- **Play / Pause** (`Space`) freezes the current frame. While paused,
  the index poller still discovers new replays but does not preempt;
  auto-cycle fires only at end-of-replay AND `!paused`. Manual prev/next
  still works (`forceLoad` bypasses the gate).
- **Scrubber** is a `<input type="range">` over the frame index
  (0..N−1). Dragging auto-pauses; programmatic updates are suppressed
  while the user has the slider grabbed so playback doesn't fight the
  drag.
- **Speed cycle** picks `0.25 / 0.5 / 1 / 2 / 4×`, applied as a
  *runtime multiplier* on top of `PLAYBACK_SPEED` and the cadence-aware
  `framesPerSec` the player computes per replay. 1× means "whatever the
  player picked"; the multiplier never replaces the auto-cadence logic.

The bar sits at z-index 9, 0.55 opacity, fading to 1.0 on hover, so it
stays out of the way during passive viewing.

## Status row

Below the recent-runs strip, the top bar shows the live player
status — `polling` / `loading <file>` / `playing <file> (N frames over
~Xs, fps)` / `skipped <file> (0 frames)` — alongside iteration, fitness,
model tag (`[gnn]` or `[edge]`), and board signature
(`r<radius> · p<num_players> · n<num_nodes> · <stride>t/frame · <s>s`).

## Auto-cadence playback

Playback follows game time, not "fit this replay into N seconds". With
`dt_per_tick_ms=100` and `tick_stride=1`, the UI shows every recorded
physics tick at 10 ticks/sec. Wider strides still render correctly but
visibly jump by that many game ticks per frame.

When the index has timestamps, the player estimates the trainer's drop
cadence from the median delta between neighboring `saved_at` values and
speeds playback up so the current replay finishes slightly before the
next one is expected — never below the configured baseline tick rate.
This is what keeps a backlog from accumulating during live monitoring.

## Mixed-experiment safety

The displayer rebuilds geometry from each replay's
`(radius, num_players, num_nodes)` signature, so radius-5 and radius-9
runs can share the same replay index. When a *newly-arrived* replay has
a different board shape, the player interrupts the current replay
instead of waiting for it to finish; a static newest entry that's been
sitting in the index doesn't preempt every poll.

## FLXR v2 binary format

Same fixed header as v1 with `version=2`. Per-frame flow record is
**6 bytes** instead of 5: `src u16, dst u16, player u8, pressure_q u8`.
`pressure_q` is quantized 0..255 from 0..MAX_EDGE=100. Geometry is
derived from `(radius, num_players)` by `buildBoard` on the client.
Writer side: [`python/flux_v2/replay.py`](../../python/flux_v2/replay.py).
Parser side: [`src_v2/replay/format.ts`](../../src_v2/replay/format.ts).

## Layout

```
index-v2.html               Vite entrypoint (root); served at /index-v2.html
src_v2/board.ts             hex grid builder (mirrors python/flux_v2/graph.py)
src_v2/replay/format.ts     FLXR v2 parser (6-byte flow records)
src_v2/replay/player.ts     index poller + frame advancer + transport API
src_v2/render/scene.ts      three.js orthographic scene, instanced nodes
src_v2/render/topbar.ts     iter row + recent-runs strip
src_v2/render/playback.ts   bottom transport bar
src_v2/main.ts              entry: wires it all up
```

## What's deliberately not there

Holding the displayer to "trainer-displayer" — no live sim, no
parameter knobs — keeps the surface small enough to evolve alongside
the v2 ruleset without breaking the v1 entrypoint. The decisions:

- No three.js debug GUI (`lil-gui`).
- No live-sim toggle, no tunables panel.
- No selection ring, no drag-to-edit affordances.
- No restart / evolve buttons.

What's kept from v1: same colors, same hex layout, same
arrow-thickness-by-pressure flow rendering, same auto-poll-and-swap
replay flow.

## Operational pointers

- Replays land at `public/v2/replays/*.flxr` + `index.json`; the Vite
  dev server serves them at `/v2/replays/`. `.gitignore` already
  excludes them, so long runs don't pollute git.
- Each replay at `record-stride=25` is ~1.5 MB and the index keeps 50,
  so steady-state disk use is small. Tick-by-tick (`record-stride=1`)
  was the v2 default early on; runs that need play-by-play resolution
  still pass `--record-stride 1`.
- To preserve a notable replay across the 50-cap rotation, force-add it
  to git: `git add -f public/v2/replays/<file>.flxr`.

Related: [[v2-trainer-displayer]] (the *why* of a separate UI track),
[[v2-training-runs]] (how to launch a run that feeds this viewer),
[[v2-edge-pressure-state]], [[v2-rules-one-pager]],
[[decisions/replay-rendering]].
