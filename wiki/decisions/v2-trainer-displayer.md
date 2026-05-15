---
title: v2 — separate UI track, trainer-displayer only
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-14
status: active
---

## What

v2 ships as a separate web entrypoint (`index-v2.html`, `src_v2/`). It is a
**trainer-displayer**, not a simulator — it plays back `.flxr` v2 replays
produced by `scripts/train_v2.py`. No in-browser game logic.

## Why separate

The PRD's "delivery shape" pins this: v2's simulation is different enough
from v1 (edge state, action space, capture strength, regen scaling) that
sharing `src/game/` would force compatibility code that would slow both
tracks down. Cleaner to have v1 keep running unmodified while v2 trains.

## What's stripped vs v1

- No three.js debug GUI (`lil-gui`).
- No live-sim toggle, no tunables panel.
- No selection ring, no drag-to-edit affordances.
- Top bar shows iter/gen + recent replay drips.

What's kept: same colors, same hex layout, same arrow-thickness-by-pressure
flow rendering, same auto-poll-and-swap replay flow.

What's added: a media-style transport bar (see [Transport controls](#transport-controls)).

## Playback contract

The displayer rebuilds geometry from each replay's
`(radius, num_players, num_nodes)` signature, so mixed training streams
such as radius-9 followed by radius-5 can share the same replay index.
When a new replay has a different board shape, the player interrupts the
old replay instead of waiting for it to finish.

Playback follows game time, not "fit this replay into N seconds". With
`dt_per_tick_ms=100` and `tick_stride=1`, the UI shows every recorded
physics tick at 10 ticks/sec. Wider strides still render correctly, but
they visibly jump by that many game ticks per frame. The user-facing
speed cycle in the transport bar applies a runtime multiplier on top of
this baseline.

## Layout

```
index-v2.html               Vite entrypoint (root); served at /index-v2.html
src_v2/board.ts             hex grid builder (mirrors python/flux_v2/graph.py)
src_v2/replay/format.ts     FLXR v2 parser (6-byte flow records)
src_v2/replay/player.ts     index poller + frame advancer + transport API
src_v2/render/scene.ts      three.js orthographic scene, instanced nodes
src_v2/render/topbar.ts     iter/recent-replay drip-feed bar
src_v2/render/playback.ts   bottom transport bar (prev / step / play / step / next + scrubber + speed)
src_v2/main.ts              entry: wires it all up
```

## Transport controls

The displayer is a player, so it gets player UI. A bottom-fixed bar carries:

```
[⏮]  [⏪]  [⏯]  [⏩]  [⏭]   ━━━●━━━━━   12 / 240   1×
 prev  step  play  step  next   scrubber    counter   speed
```

- **Prev / Next replay** walk the index round-robin (`Shift+←` / `Shift+→`).
  Skipped (unplayable) entries are filtered out, same set the auto-cycle uses.
- **Step back / forward one frame** (`←` / `→`) auto-pause and jog by ±1.
- **Play / Pause** (`Space`) freezes the current frame. While paused, the
  index poller still discovers new replays but does not preempt the current
  one — auto-cycle to the next replay only fires from the end-of-replay
  branch and is gated on `!paused`. Manual prev/next still works (`forceLoad`
  bypasses the gate).
- **Scrubber** is a `<input type="range">` over the frame index (0..N-1).
  Dragging auto-pauses; the bar is suppressed from programmatic updates
  while the user has the slider grabbed so playback doesn't fight the drag.
- **Speed cycle** picks 0.25 / 0.5 / 1 / 2 / 4×, applied as a runtime
  multiplier on top of the configured `PLAYBACK_SPEED` and the cadence-aware
  `framesPerSec` the player computes per replay. 1× means "whatever the
  player picked"; the multiplier never replaces the auto-cadence logic.

The bar lives at z-index 9 with 0.55 opacity, fading to 1.0 on hover, so it
stays out of the way during passive viewing.

The static replays land at `public/v2/replays/*.flxr` and `index.json`, served
at `/v2/replays/`.

## FLXR v2 binary format

Same fixed header as v1 with `version=2`. Per-frame flow record is **6 bytes
instead of 5**: `src u16, dst u16, player u8, pressure_q u8`. `pressure_q` is
quantized 0..255 from 0..MAX_EDGE=100. Geometry is derived from
`(radius, num_players)` by `buildBoard` on the client.

Related: [[v2-edge-pressure-state]], [[replay-rendering]].
