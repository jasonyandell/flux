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

For the full viewer feature catalog (recent-runs list, transport bar,
scrubber, speed cycle, auto-cadence, mixed-radius rebuild, FLXR v2
binary format, layout) see [[topics/v2-viewer]]. This page is just the
decision to keep v2 on its own UI track.

## Why separate

The PRD's "delivery shape" pins this: v2's simulation is different enough
from v1 (edge state, action space, capture strength, regen scaling) that
sharing `src/game/` would force compatibility code that would slow both
tracks down. Cleaner to have v1 keep running unmodified while v2 trains.

## What's stripped vs v1

- No three.js debug GUI (`lil-gui`).
- No live-sim toggle, no tunables panel.
- No selection ring, no drag-to-edit affordances.
- No restart / evolve buttons.

What's kept: same colors, same hex layout, same arrow-thickness-by-pressure
flow rendering, same auto-poll-and-swap replay flow.

What's added on top: a clickable newest-first recent-runs strip in the
top bar plus a media-style transport bar at the bottom. Details in
[[topics/v2-viewer]].

## Playback contract

The displayer rebuilds geometry from each replay's
`(radius, num_players, num_nodes)` signature, so mixed training streams
such as radius-9 followed by radius-5 can share the same replay index.
When a new replay has a different board shape, the player interrupts the
old replay instead of waiting for it to finish.

Playback follows game time, not "fit this replay into N seconds". With
`dt_per_tick_ms=100` and `tick_stride=1`, the UI shows every recorded
physics tick at 10 ticks/sec. The user-facing speed cycle in the
transport bar applies a runtime multiplier on top of this baseline.

## Replay drop location

The static replays land at `public/v2/replays/*.flxr` and `index.json`,
served at `/v2/replays/`. Index keeps the 50 newest entries
(`append_index(cap=50)` in `python/flux_v2/replay.py`), oldest evicted
on append.

Related: [[v2-edge-pressure-state]], [[topics/v2-viewer]],
[[replay-rendering]].
