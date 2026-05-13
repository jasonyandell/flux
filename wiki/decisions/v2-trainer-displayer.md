---
title: v2 — separate UI track, trainer-displayer only
kind: decision
first_seen: 2026-05-13
last_updated: 2026-05-13
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

## Layout

```
index-v2.html               Vite entrypoint (root); served at /index-v2.html
src_v2/board.ts             hex grid builder (mirrors python/flux_v2/graph.py)
src_v2/replay/format.ts     FLXR v2 parser (6-byte flow records)
src_v2/replay/player.ts     index poller + frame advancer
src_v2/render/scene.ts      three.js orthographic scene, instanced nodes
src_v2/render/topbar.ts     iter/recent-replay drip-feed bar
src_v2/main.ts              entry: wires it all up
```

The static replays land at `public/v2/replays/*.flxr` and `index.json`, served
at `/v2/replays/`.

## FLXR v2 binary format

Same fixed header as v1 with `version=2`. Per-frame flow record is **6 bytes
instead of 5**: `src u16, dst u16, player u8, pressure_q u8`. `pressure_q` is
quantized 0..255 from 0..MAX_EDGE=100. Geometry is derived from
`(radius, num_players)` by `buildBoard` on the client.

Related: [[v2-edge-pressure-state]], [[replay-rendering]].
