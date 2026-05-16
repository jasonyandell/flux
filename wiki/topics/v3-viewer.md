---
title: v3 viewer (sphere trainer-displayer)
kind: topic
first_seen: 2026-05-16
last_updated: 2026-05-16
status: active
---

## What this page is

The feature catalog for the **v3 sphere trainer-displayer** at
`/index-v3.html`. v3 plays back sphere-topology replays produced by
[`python/scripts/run_v3_solver.py`](../../python/scripts/run_v3_solver.py)
on a geodesic icosphere graph, rendered with electric "barbell" edge
tubes, selective bloom, and an activity-driven fade on cell brightness.

v3 is forked from the v2 trainer-displayer ([[v2-viewer]]) at a point in
time and then evolved independently:

- **Topology** — subdivided icosahedron (V = 2562 cells at subdiv 4),
  not a hex disc. Each cell has 6 neighbors except the 12 pentagonal
  vertices which have 5. The simulator carries a per-cell `back_slot`
  lookup so the reducer doesn't need the hex-grid `OPPOSITE_SLOT[k] =
  (k+3)%6` invariant.
- **Rendering** — three.js perspective camera + `OrbitControls` on a
  sphere shell, instanced cylinder mesh per undirected edge (the
  "barbells"), instanced node spheres, fresnel atmospheric halo,
  selective `UnrealBloomPass` restricted to the edge-tube layer only.
- **Independent of main's v2** — own `src_v3/`, own `python/flux_v3/`
  module, own `public/v3/replays/` directory, own binary format
  (the older FLXR v2 6-byte flow record format frozen at fork time).
  Main's v2 viewer and v3 sphere viewer don't share any code paths
  beyond what's incidentally identical between them, and can evolve
  independently without conflict.

## Layout

```
index-v3.html                Vite entrypoint; served at /index-v3.html
src_v3/board.ts              icosphere builder + sphere Lambert projection
                              (used for the placeholder hex board only —
                              real sphere geometry ships in metadata)
src_v3/replay/format.ts      FLXR-v2 parser; branches on graph_kind
                              to read pos3d + neighbors from metadata
src_v3/replay/player.ts      index poller + frame advancer
src_v3/render/scene.ts       three.js scene: bloom + edge shaders + halo
src_v3/render/topbar.ts      iter row + recent-runs strip
src_v3/render/playback.ts    bottom transport bar
src_v3/main.ts               entry; wires everything together
python/flux_v3/              forked simulator with sphere_graph.py +
                              per-cell back_slot State field
python/scripts/run_v3_solver.py
                              solver runner; writes to public/v3/replays/
public/v3/replays/           .flxr + index.json land here
```

## Running a sphere sim

```bash
python python/scripts/run_v3_solver.py \
    --subdiv 4 \
    --num-players 6 \
    --max-ticks 3000 \
    --record-stride 1 --ai-period-ticks 1 \
    --dead-frac 0.4 \
    --seats lightning_attn,lightning_sum_long,lightning_wave_keep_attack_long,lightning_pulse_stagger,bfs,lightning_vortex \
    --write-replay
```

- `--subdiv N` — icosahedron subdivision. V = 10·4^N + 2. Subdiv 4 →
  2562 cells, similar density to a radius-30 hex disc.
- `--dead-frac F` — fraction of cells DEAD. 0.40 = 40% dead; live
  subgraph is greedily sampled to stay connected, with
  `carve_seat_connectors` as a fallback to bridge any disconnected
  seats. Replays at this fraction look chaotic but always playable.
- `--record-stride 1 --ai-period-ticks 1` — every game tick is also
  an AI tick *and* a recorded frame, so every AI decision is visible
  on playback. ~14 KB / frame, so 1500 ticks ≈ 21 MB.

## Renderer

### Edges as instanced "barbells"

Every undirected adjacency on the icosphere is one cylinder instance
(~7680 at subdiv 4). Per-instance attributes:

- `instanceColor` — owner color when claimed, dim grey
  (`IDLE_EDGE_COLOR = 0x18243a`) when not.
- `aPressure` (float) — normalized 0..1 against the frame's max
  pressure, then `^0.7` curve. 0 means no flow recorded; small
  positive means a set outflow with built-up overflow.
- `aDirection` (float) — `+1` if flow goes from the cylinder's
  geometric `a` endpoint to `b`, `-1` reversed, `0` if no flow record
  this frame. **Gating the activity floor on `aDirection != 0` rather
  than `aPressure > 0` is critical** — it makes set-but-zero-pressure
  outflows (the moment a seat decides to push toward a neighbor, before
  any overflow has built up) visible immediately, not only after physics
  accumulates pressure.

### Vertex shader: bell-envelope bulge + traveling wave

Cylinder geometry is unit (radius 1, height 1 along local y). The
instance matrix scales to `(tubeR, length, tubeR)`, rotates local-y to
span src→dst, translates to the chord midpoint.

The shader applies a `sin(π·t)` envelope so active tubes bulge fattest
at the midpoint and taper smoothly to idle thickness at the endpoints.
This is load-bearing: the endpoint sits at the sphere center, and idle
thickness `tubeR = 0.07 · cellSpacing` fits inside the smallest sphere
radius (`0.0945 · cellSpacing`), so the cylinder never visibly overlays
or pokes through its destination node. Brief stint with uniform-radius
active tubes had corners "crowning" out the back of small spheres.

A traveling sine wave on top of the envelope provides the
direction-readable pulse without needing the spark to fire constantly.

### Fragment shader: pressure → brightness, hue-preserving cap

- `effP = (vDirection != 0) ? max(vPressure, FLOW_FLOOR=0.30) : 0`
- `baseBoost = 0.55 + 2.20 · effP` — idle 0.55×, full pressure ~2.75×
- Spark cycles src→dst at `speed = 0.7 + 1.6·effP`, with a tighter peak
  and a trailing afterglow
- Hue-preserving 75% brightness cap before bloom — divides by `maxComp`
  if any channel exceeds 0.75, so saturated edges never bleach to white

**`active` is a reserved word in GLSL** — used `flowing` instead.
Hours-saved-by-playwright lesson: load the page in chromium, capture
console output, and the silent shader compile error becomes visible.

### Selective bloom

`UnrealBloomPass` is restricted to one layer (`BLOOM_LAYER = 1`); only
the edge-tube mesh enables that layer. `render()` flips
`camera.layers.set(BLOOM_LAYER)` for the bloom composer pass and
`camera.layers.enable(0)` for the final composite. A `ShaderPass`
additively combines the bloom texture over the base render at 0.65×.

Net: arrows neon-glow at high pressure; nodes/halo/globe never halo.

### Node fade

Per-cell `framesSinceChange[i]` counter. Resets to 0 when this cell's
owner or quantized strength changes between recorded frames; otherwise
increments by 1 per playback frame advance (only when the player's
frame index actually moves — pausing/HMR doesn't tick the counter).

Brightness multiplier `fade = NODE_FADE_FLOOR + (1 - NODE_FADE_FLOOR) ·
exp(-fsc / NODE_FADE_TAU)` with floor 0.30 and τ 50 frames. Combined
with the static `NODE_BLOOM_MIN..MAX` band (0.85..1.40 across the
strength axis), it gives the eye a way to find the action: cells that
just changed pulse at the ceiling, cells sitting in the rear settle at
30% of their color.

## What's deliberately not there (vs main's v2)

- No FLXR v3 binary format support. The fork was taken before that
  landed; v3 reads the older v2 6-byte flow record format with a
  `graph_kind` metadata branch that adds the pos3d + neighbors arrays.
- No JIT'd hot path / numba acceleration. The sphere solver runs the
  python reducer directly; perfectly fast enough for the modest cell
  counts the sphere viewer cares about (a 2562-cell 9k-tick game runs
  in ~90 seconds).
- No playlist panel. Recent-runs strip in the top bar only.

## Related

- [[v2-viewer]] — main's hex viewer (independent codebase)
- [[v2-trainer-displayer]] — why v2 is a separate UI track
- [[v2-rules-one-pager]] — game rules (shared with v3)
