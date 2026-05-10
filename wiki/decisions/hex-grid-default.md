---
title: Hex grid default
kind: decision
first_seen: 2026-05-10
last_updated: 2026-05-10
status: active
---

## Decision

`makeInitialState(radius = 18, distance = 2)` produces a hexagonal board of axial-coordinate cells where `max(|q|, |r|, |s|) ≤ radius` (s = -q-r). All pairs within hex distance ≤ `distance` are connected by an undirected edge. Default board: 1027 cells, 8691 edges. Two starting bases at the axial extremes `(-radius, 0)` and `(radius, 0)`, strength 30; everything else neutral, strength 10.

Pointy-top axial-to-pixel layout with size 1: `x = √3·q + (√3/2)·r`, `y = (3/2)·r`. Hex distance via cube coords: `max(|Δq|, |Δr|, |Δq + Δr|)`. See [redblobgames hexagons](https://www.redblobgames.com/grids/hexagons/).

## Why

- The 7-node hand-laid graph is too small to expose strategy. A 1000-cell board has emergent fronts, choke points, and multiple parallel routes.
- Distance-2 connectivity adds the six diagonal neighbours so the front line is not a single brittle ring; the dumb AI's "attack weakest neighbour" heuristic has more local options.
- Radius 18 was picked as the largest value that gives a near-1000 cell count (`3·r·(r+1) + 1`) while keeping the camera frame comfortable at common aspect ratios.

## Consequences

- **Renderer:** per-node `THREE.Mesh` does not scale. Nodes are batched into a single `THREE.InstancedMesh`, edges into one `LineSegments` baked at creation, flows into a dynamic `LineSegments` rebuilt each frame. The per-node HTML overlay (totals/deltas) was deleted — 1000 absolutely-positioned spans tank the page and the strength is already conveyed by node scale and color.
- **Flow rendering:** dropped the arrowhead and second-half segment. Each flow is one colored line from source to midpoint — direction is encoded by drawing only the source-side half. Cleaner at 1000-cell density than the original arrow shape.
- **`applyAction` adjacency:** with 8691 edges the per-call `edges.some(...)` scan was the largest sim hotspot. Now backed by a `WeakMap<edges, Set<edgeKey>>` keyed on the edges array identity — built once per board.
- **`aiThink` neighbours:** now builds a local adjacency list `NodeId[][]` once per call instead of scanning all edges per node. ~22× faster per sim run.
- **Headless sim:** default run count dropped from 100 to 10 in `src/sim/run.ts`; 10 runs is ~6 seconds. `MAX_TICKS` unchanged. Still 100% draws — the dumb-AI stalemate persists at this scale, possibly more interesting given hex topology.

## Rejected

- **Keep `overlay.ts` behind a node-count threshold.** Adds a branch for a code path nobody will run again. Minimalism wins; git keeps the file.
- **Per-flow arrowhead at 1000-cell scale.** Three line segments per flow (shaft + two head wings) plus the per-frame rebuild is too noisy visually and wasteful. The midpoint half-line conveys direction cheaply.
