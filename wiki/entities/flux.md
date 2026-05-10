---
title: flux
kind: entity
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Current shape

flux is a minimal browser implementation of a [[galcon-like]]. The board is a hex grid of ~1000 axial-coordinate cells (default `radius = 18`, see [[hex-grid-default]]) with distance-2 connectivity (~8.7k undirected edges). Each cell has an owner and a continuous strength. Owners regen strength over time. Players toggle flows along edges to drain their own cells and add to (or subtract from) neighbours. When a cell's strength crosses zero under enemy pressure, ownership flips with the surplus becoming the new owner's foothold. Win = own all non-neutral capacity.

## Loop

```txt
toggle flow on edge
        ↓
drain source, push at destination (heal or attack)
        ↓
capture or hold; repeat
```

Two players: one human (player 0) and one heuristic AI (player 1, `src/ai/dumb.ts`). The AI picks its strongest non-sending owned node and starts a flow to its weakest non-friendly neighbor.

## Implementation frontier

- `src/game/` — pure TypeScript core. No DOM, no three.js. Lives by [[pure-step-function]].
  - `state.ts` — types and tunable constants.
  - `graph.ts` — hex grid generator with axial coordinates and configurable radius/distance.
  - `step.ts` — `step(state, dt) → state` and `applyAction(state, action) → state`. `applyAction` caches edge adjacency in a `WeakMap` keyed on the edges array.
- `src/ai/dumb.ts` — `aiThink(state, player) → Action[]`; builds a local adjacency list per call.
- `src/render/scene.ts` — three.js orthographic top-down view. Nodes are one `InstancedMesh`; edges are one baked `LineSegments`; flows are a per-frame rebuilt `LineSegments` drawing the source-side half of each flow.
- `src/input/pick.ts` — distance-based picking against `scene.nodePositions`; `eventToWorld` for the drag-line cursor.
- `src/main.ts` — browser entry; drag-to-associate input model (`pointerdown`/`pointermove`/`pointerup`, 8px threshold, `setPointerCapture`); render at 60Hz, game ticks at 10Hz, AI ticks at ~2Hz.
- `src/sim/run.ts` — headless N-game runner via `tsx`. Default `N = 10` (hex sims are slower per run). Imports the same `step` as the browser.

The game model is documented in [[continuous-flow-model]].
