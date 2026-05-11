---
title: flux
kind: entity
first_seen: bootstrap
last_updated: workspace
status: active
---

## Current shape

flux is a minimal browser implementation of a [[galcon-like]]. The board is a hex grid of ~1000 axial-coordinate cells (default `radius = 18`, see [[hex-grid-default]]) with distance-2 connectivity (~8.7k undirected edges). Each cell has an owner and a continuous strength. Owners regen strength over time. Players toggle flows along edges to drain their own cells and add to (or subtract from) neighbours. When a cell's strength crosses zero under enemy pressure, ownership flips with the surplus becoming the new owner's foothold. Win = sole remaining player.

Default mode is [[multi-player-free-for-all]]: 6 AI-controlled seats placed evenly around the perimeter, each running a randomly-assigned controller from the [[ai-zoo]]. Click input is disabled; input is map-style camera only: mouse-wheel zoom anchored on cursor, one-finger drag pan, two-finger pinch + pan anchored on centroid. The camera position is clamped against `scene.worldHalfWidth`/`worldHalfHeight`, so at max zoom out the board recenters and zoom in/out can't leave it offset. The game plays at 5× wall-clock speed via a constant scaler in `main.ts`.

## Loop

```txt
each AI seat picks flows on its current state
        ↓
flows drain sources, push at destinations (heal friendly, attack non-friendly with attack-bonus)
        ↓
captures, recaptures, eventual collapse to one survivor
```

## Implementation frontier

- `src/game/` — pure TypeScript core. No DOM, no three.js. Lives by [[pure-step-function]].
  - `state.ts` — types and tunable constants (`REGEN_PER_SEC`, `TRANSFER_PER_SEC`, `MAX_STRENGTH`, `MIN_STRENGTH_TO_SEND`, `ATTACK_BONUS`).
  - `graph.ts` — hex grid generator with axial coordinates, configurable radius/distance/numPlayers. Player seats are perimeter cells sorted by polar angle.
  - `step.ts` — `step(state, dt) → state` and `applyAction(state, action) → state`. `applyAction` caches edge adjacency in a `WeakMap` keyed on the edges array.
- `src/ai/` — [[ai-zoo]]. Six pure heuristics (`aggressive`, `random`, `defensive`, `greedy-neutral`, `opportunist`, `cluster`) plus shared `utils.ts` and seeded `rng.ts`. `index.ts` registers them, plus the `evolved` controller exported from `src/gpu/evolved.ts`. Each is `(state, player, seed?) => Action[]`.
- `src/gpu/` — WebGPU runtime + compute kernels for [[../decisions/webgpu-evolution|webgpu-evolution]]. `runtime.ts` returns a `{device, queue}` or `null` (falls back when WebGPU is missing). `shaders/step.wgsl` ports `step.ts` 1:1 across a games-batch; `shaders/nn.wgsl` runs the 91→32→19 per-cell forward pass and rebuilds flows from the action map. `evolution.ts` orchestrates a generation (upload weights, run N games for K ticks, read back owner buffer, tournament + mutation). `genome.ts` defines layout and a matching JS forward pass. `evolved.ts` exposes `aiThink` (uses the current champion genome) and a `setChampion` setter the evolution loop drives. `parity.ts` exposes `runParityTest` — the WGSL step must match the JS step within `1e-3` on strength and exact on owner/flow set.
- `src/render/scene.ts` — three.js orthographic top-down view. Nodes are one `InstancedMesh`; edges are one baked `LineSegments`; flows are a per-frame rebuilt `LineSegments` drawing the source-side half of each flow. 12-color palette exported as `COLORS`.
- `src/render/gameui.ts` — banner / pause / hint overlays plus `createTopBar()`. `showBanner` colors per winner; `getWinner` returns the sole remaining player or null. The top bar is the casual-demo surface: a centered row of dim monospace text-buttons (no background pill, no border) — `↻ restart`, an `evolve` toggle whose state shows as a tiny hollow/glowing-amber dot, and a `gen N · best F` readout. The evolve button stacks `drains battery` as a small subscript under the label so the warning stays visible without consuming horizontal space. `white-space:nowrap` on the root prevents the caption from breaking; the label's `min-width:8ch` keeps the column stable across `evolve`/`evolving`. lil-gui is collapsed by default and reserved for advanced/dev tuning.
- The status HUD (`index.html` `#hud`, populated by `updateHud` in `main.ts`) is a tap-to-expand expando: collapsed it shows `▸ tick N · X/Y alive`; expanded it lists each seat with color swatch, AI name, and cell count (dead seats dimmed) plus a neutral row. Sized for mobile — collapsed line stays narrow so it doesn't collide with the centered top bar.
- `src/input/pick.ts` — distance-based picking against `scene.nodePositions`; `eventToWorld` is the anchor helper used by both wheel zoom and the pinch centroid.
- `src/main.ts` — browser entry. Spectator: wheel zoom + pointer pan/pinch via a shared `zoomAndPanAt(before, after, factor)` helper. Tracks active pointers in a flat list and derives centroid + mean spread to drive zoom-and-pan in one step. lil-gui exposes `players` (2/4/6/8/12), `paused`, `aiPeriodSec`, `respawn`. Render at 60Hz; game ticks at 10Hz × `SPEED = 5`. Maintains the ring buffer that drives [[../decisions/stasis-detection|stasis-detection]].
- `src/sim/run.ts` — headless runner via `tsx`. Supports default, pair (`npm run sim -- agg random 10`), and tournament (`npm run sim -- tournament 3`) modes.
- `src/sim/stasis.ts` — pure variance-window detector behind the browser's `STASIS` banner. See [[../decisions/stasis-detection|stasis-detection]].

The game model is documented in [[continuous-flow-model]]. [[neuroevolution]] is in progress via [[../decisions/webgpu-evolution|webgpu-evolution]]; the `evolved` seat is selectable in the spectator zoo.
