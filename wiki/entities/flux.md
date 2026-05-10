---
title: flux
kind: entity
first_seen: bootstrap
last_updated: workspace
status: active
---

## Current shape

flux is a minimal browser implementation of a [[galcon-like]]. The board is a hex grid of ~1000 axial-coordinate cells (default `radius = 18`, see [[hex-grid-default]]) with distance-2 connectivity (~8.7k undirected edges). Each cell has an owner and a continuous strength. Owners regen strength over time. Players toggle flows along edges to drain their own cells and add to (or subtract from) neighbours. When a cell's strength crosses zero under enemy pressure, ownership flips with the surplus becoming the new owner's foothold. Win = sole remaining player.

Default mode is [[multi-player-free-for-all]]: 6 AI-controlled seats placed evenly around the perimeter, each running a randomly-assigned controller from the [[ai-zoo]]. Click input is disabled in this mode; mouse-wheel zoom remains. The game plays at 5× wall-clock speed via a constant scaler in `main.ts`.

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
- `src/ai/` — [[ai-zoo]]. Six pure heuristics (`aggressive`, `random`, `defensive`, `greedy-neutral`, `opportunist`, `cluster`) plus shared `utils.ts` and seeded `rng.ts`. `index.ts` registers them. Each is `(state, player, seed?) => Action[]`.
- `src/render/scene.ts` — three.js orthographic top-down view. Nodes are one `InstancedMesh`; edges are one baked `LineSegments`; flows are a per-frame rebuilt `LineSegments` drawing the source-side half of each flow. 12-color palette exported as `COLORS`.
- `src/render/gameui.ts` — banner / pause / hint overlays. `showBanner` colors per winner; `getWinner` returns the sole remaining player or null.
- `src/input/pick.ts` — distance-based picking against `scene.nodePositions`; `eventToWorld` for the wheel-zoom cursor anchor.
- `src/main.ts` — browser entry. Spectator: wheel zoom only, no click input. lil-gui exposes `players` (2/4/6/8/12), `paused`, `aiPeriodSec`, `respawn`. Render at 60Hz; game ticks at 10Hz × `SPEED = 5`. Maintains the ring buffer that drives [[../decisions/stasis-detection|stasis-detection]].
- `src/sim/run.ts` — headless runner via `tsx`. Supports default, pair (`npm run sim -- agg random 10`), and tournament (`npm run sim -- tournament 3`) modes.
- `src/sim/stasis.ts` — pure variance-window detector behind the browser's `STASIS` banner. See [[../decisions/stasis-detection|stasis-detection]].

The game model is documented in [[continuous-flow-model]]. The planned next step is [[neuroevolution]].
