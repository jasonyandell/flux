---
title: flux
kind: entity
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Current shape

flux is a minimal browser implementation of a [[galcon-like]]. The board is a small graph of nodes (~7) with bidirectional edges; each node has an owner and a continuous strength. Owners regen strength over time. Players toggle flows along edges to drain their own nodes and add to (or subtract from) neighbors. When a node's strength crosses zero under enemy pressure, ownership flips with the surplus becoming the new owner's foothold. Win = own all non-neutral capacity.

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
  - `graph.ts` — hand-laid 7-node mirror-symmetric graph with two starting bases.
  - `step.ts` — `step(state, dt) → state` and `applyAction(state, action) → state`.
- `src/ai/dumb.ts` — `aiThink(state, player) → Action[]`.
- `src/render/scene.ts` — three.js orthographic top-down view; reads state, never mutates.
- `src/input/pick.ts` — raycasting for click-to-select.
- `src/main.ts` — browser entry; render at 60Hz, game ticks at 10Hz, AI ticks at ~2Hz.
- `src/sim/run.ts` — headless N-game runner via `tsx`. Imports the same `step` as the browser.

The game model is documented in [[continuous-flow-model]].
