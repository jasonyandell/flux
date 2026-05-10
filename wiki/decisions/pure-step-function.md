---
title: Pure step function
kind: decision
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Decision

The whole game advances through one pure function: `step(state, dt) → state`. Player and AI intent enters via `applyAction(state, action) → state`. Both are exported from `src/game/step.ts`.

## Constraints

- `src/game/` and `src/ai/` MUST NOT import `three`, `lil-gui`, or touch any DOM/window globals.
- `step` MUST NOT mutate its input. It returns a new state.
- `step` MUST be deterministic given `(state, dt)`.

## Consequences

- The browser renders state and dispatches actions; it never reaches into game logic.
- The headless runner (`src/sim/run.ts`) imports the same `step` and `aiThink`. If the sim disagrees with the browser, the rendering layer is wrong.
- AI is a function `(state, player) → Action[]`. New AIs slot in the same way.

## Why

- One implementation, one source of truth.
- Replays and tests reduce to comparing state sequences.
- Balance changes are a single constant edit, with `npm run sim` for fast feedback.
