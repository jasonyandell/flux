# flux — agent notes

Read `wiki/AGENTS.md` first. The wiki is the project's frontier of truth; consult it before code.

## Run

```sh
npm install
npm run dev          # browser
npm run sim          # headless simulation (npm run sim -- 500 for 500 runs)
npm run typecheck
npm run build
```

## House rules

- `src/game/` and `src/ai/` are pure TypeScript. Never import `three`, `lil-gui`, or touch DOM/window from there.
- `step(state, dt) → state` is the contract. Don't mutate state.
- The browser and the headless sim share the same `step` and `aiThink`. If they disagree, the renderer is wrong.
- Update the wiki when truth changes; see `wiki/AGENTS.md`.

For everything else (game model, architecture, tradeoffs), start at `wiki/entities/flux.md`.
