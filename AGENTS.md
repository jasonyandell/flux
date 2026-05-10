# flux — agent notes

- Consult `wiki/index.md` first, then read code. The wiki is the project memory; update it whenever the frontier changes.
- Follow `wiki/AGENTS.md` for page shape, backlinks, source digests, and log updates.
- `src/game/` and `src/ai/` are pure TypeScript. Never import `three`, `lil-gui`, or touch DOM/window from there.
- `step(state, dt) → state` is the contract. Don't mutate state. The browser and the headless sim share the same `step` and `aiThink`; if they disagree, the renderer is wrong.

## Run

```sh
npm install
npm run dev          # browser
npm run sim          # headless simulation (npm run sim -- 500 for 500 runs)
npm run typecheck
npm run build
```

For everything else (game model, architecture, tradeoffs), start at `wiki/entities/flux.md`.
