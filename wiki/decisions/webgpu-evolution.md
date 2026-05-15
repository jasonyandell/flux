---
title: WebGPU Evolution
kind: decision
first_seen: workspace
last_updated: workspace
status: superseded
---

## Status (2026-05)

Superseded as the active training path by Python + MLX neuroevolution (v1)
and PPO (v2). The browser still ships the WebGPU runtime, kernels, and
`evolved` seat, but champions now come from Python — the in-browser
evolution loop is no longer where the work happens. Kept live because: it
remains the parity reference for WGSL ≡ JS `step`, the trained `evolved`
seat plays in the v1 spectator zoo, and the architectural notes below are
still the cleanest explanation of how the WebGPU compute path was wired.
See [[../topics/neuroevolution|neuroevolution]] for the Tier 5 MLX path
that replaced this.

## Choice

Use WebGPU compute (via the browser, Metal under the hood on Apple Silicon) for
[[../topics/neuroevolution|neuroevolution]] of fixed-topology per-cell
controllers. Evolution loop in JS; game simulation and NN forward passes on the
GPU; champion genome serves as the `evolved` AI in the spectator zoo.

## Why not Python / pytorch / numpy

- **Single codebase.** The browser is already the deployment target; no install,
  no model server, no IPC. Everything ships when `npm run build` ships.
- **Metal under the hood.** On Apple Silicon, WebGPU translates to Metal, so the
  same kernels light up the unified-memory GPU that PyTorch uses via MPS.
- **The game already runs in the browser.** Spectator-mode evolution is the
  product. Round-tripping state through a Python process would be the heavy
  thing, not the GPU work.

## MVP architecture

1. **Population eval on GPU.** Each generation packs P=12 genomes into one flat
   weight buffer. K=ceil(P · 2 / numSeats) games (each ~4 seats), all advancing
   in parallel by ticking a `step_cells` compute shader that mirrors
   `src/game/step.ts` 1:1 across the games-dimension of the dispatch.
2. **NN forward on GPU.** Every `aiPeriodTicks`, an `nn_forward` kernel runs
   one thread per (cell, game). Each owned cell builds a 91-feature input from
   its own strength plus 18 sorted neighbors (strength + 4-way owner one-hot),
   runs 91 → 32 (tanh) → 19, argmaxes, writes an action index. A `build_flows`
   kernel then replaces each game's flow array from the new action map.
3. **Evolution loop in JS.** After K ticks, owner buffer reads back to CPU,
   fitness = mean cells owned across the genome's seat slots. Tournament
   selection (3-way), elites (top 3), gaussian mutation (σ=0.05). Re-upload
   weights, repeat.
4. **`evolved` AI seat.** `src/gpu/evolved.ts` holds the current champion in
   module state; its `aiThink` runs the same forward pass in JS (small enough
   that this is cheap) and emits `toggleFlow` actions to install the desired
   per-cell out-targets. Registered in `src/ai/index.ts` as `evolved`.

The browser drives generations in the background via chained
`setTimeout(0)`; the render loop is never blocked.

## Controller shape

- **Inputs (91):** own strength / `MAX_STRENGTH`, then per neighbor (18 slots,
  sorted by `(pos.x, pos.y)`, padded with -1 sentinels): strength/MAX,
  isFriendA, isFriendB (currently dup of A), isEnemy, isNeutral.
- **Hidden (32):** dense, tanh.
- **Outputs (19):** linear; argmax picks neighbor index 0..17 or 18 = "do
  nothing."
- **Genome:** 91·32 + 32 + 32·19 + 19 = 3571 weights as one `Float32Array`.

## Hyperparameters

| | |
|---|---|
| Population | 12 |
| Games per genome | 2 |
| Seats per game | 4 |
| Ticks per evaluation | 500 |
| Mutation σ | 0.05 (gaussian per weight) |
| Elites | 3 |
| Tournament K | 3 |
| Init weight σ | 0.5 |
| Board radius for evolution | 8 (~217 cells; spectator stays at 18) |

Fitness = mean cells owned by the genome's seats at game end.

## Parity invariant

`step_cells` ≡ `src/game/step.ts` for the same start state and flow array.
`src/gpu/parity.ts` exposes `runParityTest(ctx, opts)`; the lil-gui "run parity
test" button calls it. Tolerance is `1e-3` on strength (float ops aren't
bit-exact across IEEE/MAD-fusion); owner and flow set must match exactly.

If the parity test ever fails, the GPU step is the bug — JS step is the
contract per [[pure-step-function]]. Do not "fix" parity by changing JS.

## Falls back gracefully

`src/gpu/runtime.ts:initGPU()` returns null when `navigator.gpu` is absent or
adapter request fails. `main.ts` logs a warning, leaves the `evolve` toggle
inert, and the `evolved` AI uses a random initial genome (so it plays badly
but the game still runs).

## Out of scope (deferred to higher tiers)

- rtNEAT continuous replacement — batch generations only.
- Structural mutations / proper NEAT — fixed topology only.
- Genome save/load to disk.
- WebGPU for rendering — three.js retains the draw pipeline.
- Fancy UI for genome inspection / fitness graphs.

See [[../topics/neuroevolution|neuroevolution]] for the higher tiers.
