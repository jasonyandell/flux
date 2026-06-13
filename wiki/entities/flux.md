---
title: flux
kind: entity
first_seen: bootstrap
last_updated: workspace
status: active
---

## What flux is now

flux is a research codebase exploring [[galcon-like]] RTS as a substrate
for neural and algorithmic agents on a hex graph. It has split into **two
tracks** that share theming and graph shape but nothing else:

- **v1** — the original browser game. Continuous-strength cells, transfer-
  flow combat, hand-coded [[ai-zoo]] in a free-for-all spectator. WebGPU
  evolution lives here ([[../decisions/webgpu-evolution|webgpu-evolution]])
  and Python-trained MLX champions plug in via `src/gpu/evolved.ts`.
  Deployed at `/` and `/?demo=1` (see
  [[../topics/showcase-demo|showcase-demo]]). At this point v1 is **frozen
  at saturation** — the long-running neuroevolution arc topped out around
  gen 5372 / fit 1540 and is no longer where new design happens.

- **v2** — the current frontier. Persistent-edge **pressure-state** sim
  ([[../decisions/v2-edge-pressure-state|v2-edge-pressure-state]]),
  13-action Set/Clear policy
  ([[../decisions/v2-set-clear-actions|v2-set-clear-actions]]), trained in
  Python under `python/flux_v2/` + `python/scripts/train_v2.py`, played
  back in the browser by a stripped trainer-displayer at `/index-v2.html`
  ([[../decisions/v2-trainer-displayer|v2-trainer-displayer]],
  [[../topics/v2-viewer|v2-viewer]]). The simulator is bit-exact in Python;
  the browser never re-implements game logic.

If you're reading this to understand "the project right now," skim the v2
section below and read [[../topics/v2-rules-one-pager|v2-rules-one-pager]].

## v2 frontier (read this)

State, rules, and reward shape are pinned in [[../v2-prd|v2-prd]]. The
compact reference is [[../topics/v2-rules-one-pager|v2-rules-one-pager]].
Active threads:

- **Pete** ([[../topics/v2-vectorized|v2-vectorized]]). Pete is the
  vectorized v2 generator / trainer / solver path: JIT board setup,
  vectorized solver execution, batched per-AI-tick runs, FLXR v3 replay
  output, and the fast measurement loop. "Run that through Pete" means use
  this path as the lab surface before trusting slower legacy loops.

- **Todd** ([[../topics/v2-todd-measurement-lab|v2-todd-measurement-lab]])
  and the **Pete factory**
  ([[../topics/v2-pete-factory|v2-pete-factory]]). Todd is the M5-scale
  measurement lab: matched pairs, scoreboards, and wandb summaries. The Pete
  factory produces deterministic raw material for Todd: boards, games, replay
  samples, teacher shards, and divergence candidates.

- **Algorithmic solvers**
  ([[../topics/v2-algorithmic-solvers|v2-algorithmic-solvers]]). BFS plus
  a family of Lightning solvers — `max`, `sum`, `sum_pw`, `sum_long`,
  `sum_wave`, `wave_long`, `loop`, `attn`, plus pulse/wave/vortex/flood
  variants. All live in `python/flux_v2/solver_lightning.py` and are
  registered in `python/scripts/run_v2_solver.py`. Current head-to-head
  ranking under the big-bag-of-pressure rules (R=20, 10% dead,
  matched-pair tournaments): **`wave_long` > `sum` > `bfs` ≈ `max` >>
  `attn` >> `pulse` / `pulse_stagger`** — see
  [[../topics/v2-overnight-research|v2-overnight-research]] for the
  methodology and the noise floor (naive single-direction tournaments hit
  a 6pp seat-bias floor). The diffusion-mode design story is in
  [[../topics/v2-edge-loop-emergence|v2-edge-loop-emergence]]; its
  internal "final ranking" is superseded by the overnight matched-pair
  results.

- **PPO** ([[../decisions/ppo-gnn|ppo-gnn]],
  [[../decisions/v2-three-term-reward|v2-three-term-reward]]). PPO + GCN
  backbone on the v2 sim, trained via MLX autograd in
  `python/flux_v2/ppo.py` and `python/scripts/train_v2.py`. Operational
  notes and run history in
  [[../topics/v2-training-runs|v2-training-runs]]. As of 2026-05-15, PPO
  has not yet caught the best hand-coded solver under big-bag rules; the
  attention-headed PPO variant was abandoned after losing decisively. The
  open question is whether a learned edge-voting head
  ([[../topics/v2-edge-voting-policy|v2-edge-voting-policy]], proposed)
  can close the gap.

- **Boards.** Connectivity invariants in
  [[../decisions/v2-board-connectivity|v2-board-connectivity]]. Active
  arena sizes range from R=5 P=6 ("tiny single-tick") to R=25 P=6
  (overnight tournaments).

## v1 surface (still deployed, not the frontier)

The v1 browser game and its supporting Python port are intact. They are
useful as: a deployed live demo (`/`, `/?demo=1`), the WGSL ↔ JS ↔ NumPy
parity reference ([[../decisions/python-port|python-port]]), and the home
of the WebGPU evolution experiment
([[../decisions/webgpu-evolution|webgpu-evolution]] — now a side-quest, no
longer the training path). The model is documented in
[[../decisions/continuous-flow-model|continuous-flow-model]].

`src/game/` and `src/ai/` are still pure TypeScript with the
`step(state, dt) → state` contract
([[../decisions/pure-step-function|pure-step-function]]) and remain the
reference for anything that asks "how does v1 compute combat."

## Implementation map

- **v2 simulation** — `python/flux_v2/`
  - `state.py`, `graph.py`, `step.py` — pure pressure-state sim.
  - `solver.py`, `solver_lightning.py` — algorithmic baselines.
  - `ppo.py` — PPO + GCN policy with MLX autograd.
  - `edge_features.py`, `edge_flow.py` — staged edge-centric perception
    (see [[../topics/v2-edge-voting-policy|v2-edge-voting-policy]]).
  - `replay.py` — `.flxr` v2 binary writer; index format at
    `public/v2/replays/index.json`.
- **v2 trainer-displayer** — `src_v2/`
  - `main.ts` entry; `render/scene.ts`, `render/topbar.ts`,
    `render/playback.ts`; `replay/format.ts`, `replay/player.ts`.
  - Vite entry at `index-v2.html`.
- **v2 training scripts** — `python/scripts/`
  - `train_v2.py` — PPO trainer.
  - `run_v2_solver.py` — algorithmic seat registry; tournament runner.
  - `sweep_*.py`, `pretrain_v2_edge_aux.py`, `build_greatest_hits.py`.
- **v1 game core** — `src/game/`, `src/ai/`, `src/gpu/`, `src/render/`,
  `src/input/`, `src/sim/`, `src/replay/`, `src/demo/`, `src/main.ts`.
- **v1 Python port** — `python/flux/`. Bit-exact reference for v1 plus
  MLX neuroevolution support; checkpoints at `python/checkpoints/`.
- **Replays** — `.flxr` is the contract
  ([[../decisions/replay-rendering|replay-rendering]]). v1 land in
  `public/replays/`, v2 in `public/v2/replays/`.

## What `step` means in each track

Both tracks share the invariant that the simulator is a pure function and
the browser never re-implements it.

- v1: `step(state, dt) → state` in TS at `src/game/step.ts` (and a WGSL
  twin at `src/gpu/shaders/step.wgsl`), NumPy twin at
  `python/flux/step.py`. Continuous strength; flow edges drain sources
  and push at destinations with an
  [[../decisions/attack-bonus|attack-bonus]].
- v2: `step(state)` in `python/flux_v2/step.py` (no `dt` — one physics
  tick at a time). Persistent directed half-edges; fill-then-overflow at
  the destination; capture is deterministic strength=1.

See also [[../decisions/regen-flow-rules|regen-flow-rules]] for the
parallel symmetric-damage ruleset proposed during the v2 reward-shaping
rework.

## Where to go from here

Context: [[galcon-like]], [[../topics/neuroevolution|neuroevolution]],
[[../topics/showcase-demo|showcase-demo]], [[../v2-prd|v2-prd]]. Active
research (freshest first):
[[../topics/v2-overnight-research|v2-overnight-research]],
[[../topics/v2-edge-loop-emergence|v2-edge-loop-emergence]],
[[../topics/v2-training-runs|v2-training-runs]].
