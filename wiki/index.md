---
title: flux Wiki Index
kind: index
first_seen: bootstrap
last_updated: workspace
status: active
---

## Route map

Start at [[entities/flux|flux]] for the whole project.

Core entities:

- [[entities/flux|flux]] — the game.

Topics:

- [[topics/galcon-like|galcon-like]] — the genre and its lineage.
- [[topics/neuroevolution|neuroevolution]] — planned next step; rtNEAT-style real-time evolution of NN controllers.
- [[topics/showcase-demo|showcase-demo]] — `?demo=1` evolution-arc demo: five 5s scenes, canned champions in `public/champions/`, hot-area intro + "AI WARS" title card.

Decisions:

- [[decisions/continuous-flow-model|continuous-flow-model]] — why strength is a continuous scalar with rate-based flows.
- [[decisions/pure-step-function|pure-step-function]] — game logic is a pure `step(state, dt) → state`, shared by browser and headless sim.
- [[decisions/one-flow-per-edge|one-flow-per-edge]] — at most one flow per undirected edge; reverse means flip.
- [[decisions/hex-grid-default|hex-grid-default]] — default board is a ~1000-cell hex grid with distance-2 connectivity; renderer uses instancing.
- [[decisions/attack-bonus|attack-bonus]] — non-friendly destinations get a `1 + ATTACK_BONUS` multiplier so attacks deal more damage than they cost.
- [[decisions/ai-zoo|ai-zoo]] — six hand-written heuristics registered under `src/ai/`; tournament shows four converge to the same play.
- [[decisions/multi-player-free-for-all|multi-player-free-for-all]] — default is 2–12 AI seats on the hex perimeter; spectator mode at 5× speed.
- [[decisions/stasis-detection|stasis-detection]] — variance-window detector that ends stalemated spectator sessions with a `STASIS` banner; suppressed in 1v1 endgame and cleanup phases.
- [[decisions/webgpu-evolution|webgpu-evolution]] — fixed-topology neuroevolution on WebGPU compute. Population eval on GPU, JS evolution loop, champion plays as `evolved` in the spectator zoo. WGSL step ≡ JS step parity invariant.

Questions:

- [[questions/open|open]] — unresolved questions.

Process:

- [[todo|todo]] — active threads scratchpad (pre-issues).
