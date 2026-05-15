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
- [[topics/neuroevolution|neuroevolution]] — MLX neuroevolution is the training path. v1 (91-input, 2-hop) and v2 (181-input, 3-hop) coexist; `scripts/train.py` runs the loop.
- [[topics/showcase-demo|showcase-demo]] — `?demo=1` evolution-arc demo: five 5s scenes, canned champions in `public/champions/`, hot-area intro + "AI WARS" title card.
- [[topics/v2-training-runs|v2-training-runs]] — record of v2 PPO training runs worth keeping. Cross-run lessons plus per-run config + observations. Current frontier: small radius-5 arenas with 20% dead cells, single-tick AI decisions, and tick-by-tick v2 replays (`v2-r5-tick1-001`).
- [[topics/v2-rules-one-pager|v2-rules-one-pager]] — compact reference for flux v2 rules, state, tick/capture/action semantics, reward intuition, and the current node/edge vision diagram.
- [[topics/v2-edge-voting-policy|v2-edge-voting-policy]] — spec for an edge-centric local-flow representation: derived edge types, distributed edge votes, multi-signal flow channels, and pulse-preserving hold/release learning.

Decisions:

- [[decisions/continuous-flow-model|continuous-flow-model]] — why strength is a continuous scalar with rate-based flows.
- [[decisions/pure-step-function|pure-step-function]] — game logic is a pure `step(state, dt) → state`, shared by browser and headless sim.
- [[decisions/one-flow-per-edge|one-flow-per-edge]] — at most one flow per undirected edge; reverse means flip.
- [[decisions/hex-grid-default|hex-grid-default]] — default board is a ~1000-cell hex grid with distance-2 connectivity; renderer uses instancing.
- [[decisions/attack-bonus|attack-bonus]] — non-friendly destinations get a `1 + ATTACK_BONUS` multiplier so attacks deal more damage than they cost.
- [[decisions/ai-zoo|ai-zoo]] — six hand-written heuristics registered under `src/ai/`; tournament shows four converge to the same play.
- [[decisions/multi-player-free-for-all|multi-player-free-for-all]] — default is 2–12 AI seats on the hex perimeter; spectator mode at 5× speed.
- [[decisions/stasis-detection|stasis-detection]] — variance-window detector that ends stalemated spectator sessions with a `STASIS` banner; suppressed in 1v1 endgame and cleanup phases.
- [[decisions/webgpu-evolution|webgpu-evolution]] — fixed-topology neuroevolution on WebGPU compute. Population eval on GPU, JS evolution loop, champion plays as `evolved` in the spectator zoo. WGSL step ≡ JS step parity invariant. Coexists with the MLX training path but is no longer where champions are trained.
- [[decisions/python-port|python-port]] — `python/` reimplements `step` and the NN forward pass for offline training. NumPy module is the bit-exact JS-parity reference; MLX is the compute backend for the evolution loop (tolerance-based parity, `float32`).
- [[decisions/replay-rendering|replay-rendering]] — Python is the lab, the web is the replay player, `.flxr` is the contract. Browser default mode is "watch replays."
- [[decisions/v2-vision|v2-vision]] — 3-hop receptive field experiment (181 input, 36 neighbors). v1 stays intact; v2 trains in parallel.
- [[decisions/v3-gnn|v3-gnn]] — 2-layer GCN policy on radius-9 boards. NEAT-evolved variant proved weak (credit assignment was the bottleneck, not architecture).
- [[decisions/ppo-gnn|ppo-gnn]] — PPO + v3's GCN backbone + value head, trained via MLX autograd. Self-play 12-seat FFA, dense per-AI-tick cell-delta reward. Currently the active training path.
- [[decisions/regen-flow-rules|regen-flow-rules]] — second game ruleset. Outflow has no health cost; sender forfeits regen. Damage is symmetric. Linear regen scaling with strength. Capture is deterministic strength=1. Existing transfer-flow ruleset stays as the default; replays carry `ruleset` tag.
- [[decisions/v2-edge-pressure-state|v2-edge-pressure-state]] — v2 makes directed half-edges first-class state. Set an outflow once; it persists. Loops, multi-hop transport, and maxed-cell fanout all fall out of one fill-then-overflow rule.
- [[decisions/v2-set-clear-actions|v2-set-clear-actions]] — v2's 13-action space: Set k=0..5, Clear k=0..5, No-op. Idempotent, state-independent — the network doesn't need to invert "what does toggle do here?"
- [[decisions/v2-three-term-reward|v2-three-term-reward]] — action-conditioned v2 reward stack: power/damage/capture, attributed waste, optional strict transit credit, kill pressure, time, and terminal bonus. Engagement / output-boost coefs from v1 are gone — persistence makes activity measures meaningless.
- [[decisions/v2-trainer-displayer|v2-trainer-displayer]] — v2 web UI is a stripped trainer-displayer at `/index-v2.html` (Vite). Plays back `.flxr` v2 replays from `/v2/replays/`. No in-browser game logic; v1 page unchanged.

Questions:

- [[questions/open|open]] — unresolved questions.

Process:

- [[todo|todo]] — active threads scratchpad (pre-issues).
