---
title: flux Wiki Index
kind: index
first_seen: bootstrap
last_updated: workspace
status: active
---

## Route map

Start at [[entities/flux|flux]] for the project overview (v1 and v2 in one
page). If you only care about the current frontier, jump straight to
[[topics/v2-rules-one-pager|v2-rules-one-pager]] +
[[topics/v2-overnight-research|v2-overnight-research]].

Core entities:

- [[entities/flux|flux]] — the project: v1 (frozen at saturation, deployed)
  and v2 (current frontier).

### Topics

Context:

- [[topics/galcon-like|galcon-like]] — the genre and its lineage.
- [[topics/neuroevolution|neuroevolution]] — neuroevolution arcs across v1
  (MLX champions at saturation) and v2 (PPO + GCN, plus algorithmic
  solvers below). WebGPU tier is a side-quest.
- [[topics/showcase-demo|showcase-demo]] — v1 `/?demo=1` evolution-arc
  demo: five 5s scenes, canned champions in `public/champions/`, hot-area
  intro + "AI WARS" title card.

v2 reference:

- [[topics/v2-rules-one-pager|v2-rules-one-pager]] — compact reference for
  v2 rules, state, tick/capture/action semantics, reward intuition, and
  the current node/edge vision diagram.
- [[v2-prd|v2-prd]] — pinned design doc (state shape, action space,
  capture semantics, delivery shape).
- [[topics/v2-viewer|v2-viewer]] — feature catalog for `/index-v2.html`:
  clickable newest-first recent-runs strip (all 50 indexed runs),
  transport bar, auto-cadence, mixed-radius rebuild, FLXR v2 format.

v2 research (freshest first):

- [[topics/v2-temporal-strategy|v2-temporal-strategy]] —
  ML-scientist synthesis (2026-05-15). Throttle (pressure/waste cliff),
  target dithering (multi-enemy cycling), and the strategic vocabulary
  (migrate/spread/flank/chase/corner) are one options-framework problem.
  Architecture is a slow manager over `(target, throttle, intent)` with a
  goal-conditioned worker. **Validation (same day):**
  `lightning_sum_throttled` dominates both bfs (7/7 coherent, p≈0.016)
  and vanilla `lightning_sum` (9/9, p≈0.004) at R=25 P=12 40%-dead.
  Sits above the overnight-research ranking below.
- [[topics/v2-grand-research-plan|v2-grand-research-plan]] —
  grant-scale plan for solving v2 without compute theater: vectorized
  measurement lab, solver factory, solver distillation, residual policies,
  temporal manager, then league scaling.
- [[topics/v2-vectorized|Pete / v2-vectorized]] — **the current hot path.**
  Numba-JIT'd solver pipeline, fluid `EDGE_ALPHA` momentum knob,
  warm-start Bellman, batched per-AI-tick solver, JIT'd board-setup
  BFS, FLXR v3 gzip-compressed replays. R=30 6000-tick game runs in
  1.3s (~24× pre-vec); R=100 in 13.9s (~36×). "Run that through Pete"
  means use this vectorized generator / trainer / solver lab path. One
  open item: the matched-pair rankings need a rerun under the new code
  before the v2-overnight-research rankings get refreshed.
- [[topics/v2-ml-gameplay-opportunities|v2-ml-gameplay-opportunities]] —
  ML-scientist synthesis of where learning can most plausibly improve v2:
  solver distillation, residual policies, edge-aware auxiliary learning, and
  matched-pair preference data.
- [[topics/v2-overnight-research|v2-overnight-research]] — autonomous
  overnight matched-pair tournaments (2026-05-15). **Current ranking:
  `wave_long` > `sum` > `bfs` ≈ `max` >> `attn` >> `pulse`/`pulse_stagger`.**
  Methodology section explains the 6pp seat-bias noise floor that
  invalidated several earlier "wins."
- [[topics/v2-algorithmic-solvers|v2-algorithmic-solvers]] — BFS and the
  Lightning solver family. Registered seat names in
  `python/scripts/run_v2_solver.py`.
- [[topics/v2-edge-loop-emergence|v2-edge-loop-emergence]] — the
  diffusion-mode design story (sum / sum_pw / loop / attn) and the PPO
  attempt to replace `lightning_attn` with a learned head. Mechanism
  remains useful; its internal "final ranking" is **superseded** by the
  overnight matched-pair results above.
- [[topics/v2-training-runs|v2-training-runs]] — record of v2 PPO training
  runs worth keeping. Cross-run lessons plus per-run config + observations.
- [[topics/v2-edge-voting-policy|v2-edge-voting-policy]] — *proposed*
  edge-centric local-flow representation. Partially landed
  (`python/flux_v2/edge_features.py`, `edge_flow.py`, `--model edge` in
  `train_v2.py`); a full PPO win against `sum` has not yet shipped.

### Decisions

v2 (current track):

- [[decisions/v2-edge-pressure-state|v2-edge-pressure-state]] — v2 makes
  directed half-edges first-class state. Set an outflow once; it
  persists. Loops, multi-hop transport, and maxed-cell fanout all fall
  out of one fill-then-overflow rule.
- [[decisions/v2-set-clear-actions|v2-set-clear-actions]] — v2's 13-action
  space: Set k=0..5, Clear k=0..5, No-op. Idempotent, state-independent.
- [[decisions/v2-three-term-reward|v2-three-term-reward]] — action-
  conditioned v2 reward stack: power/damage/capture, attributed waste,
  optional strict transit credit, kill pressure, time, terminal bonus.
- [[decisions/v2-board-connectivity|v2-board-connectivity]] — every non-
  DEAD cell must reach every other non-DEAD cell; every seat must reach
  every other seat; max seat-pair distance ≤ 4·R. Enforced via
  `random_seat_and_dead`, `seats_mutually_reachable`, and
  `carve_seat_connectors`.
- [[decisions/v2-trainer-displayer|v2-trainer-displayer]] — v2 web UI is a
  stripped trainer-displayer at `/index-v2.html`. Plays back `.flxr` v2
  replays from `/v2/replays/`. No in-browser game logic.
- [[decisions/ppo-gnn|ppo-gnn]] — PPO + v3's GCN backbone + value head,
  trained via MLX autograd. Currently the active training path.

Cross-track:

- [[decisions/regen-flow-rules|regen-flow-rules]] — second game ruleset
  (symmetric damage, linear regen, capture surplus). Replays carry a
  `ruleset` tag so the two worlds don't bleed into each other.
- [[decisions/replay-rendering|replay-rendering]] — Python is the lab,
  the web is the replay player, `.flxr` is the contract. Browser default
  mode is "watch replays."
- [[decisions/python-port|python-port]] — `python/` reimplements `step`
  and the NN forward pass for offline training. NumPy module is the bit-
  exact JS-parity reference; MLX is the compute backend (tolerance-based
  parity, `float32`).
- [[decisions/pure-step-function|pure-step-function]] — game logic is a
  pure `step(state, …) → state` in both v1 and v2.

v1 (frozen at saturation, still deployed):

- [[decisions/continuous-flow-model|continuous-flow-model]] — why v1
  strength is a continuous scalar with rate-based flows.
- [[decisions/one-flow-per-edge|one-flow-per-edge]] — v1 invariant: at
  most one flow per undirected edge; reverse means flip.
- [[decisions/hex-grid-default|hex-grid-default]] — default board is a
  ~1000-cell hex grid with distance-2 connectivity; renderer uses
  instancing.
- [[decisions/attack-bonus|attack-bonus]] — non-friendly destinations get
  a `1 + ATTACK_BONUS` multiplier so v1 attacks deal more damage than
  they cost.
- [[decisions/ai-zoo|ai-zoo]] — six hand-written heuristics registered
  under `src/ai/`; tournament shows four converge to the same play.
- [[decisions/multi-player-free-for-all|multi-player-free-for-all]] —
  v1 default is 2–12 AI seats on the hex perimeter; spectator at 5×.
- [[decisions/stasis-detection|stasis-detection]] — variance-window
  detector that ends stalemated v1 spectator sessions with a `STASIS`
  banner; suppressed in 1v1 endgame and cleanup phases.

Earlier experiments (history, not the frontier):

- [[decisions/webgpu-evolution|webgpu-evolution]] — fixed-topology
  neuroevolution on WebGPU compute. Champion still plays as `evolved` in
  the v1 zoo, but champions now come from Python/MLX, not in-browser
  evolution.
- [[decisions/v2-vision|v2-vision]] — 3-hop receptive field experiment
  (181 input, 36 neighbors). Predates v2 edge-pressure; useful as
  motivation but the v2 pressure-state direction superseded the input-
  vector approach.
- [[decisions/v3-gnn|v3-gnn]] — 2-layer GCN on radius-9 boards. NEAT-
  evolved variant proved weak (credit assignment was the bottleneck, not
  architecture). The GCN survived in [[decisions/ppo-gnn|ppo-gnn]].

Retired (kept as history):

- [[decisions/inbound-bonus|inbound-bonus]] — per-tick regen bonus
  proportional to active inbound friendly flow count. Retired 2026-05-10.
- [[decisions/loop-bonus|loop-bonus]] — friendly circulation bonus.
  Retired 2026-05-10 in favour of [[decisions/attack-bonus|attack-bonus]].

### Questions

- [[questions/open|open]] — unresolved questions across both tracks.

### Process

- [[todo|todo]] — active threads scratchpad (pre-issues).
- [[log|log]] — chronological wiki update log.
