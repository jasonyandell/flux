---
title: Neuroevolution
kind: topic
first_seen: workspace
last_updated: workspace
status: active
---

## Concept

Evolve neural-network controllers that play flux, watching them improve in real time. Each seat in the multi-player [[../entities/flux|flux]] board hosts one controller drawn from a population; after each game (or continuously, rtNEAT-style), the weakest controllers are replaced by mutated copies of the strongest. The [[../decisions/ai-zoo|ai-zoo]] failure mode — four "weakest-local-neighbor" heuristics converging to the same play — is the prompt: heuristics this shallow can be discovered, not hand-written.

## Lineage

- **NEAT** (Stanley & Miikkulainen, 2002, UT Austin NNRG) — evolves weights AND topology. Innovation numbers track structural mutations; speciation prevents premature convergence.
- **rtNEAT** (Stanley, Bryant, & Miikkulainen, 2005) — real-time variant. Continuously replaces the single weakest individual instead of batch generations. Built for spectator-mode evolution.
- **NERO** (*Neuro-Evolving Robotic Operatives*) — academic game where rtNEAT-driven soldiers evolve while you watch. Same shape as flux's spectator mode.
- **OpenNERO** — open-source platform around rtNEAT, maintained by Risto Miikkulainen's lab at UT Austin.

For flux, **rtNEAT is the right algorithmic fit** because the project is already spectator-shaped: long-running sessions where you watch many concurrent games. No need for clean generational boundaries.

## Controller shape (proposed)

Each cell makes its own decision using a shared per-genome network applied locally — same weights, run independently per cell. Scales to the 1000-cell board without re-architecting the game.

Input vector per cell (~91 features):

- Own strength (1 scalar, normalized by `MAX_STRENGTH`).
- For each of the ~18 reachable neighbors (distance ≤ 2 on the hex grid): the neighbor's strength + a 4-way one-hot for relative ownership (self / friend / enemy / neutral). 5 features × 18 = 90.

Output per cell (~19 outputs):

- A score per reachable neighbor (18 scores) plus a "do nothing" score.
- Argmax (or temperature-sampled) selects the flow target; if "do nothing" wins, no flow this tick.

Initial architecture: 91 → 32 (tanh) → 19. ~3.5k weights per genome. Small enough to run dozens of genomes per WebGPU dispatch.

## Evolution loop (proposed)

1. Population size: 12–50 genomes.
2. Each game: assign N genomes (N = `numPlayers`) to seats randomly. Run to completion or tick cap.
3. Fitness: `end_cells + earlyWeight * mid_cells - lingerPenalty * opponent_cells_at_end`. Two samples (midpoint and end) give a time-integrated signal; the linger penalty punishes leaving holdouts alive. Current defaults: `earlyWeight = 0.5`, `lingerPenalty = 2.0`.
4. After each game, replace the bottom k (k = 1 for rtNEAT-style continuous, k = 1–3 for simpler batch) with mutated copies of top performers.
5. Mutations: gaussian noise on weights. Structural mutations (add node, add connection) are NEAT-proper but optional for v1.

## Effort tiers

This was the v1 era's framing. For the v2 frontier, neuroevolution is no
longer the active path — see "Where this fits now" below.

- **Tier 1 — fixed topology, evolved weights only.** *Working.* Genome = `Float32Array` of weights. Forward pass per cell = two matmuls.
- **Tier 2 — proper NEAT.** ~800–1500 LOC. Innovation numbers, speciation, structural mutations, compatibility distance. Not on the active path.
- **Tier 3 — rtNEAT.** Time-based selection + real-time speciation. Not on the active path; batch generations win on throughput with MLX vectorization.
- **Tier 4 — WebGPU compute.** *Shipped, side-quest.* `src/gpu/` runs population eval as compute shaders in-browser. See [[../decisions/webgpu-evolution|webgpu-evolution]]. Champions on disk now come from Python.
- **Tier 5 — MLX neuroevolution (v1, frozen).** *Shipped end-to-end, then saturated.* `python/flux/mlx_*` runs the whole evolution loop on Metal via MLX. Two model widths landed: v1 (91-input, 2-hop neighborhood, 32 hidden, 19 output, 3571 weights) and v2-vision (181-input, 3-hop neighborhood, 32 hidden, 19 output, 6451 weights — see [[../decisions/v2-vision|v2-vision]]). Win-based termination per game, 10k tick cap. Champions serialize to `public/champions/` and `public/champions/v2/`. Replays serialize to `public/replays/*.flxr`. **At saturation as of 2026-05-11** (v1 gen 5372 / fit 1540, v2-vision gen 347 / fit 1521). No longer where new design happens.
- **Tier 6 — v2 PPO on persistent-edge sim (the current training path).** *Active.* Different sim ([[../decisions/v2-edge-pressure-state|v2-edge-pressure-state]]), different action space ([[../decisions/v2-set-clear-actions|v2-set-clear-actions]]), different reward shape ([[../decisions/v2-three-term-reward|v2-three-term-reward]]). PPO + GCN policy in `python/flux_v2/ppo.py` driven by `python/scripts/train_v2.py`. See [[../decisions/ppo-gnn|ppo-gnn]] and [[../topics/v2-training-runs|v2-training-runs]]. The proposed edge-voting head ([[../topics/v2-edge-voting-policy|v2-edge-voting-policy]]) is the next move.

## Where this fits now

Tier 5 is the high-water mark for neuroevolution proper in this repo and
remains the deployed `evolved` seat. Tier 6 (v2 PPO) is what's actively
learning. The current open question is whether a learned policy on v2 can
beat the best hand-coded Lightning solver
([[../topics/v2-overnight-research|v2-overnight-research]]) — under big-bag
rules as of 2026-05-15 it has not yet, and the attn-headed PPO variant was
abandoned.

## Python pipeline (Tier 5) — files

Game core + NN reference (NumPy, bit-exact JS parity):

- `python/flux/state.py`, `graph.py`, `step.py`, `rng.py`, `genome.py` — algorithm reference. See [[../decisions/python-port|python-port]] for the parity invariant.

MLX evolution path (Metal-accelerated, `float32`):

- `python/flux/mlx_step.py` — single-game MLX `step` + `apply_action`.
- `python/flux/mlx_batch.py` — batched (G games × S seats × N cells) MLX step, NN forward, and the **vectorized AI tick**. `build_flows_batched` / `build_flows_batched_v2` build dense (G, N) flow tensors on GPU per AI tick — one batched NN forward → per-cell owner-action argmax → aggressive overlay → flow tensor, no per-game Python flow-reconcile loop. This is the main perf win over the JS-style reconcile.
- `python/flux/mlx_genome.py` — v1 layout constants (`IN=91`, `HID=32`, `OUT=19`).
- `python/flux/mlx_genome_v2.py` — v2 layout constants (`IN=181`, `HID=32`, `OUT=19`).
- `python/flux/vision.py` — 3-hop neighbor table for v2. `STRIDE_V2 = 36`. v1 still uses the 18-neighbor distance-2 table from `genome.py`.
- `python/flux/game_loop.py` — `play_batch_games`. Runs G games in parallel under MLX, terminates a game when alive ≤ 1, hard-caps at 10k ticks.
- `python/flux/evolve_mlx.py` — `run_one_batch`, selection, checkpoint, champion JSON. Tournament selection + gaussian mutation, same fitness shape as `src/gpu/evolution.ts`.
- `python/flux/replay.py` — `.flxr` writer (binary, header + sampled-frame body). See [[../decisions/replay-rendering|replay-rendering]].
- `python/scripts/train.py` — CLI entry point. Auto-resumes from `python/checkpoints/{latest.npz|v2/latest.npz}` unless `--fresh`. Flags: `--model {v1,v2}`, `--games-per-batch`, `--pop`, `--ticks`, `--ai-period-ticks`, `--checkpoint`, `--champion-dir`, `--fresh`, `--aggressive-seat`.

The **aggressive seat** is hand-coded in MLX (vectorized argmin over non-friendly neighbor strength) and serves as the narrative anchor: every batch contains one or more aggressive opponents the population must beat.

### Persistence

- Checkpoint: `python/checkpoints/latest.npz` (v1) / `python/checkpoints/v2/latest.npz` (v2). Auto-resume on `train.py` startup unless `--fresh`.
- Champion JSON for new all-time bests: `public/champions/*.json` (v1, browser default) and `public/champions/v2/*.json` (v2).
- Replays: `public/replays/*.flxr`, pruned to 50 entries in `public/replays/index.json`.

### Status (2026-05-11)

- v1 — gen 5372, all-time-best fitness 1540.50. Beats hand-coded aggressive consistently.
- v2 — gen 347 (started fresh today), all-time-best 1521.50. Caught v1's fitness in ~6.5% of v1's generation budget. Wider receptive field reaches reinforcement-distance cells; appears to massively accelerate evolution but unverified at scale.

### G knob (games per batch)

`--games-per-batch G` controls parallel game count per evolution step. Bench at radius 18, 12 seats, 10k tick cap, ai_period 5:

| G | per-batch | gens/min | samples/genome/batch |
|---|-----------|----------|---------------------|
| 1 | 0.65s (loop) / 2.6s (train.py with writes) | 22 (after sync+async wins) | 0.46 |
| 2 | 1.0s | 60 | 0.92 |
| 4 | 1.4s | 43 | 1.83 |
| 8 | 2.3s | 26 | 3.67 |
| 24 | 11.0s | 5 | ~11 |
| 128 | 113.6s | 0.5 | ~59 |

G=4 was the pre-optimization sweet spot. G=1 is the current choice — max generation rate, accept noisier fitness signal. Memory bandwidth (features tensor scales G×S×N×91) makes G > 24 throughput-counterproductive.

The browser's WebGPU evolution ([[../decisions/webgpu-evolution|webgpu-evolution]]) coexists; the two share genome layout, fitness shape, and champion JSON format, but the production training path is MLX.

## Why it fits flux

- **Per-cell controllers preserve purity.** Each cell's forward pass takes local neighborhood — no globals, no DOM. Drops into `src/ai/` alongside the [[../decisions/ai-zoo|zoo]].
- **Spectator mode is already the shape.** Population members slot into existing seats; evolution is "after a game, swap some genomes."
- **The zoo provides the floor.** Heuristic AIs hit a structural stalemate. An evolved controller that beats them is genuinely doing something. Cheap baseline for "is this any good?"

## Open implementation questions

- **Local vs strategic input.** Per-cell controllers (above) scale easily but are myopic. Strategic-level (one network producing global priorities) is more interesting but needs state aggregation. Hybrid possible.
- **What is fitness?** Final territory is obvious; survival time is decent secondary. Could reward consistency across opponents (cross-validation against the zoo) or robustness across seeds.
- **PRNG threading.** Mutations need a seeded PRNG; existing `src/ai/rng.ts` mulberry32 works. Genomes reproducible from a seed for replay.
- **Genome storage.** In-memory `Float32Array` is fine for ephemeral runs. JSON-serialize to save "interesting" genomes for later reload.
- **Speed multiplier interaction.** `SPEED = 5` in `main.ts` lets games finish faster in wall-clock time, which helps evolution throughput. Higher speeds may be worth exposing once evolution is live.
