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

- **Tier 1 — fixed topology, evolved weights only.** *Working — evolved champions sometimes win against the hand-written zoo (qualified validation, 2026-05-10).* Genome = `Float32Array` of weights. Forward pass per cell = two matmuls. Already powerful for this game space.
- **Tier 2 — proper NEAT.** ~800–1500 LOC. Innovation numbers, speciation, structural mutations, compatibility distance. Genuinely more interesting evolution.
- **Tier 3 — rtNEAT.** Time-based selection (oldest most likely to be replaced) + real-time speciation. True NERO-style spectator evolution.
- **Tier 4 — WebGPU compute.** *In progress.* Forward pass and `step` both run as compute shaders; whole population × all games per generation in parallel. See [[../decisions/webgpu-evolution|webgpu-evolution]]. Tiers 1 and 4 are merged here — Tier 4 is what `src/gpu/` shipped first because the browser is the deployment target.
- **Tier 5 — offline MLX training.** *Parity foundation landed; MLX evolution loop is the next thread.* `python/` reimplements `step` + NN forward in NumPy with bit-exact parity against JS — that's the algorithm-correctness anchor and it stays. The evolution loop itself is being built directly on MLX (Apple Silicon GPU acceleration, no NumPy-evolution interim). Champions serialize to the existing `public/champions/` JSON format for the browser to load. See [[../decisions/python-port|python-port]]. MLX runs `float32` so bit-exact JS parity isn't an MLX invariant — tolerance-based parity against the NumPy reference is, and the shared JSON format is what keeps the offline-train / online-deploy bridge honest.

## Python bridge (Tier 5 status)

What's in `python/` today:

- `flux.state` / `flux.graph` / `flux.step` / `flux.rng` — exact port of `src/game/` + `src/ai/rng.ts`.
- `flux.genome` — `nn_infer_cell` (Float32Array semantics replicated via NumPy `float32` storage + Python `float` arithmetic), `build_neighbor_table`, `random_genome`, `ai_think`.
- `python/tests/test_parity.py` ↔ `python/tests/dump_reference.ts` — deterministic 100-tick scenario, SHA-256 hashes every 10 ticks, both sides match.

Not yet ported (explicit follow-ups, NOT part of the parity foundation):

- **MLX evolution loop.** Same algorithm as `src/gpu/evolution.ts` (population, tournament selection, gaussian mutation, midpoint + end fitness with linger penalty), vectorized in MLX for big-board / many-genome scale. MLX is the compute backend from day one — there is no NumPy-evolution-loop step in between. The NumPy `flux` module stays as the algorithm reference the MLX side validates against (tolerance-based, since MLX is `float32`).
- Champion JSON load/save on the Python side (the JS format at `public/champions/*.json` is the target; that's the deployment bridge).
- HTTP or filesystem hot-reload of champions from Python into the browser.

The browser's WebGPU evolution ([[../decisions/webgpu-evolution|webgpu-evolution]]) remains the in-browser path. The MLX loop and the WebGPU loop coexist; they share the genome layout, the fitness shape, and the champion JSON format.

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
