---
title: v3 GNN Policy
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Choice

Run a third model class — **v3** — alongside v1 and v2. v3 swaps the flat per-cell MLP for a 2-layer **graph neural network** (GCN-style mean aggregation over the game's graph). Per-cell input drops from 91 (v1) / 181 (v2) flat features to **4 features × per-cell graph neighborhood** — message passing supplies the rest.

The evolution loop, the step kernel, the aggressive seat, replays, checkpointing, and wandb metrics are **unchanged**. Only the policy architecture changes. `--model v3` on `python/scripts/train.py` selects it; checkpoints land in `python/checkpoints/v3/latest.npz`, champions in `public/champions/v3/`.

v1 and v2 stay intact. v3 is a separate experiment.

## Architecture

Per-(game, seat, cell) input — 4 channels:

- `strength / MAX` — own strength
- `is_mine` — owner == seat
- `is_enemy` — owner is some other player
- `is_neutral_or_empty` — owner == -1

Two message-passing layers:

```
H1[c] = relu(W_self_1 · H0[c] + W_neigh_1 · mean(H0[n] for n in neighbors(c)) + b1)
H2[c] = relu(W_self_2 · H1[c] + W_neigh_2 · mean(H1[n] for n in neighbors(c)) + b2)
```

`neighbors(c)` is the game's distance-≤2 graph (18 cells, padded with -1 for boundary cells — same `NEIGHBOR_STRIDE` as v1).

Output head per cell:

```
out[c] = W_out · H2[c] + b_out      # → 19 logits
action[c] = argmax(out[c])           # 0..17 = flow target, 18 = noop
```

Sizes: hidden=32, output=19. **2995 weights per genome** — actually *smaller* than v1 (3571) and well under v2 (6451). The architectural power comes from message passing, not parameter count.

## Why

The v1/v2 MLPs see a flat list of neighbor features. To exploit a chain or loop dynamic (3 friendly cells in a row delivering ~3× damage via the regen + attack-bonus mechanic — see [[attack-bonus]] and the choo-choo entry in [[../log]]), the network has to discover from random weight mutations the pattern "if my neighbor is friendly and pushing forward, push toward them." That's a needle-in-haystack search.

A 2-layer GCN propagates information across the same edges that flows travel over. After one round of MP, each cell knows its neighbors' state. After two rounds, each cell's representation is informed by everything within 2 hops, *with structure preserved* (vs. v1/v2 which sees the same cells as a flat unordered-pair-of-rings). Chains are 2-hop coordination patterns; GNNs are how you parameterize 2-hop coordination patterns.

## Why keep v1 and v2

Same rationale as v2's existence: separate experiments, independent checkpoints, distinct artifact dirs. We can compare:

- v1 → v2: does wider vision accelerate evolution? (Open question — v2 caught v1's plateau in ~6.5% of v1's gen budget but unverified at scale.)
- v1/v2 → v3: does graph-structured policy enable strategy that flat MLPs can't find?

If v3 obviously dominates at low gen count → architecture mattered. If v3 plateaus at the same level → architecture isn't the bottleneck (credit assignment probably is, see [[../todo|ppo+gnn]] thread).

## Cost

- Per-tick MLX work roughly **2×** v1's (two MP layers instead of one matmul). Smoke at G=1 measured 1.03s/game vs v1's ~0.65s. Still well under the 2s/game tolerance.
- Memory: each MP layer materializes a `(G, S, N, STRIDE, F)` gathered tensor of `1 × 12 × 1027 × 18 × 32 × 4 bytes ≈ 28 MB` at hidden=32. Fine at G=1, would be tight at G ≥ 16 (loop-and-accumulate fallback is one branch away if needed).
- No new genome size hits — 2995 weights is lighter than v1.

## What v3 does NOT change

- The fitness signal (end_cells + early·mid - linger·opp). Still sparse, still applied at decision_tick.
- The evolution algorithm. Still NEAT-flavored mutate-bottom-k-tournament-select. ([[../topics/neuroevolution|topics/neuroevolution]] talks about CMA-ES and PPO as future swaps that pair well with the GNN.)
- The aggressive seat. Still hand-coded, still overlays seat 0.
- Replay format. Still `.flxr` with `model: "v3"` in metadata.

## File map

- `python/flux/mlx_genome_v3.py` — constants + weight offsets.
- `python/flux/mlx_batch.py` — added `nn_infer_batched_v3`, `build_flows_batched_v3`, `_graph_neighbor_aggregate`, `_build_features_v3`.
- `python/flux/game_loop.py` — `GameConfig.model_kind = "v3"` branch + `weights_per_genome("v3")`.
- `python/scripts/train.py` — `--model v3` choice; default paths `python/checkpoints/v3/latest.npz` and `public/champions/v3/`.

## Status

Launched 2026-05-11 from a fresh population. First batch logged to wandb run **v3-gnn-fresh**. Watching for whether early-gen fitness climbs faster than v1's first 100 generations did — and, qualitatively, whether replays show recognizable chain/coordination behavior.

## Result (2026-05-11)

Trained ~500 generations on radius 9, G=1, every-batch replay writes. Wandb runs: `v3-gnn-r9`, `v3-r9-stride10`, `v3-r9-everybatch` (all with aggressive at seat 0), and `v3-r9-no-aggressive` (just launched — seat 0 is another NN; observations pending).

- `all_time_best_fitness` stayed negative the entire run (~-2168 at gen 500).
- Aggressive consistently won the vast majority of games.
- Qualitative read: **interesting but super weak.** The 2-hop GNN receptive field didn't visibly improve strategy over v1/v2.

Architecture is fine end-to-end — `nn_infer_batched_v3`, message passing, `build_flows_batched_v3` all behave. The diagnosis is **not** that GNN is wrong. The bottleneck is **credit assignment**: NEAT-style fitness (one scalar per game) gives the network no information about which decisions were good. The whole genome's weights get a noisy bump from one number — fine for a 3.5k-param flat MLP, far too lossy for a 3k-param GNN whose payoff comes from learning specific multi-hop coordination patterns. Pattern across v1 (plateaued ~1540 after 5000+ gens), v2 (caught it in ~100 gens, same plateau), and v3 (-2168 after 500 gens) says architecture is not the gating factor.

Forward: see [[ppo-gnn]] — keep this GNN, swap NEAT for PPO so per-tick decisions get real gradient signal via value baselines and GAE.
