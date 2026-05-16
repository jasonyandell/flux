---
title: v2 vectorized
kind: topic
first_seen: 2026-05-15
last_updated: 2026-05-16
status: active
---

## Status

Landed on main 2026-05-16. The hot path is now Numba-JIT'd top to
bottom and parallel-friendly. **One open question remains**: the
matched-pair tournament rankings in
[[v2-overnight-research|v2-overnight-research]] were produced
against the pre-vectorize per-cell-loop solvers. The two semantic
deltas documented below (RNG draw schedule, ε-tie relay rule) are
individually small but the rankings live inside a 6 pp seat-bias
noise floor, so they need a rerun under the new code before the
official rankings get refreshed.

## What this is

A single-shot rewrite of the v2 hot path:

- **All 17 lightning modes + BFS go through one vectorized pipeline**
  in `python/flux_v2/solver_vec.py`. Each mode produces an `(N, K)`
  bool `desired` mask, then a shared picker decodes
  SET-missing > CLEAR-stale > NOOP across all owned cells in one
  numpy pass. The eight per-cell Python loops that used to live in
  `solver_lightning.py` and `solver.py` are gone.
- **`step.apply_actions` ported to `(N, K)` numpy**, modeled on
  `mlx_step.apply_actions_batched`. The 1200-trip `for c in range(N)`
  loop is gone.
- **FLXR v3 replay format**: JSON header + gzip-compressed dense
  per-frame encoding (owners + strengths + outflow bitmask + popcount
  pressure bytes). Strength / edge scale now self-describing in the
  header. Old `.flxr` v1/v2 files are unreadable by the new player —
  see the format section below.

## Headline numbers

6000-tick 6-seat all-`lightning_sum_long` fluid `EDGE_ALPHA=0.05`,
single process, M5 Max. Pre-vec column is measured for R=20/30 and
extrapolated for R≥60 (pre-vec board setup is O(N²) Python BFS, game
loop is ≈linear in N).

| R | N (cells) | new | pre-vec | speedup |
| --- | --- | --- | --- | --- |
| 20 | 1,261 | 0.7s | 9.7s | ~14× |
| 30 | 2,791 | 1.3s | 31.3s | ~24× |
| 60 | 10,981 | 5.1s | ~140s | ~27× |
| 80 | 19,441 | 8.8s | ~280s | ~32× |
| 100 | 30,301 | 13.9s | ~500s | ~36× |

The speedup *grows* with board size because the pre-vec
`_live_subgraph_connected` ran a pure-Python BFS over the live
subgraph once per dead-cell candidate during board setup — N
candidates × N-cell BFS = O(N²) Python ops. The JIT'd path is
linear in N for both game loop and board setup, so the gap widens.

### Parallel throughput

`--workers` parallelizes across games. R=100 6000-tick × 10 games,
`--workers 10` on M5 Max (12 P-cores): **15.3s wall, 1.53s per game
amortized**. Aggregate CPU usage was ~145s — full saturation. A
100-game tournament at R=100 fits in ~150s; at R=30 in ~13s.

### Replay file sizes (FLXR v3, gzip-compressed dense per-frame)

| Config | size |
| --- | --- |
| R=20 6000-tick stride-5 | 468 KB |
| R=20 6000-tick stride-25 | 109 KB |
| R=60 6000-tick stride-25 | 3.8 MB |
| R=100 6000-tick stride-25 | 9.2 MB |

All comfortably under Cloudflare free-tier asset limits.
Pre-vec FLXR v2 size at R=20 stride-5 was ~20 MB → v3 is 43× smaller
at the same fidelity, ~180× at stride-25.

### Build / test gates

141/141 Python tests pass. `tsc --noEmit` clean. `vite build` clean.

## Why the file size dropped 43-180×

The v2 format wrote `(owners + strengths + flow records)` per frame,
where each flow was `(src u16, dst u16, player u8, pressure_q u8) = 6
bytes`. For a typical R=20 frame with a few hundred active flows that
was ~3 KB/frame uncompressed and *not* compressed. The v3 stream is
dense per-frame (`owners + strengths + outflow bitmask + popcount
pressure bytes` ≈ same total bytes pre-compression) then gzipped end
to end. Game state is highly redundant frame-to-frame, so gzip wins
hard on the cell-major ordering. See
the format section below for the
layout details.

## Two intentional semantic deltas from the loop solvers

1. **RNG draw schedule changes.** The picker draws a per-cell rotation
   offset and picks the first qualifying slot after rotation. The old
   `_pick(missing, rng)` drew once per cell *inside* a Python loop.
   Different draw schedule → seed-replay bit-exactness is broken.
   Replays from older runs don't bit-replay; new replays are
   deterministic given a seed.

2. **Relay ε-tie rule is now globally consistent.** The new rule:
   *"friendly slots whose pot is within `fanout_eps` of the cell's
   max friendly-slot pot AND strictly above pot[c]."* The old loop
   tracked an incremental running-best variable; consecutive
   within-ε pairs that spanned > ε total could end up either in or
   out of the relay set depending on traversal order. The new rule
   is the obvious tight-cluster definition. At 100-game matched-pair
   sample sizes this almost certainly does not move the overnight
   rankings, but the test hasn't run yet.

## What is *not* yet verified (the headline open question)

The [[v2-overnight-research|v2-overnight-research]] page documents a
clean hierarchy under matched-pair analysis:

> `wave_long` > `sum` > `bfs` ≈ `max` >> `attn` >> `pulse`/`pulse_stagger`

That ranking was produced against the per-cell-loop solvers. **Until
we rerun the matched-pair test on the vectorized code, we don't know
whether the same ordering holds.** The two behavioral deltas above
are individually small, but the ranking lives inside a 6pp seat-bias
noise floor, so "small" doesn't guarantee "invisible."

The methodology to re-validate is already on
[[v2-overnight-research|v2-overnight-research]]:

- ≥100 games per cell, matched-pair (both seat orderings on the same
  random board), 10% dead at R=20.
- `wave_long vs sum`, `sum vs bfs`, `sum vs max`, `sum vs attn` are
  the four pairwise checks that produced the hierarchy.

If the new code reproduces those, we promote v2-vectorized from
provisional to the main track and update the wiki rankings to point
at it. If it doesn't, we either revert the relay-rule change to
match the path-dependent behavior or accept the new ordering and
update the rankings.

## Files touched

```
python/flux_v2/solver_vec.py        (new, ~600 LOC — the new pipeline)
python/flux_v2/solver.py            (shim → solver_vec.bfs_actions)
python/flux_v2/solver_lightning.py  (shim → solver_vec.lightning_actions)
python/flux_v2/step.py              (apply_actions vectorized)
python/flux_v2/replay.py            (FLXR v3 writer)
python/scripts/train_v2.py          (writer-path uses new Frame dataclass)
src_v2/replay/format.ts             (FLXR v3 reader, async via gzip stream)
src_v2/replay/player.ts             (awaits parseReplay)
src_v2/render/scene.ts              (no longer imports MAX_STRENGTH —
                                     the v2 const was wrong anyway,
                                     scale now read from header)
```

Net diff: 1,037 lines added, 1,266 deleted. Mostly removal — the
per-mode `_*_actions` Python functions in `solver_lightning.py`
collapse into shared `(N, K)` builders in `solver_vec.py`.

## Hidden bug fix surfaced during the rewrite

The v2 replay writer quantized strength against `MAX_STRENGTH = 1000`
on the Python side; the v2 reader dequantized against
`MAX_STRENGTH = 100` on the TS side. End-to-end the visualizer's
`scale = 0.45 + (strength / MAX_STRENGTH) * 1.0` clamped most cells
into a visually-acceptable but technically-wrong band. v3 carries the
scale in the JSON header so the writer and reader can't disagree.

## FLXR v3 wire format

Layout (little-endian):

```
magic        4 bytes   "FLXR"
version      u8        = 3
reserved     u8
header_len   u32       byte-length of the JSON metadata blob
header_json  bytes     UTF-8 JSON: radius, num_players, num_nodes,
                                   tick_stride, dt_per_tick_ms,
                                   num_frames, max_strength, max_edge,
                                   metadata
frames_gz    bytes     gzip-compressed frame stream (to EOF)
```

Frame stream (uncompressed; concatenated, fixed-size per game):

```
owners            N bytes int8     -2 dead, -1 neutral, 0..P-1 seat
strengths         N bytes uint8    quantized 0..255 over [0, max_strength]
outflow_bits      ceil(N*K/8)      bit i: cell (i // K), slot (i % K)
pressure_bytes    popcount bytes   uint8 quantized over [0, max_edge],
                                   same iteration order as outflow_bits
```

Geometry is *not* stored — the reader rebuilds it deterministically
from `(radius, num_players)` via `buildBoard` (TS) /
`make_board` (Python). The v2 board-connectivity invariant
(see [[../decisions/v2-board-connectivity|v2-board-connectivity]])
makes this safe.

Browser decode: `DecompressionStream('gzip')` is standard in modern
browsers (Safari ≥ 16.4, all evergreen). No third-party gunzip
needed. `parseReplay` is now `async`.

## Batched solver + JIT board setup (commits `4fb7e23`, `b0d19c7`)

Two more big wins after the warm-start landing.

**Batched solver pipeline.** When all P seats run `lightning_sum_long`,
the runner now dispatches one batched JIT call per AI tick instead of
P × 3 separate calls. Combines `compute_potential` + gradient-relay +
picker into a single function that runs all P seats back-to-back
without leaving JIT space. Also caches the per-cell `deg` precompute
(non-DEAD non-off-grid neighbor count) per game keyed by
`id(neighbors)`, since DEAD cells are static within a game.

Game-loop work: **0.42s → 0.26s (38% faster on the per-tick portion)**.

**JIT'd board setup.** Profile-driven: `_build_initial_state` took
**1.16s per game** at R=30 with 800 dead cells. Each dead-cell
candidate ran `_live_subgraph_connected` as a pure-Python BFS over
N=2791 cells, called ~N times → ~8M Python ops per board generation.
Three @njit BFS cores plus `_place_dead_cells_core` (fuses the
candidate-iteration loop with the inner BFS into one JIT call) brought
board setup to **0.14s — 8× faster**.

| Stage | R=30 6000-tick |
| --- | --- |
| pre-vec main | 31.3s |
| post-vec (numpy) | 23.5s |
| Numba JIT all solver paths | 3.8s |
| warm-start compute_potential | 2.5s |
| + batched pipeline | 2.4s |
| **+ JIT board setup** | **1.3s** |
| **total speedup vs pre-vec** | **24×** |

R=40 6000-tick: 45.5s post-vec → 2.2s = 20.7× total.

`_WARM_ITERS` also dropped from 6 to 4 — empirical sweet spot at
`EDGE_ALPHA=0.05` (3 visibly degrades game quality, 6 is no better
than 4 at this state-change rate).

Things that did NOT pay off and were reverted:
- `parallel=True` over the seat dimension in
  `_lightning_sum_batched_core`: P=6 seats × ~400 µs work is too
  little to overcome thread-sync overhead.
- `parallel=True` inside `_compute_potential_core` over cells:
  76,800 prange invocations × tiny per-iter work = 10× *slower*.

## Warm-start `compute_potential` (commit `88d8fd3`)

After all the JIT work the profile showed `compute_potential` at 77%
of remaining time — 32-iter Bellman value iteration, 6 seats per AI
tick, 400 AI ticks per 6000-game-tick run. The 32-iter cold start
does not actually converge tight at γ=0.94; the field at 16 iters is
still ~25 % off the 32-iter result by L2-norm. But the *fixed point
moves slowly* between AI ticks under fluid physics — only ~5 game
ticks elapse, and `EDGE_ALPHA=0.05` low-passes the state evolution.

Solution: cache last pot per (board id, seat, mode, gamma) and warm-
start from it. 6 iters of contraction from a close-to-truth init
reaches the new fixed point. Cold start (cache miss) still does the
full 32-iter solve.

| Config | pre-vec | full-JIT | + warm-start | total vs pre-vec |
| --- | --- | --- | --- | --- |
| R=20 6000-tick decisive | ~9.7s | 2.0s | **0.7s** | **~14×** |
| R=30 6000-tick stalemate | 31.3s | 3.8s | **2.5s** | **12.5×** |
| R=40 6000-tick stalemate | (45.5s post-vec) | 9.0s | **6.1s** | 7.5× post-vec |

The cache is keyed by `id(neighbors)` because `copy_state` preserves
neighbor identity within a game; each game maps to its own cache
entries with no risk of cross-game contamination.
`solver_vec.reset_potential_cache()` is exposed for explicit teardown
between games where structure changes (different radius / players).

## MLX experiment (commit `88d8fd3`)

A microbenchmark of `compute_potential` alone measured MLX at 1.79×
faster than Numba: 0.30 ms/call (Numba JIT) vs 0.17 ms/call
(`mx.compile`'d 32-iter Bellman on M5 Max GPU). Wired into the actual
solver hot path, MLX ran **slower** by ~1.5× because per-call
`numpy→mx.array` upload + per-seat `mx.eval` synchronization exceeds
the in-kernel savings — the benchmark had been reusing the same
intrinsic tensor across all calls, hiding the real upload cost.

To win with MLX would require keeping the whole solver pipeline
(intrinsic, attack/relay masks, picker) in MLX and syncing once per
AI tick across all 6 seats — a sizable rewrite. Left for future
work; `FLUX_V2_BACKEND=numba` is the default. `FLUX_V2_BACKEND=mlx`
exposes the hooks for that future direction.

## Numba JIT (commit `82e80b4`)

Three @njit-cached cores added on top of the vectorized pipeline:

- `step._tick_core` — explicit per-cell physics. Replaces the inner
  K-loop + numpy chain in `tick()`. 28 µs per call at R=20 (was
  ~500 µs in numpy).
- `solver_vec._compute_potential_core` — JIT'd Bellman value iteration
  (modes max / sum / sum_pw). 32-iter loop now runs as native code
  instead of 32 small numpy reductions.
- `solver_vec._picker_core` — explicit slot-walk in rotation order.
  Replaces an `np.take_along_axis` + `argmax` chain that profiled
  surprisingly heavy at this array size.

Wallclock, 6000-tick 6-seat all `lightning_sum_long` under fluid
`EDGE_ALPHA=0.05`, single process, M5 Max:

| Board | pre-vec | post-vec | JIT | total speedup vs pre-vec |
| --- | --- | --- | --- | --- |
| R=20 (3539-tick decisive) | ~9.7s | 6.2s | **2.0s** | ~4.8× |
| R=30 6000-tick | 31.3s | 23.5s | **6.4s** | **4.9×** |
| R=40 6000-tick | — | 45.5s | **15.6s** | 2.9× vs post-vec |

First call triggers a 1-2s JIT compile that's then disk-cached, so
subsequent runs pay zero warmup. 141/141 tests still pass.

**Why Numba mattered here:** pure-numpy on (1261,) / (1261, 6) arrays
hits a small-array dispatch floor — each numpy op pays a fixed
Python-side cost regardless of the array size, and the field
iteration in `compute_potential` made ~30 such calls per tick per
seat. JIT-compiling the inner loop as a single native function
collapses 30 dispatches into one, which is where the 5× lives.

Cost: +2 deps (numba 0.65.1, llvmlite 0.47.0). Acceptable for a
research-mode project where the lab loop is the bottleneck.

## Fluid-mode pilot (EDGE_ALPHA, commit `b9b5b6b`)

A separate one-line physics knob layered on top: edge pressure is no
longer recomputed-from-scratch each tick, it relaxes toward its
source-overflow target.

```python
edge_pressure_next = (1 - alpha) * edge_pressure + alpha * target
```

`EDGE_ALPHA=1.0` (default) is bit-exact original v2 — 141 tests pass
unchanged. `EDGE_ALPHA<1` turns pressure into a state variable with
momentum: ~1/alpha ticks to build up after a valve opens, same to
bleed off. The "fluid" framing — pressure as a slow-moving inertial
field rather than an instantaneous snapshot of the source's spill
this tick. CLI: `--edge-alpha 0.05` on `run_v2_solver.py`. Replay
metadata's `ruleset` becomes `"v2-fluid-0.05"` so future matched-pair
runs can group by rules.

Pilot smoke result, `lightning_sum_long` all seats, seed 42,
parallel-4 / 2:

| | alpha=1.0 (snap, original) | alpha=0.05 (fluid) |
| --- | --- | --- |
| R=20 6000-tick × 4 stalemates | 2/4 | 1/4 |
| R=20 decisive-game length | 1916 – 3133 ticks | 3499 – 3754 ticks |
| R=30 6000-tick × 2 stalemates | 2/2 | 1/2 (other at 0.98 dom) |
| Per-tick compute cost | baseline | unchanged |

The stalemate rate roughly halves on both board sizes; decisive
games take longer but bunch up tighter in length. The intuition: at
alpha=1.0 the policy can twitch a valve off to instantly cut
incoming pressure, so a clever defender disrupts attackers without
building any counter-flow. At alpha=0.05 pressure persists for ~20
ticks regardless of valve state, so disrupting an attacker requires
*sustained* counter-pressure, not a quick toggle. Flow becomes
load-bearing infrastructure instead of an output-of-the-moment.

**This is still a pilot** — no matched-pair tournament under fluid
rules has been run. Two things to validate before promoting:

1. **Do the wave/long modes still dominate?** `wave_long` and
   `sum_long` were tuned for a world without momentum. Their
   advantage might shrink if the physics already integrates time.
   Could also widen if their long-field view aligns better with
   true equilibrium pressure.
2. **Does the bigger perf win materialize?** With pressure as a
   meaningful inertial state, `compute_potential`'s 32-iter Bellman
   solve has redundancy: the live `edge_pressure` field already
   *is* a noisy estimate of the steady-state potential. Solvers
   that read the live field directly should be 5-10× faster and
   roughly as strategically informed. Untested.

## Related

- [[v2-overnight-research|v2-overnight-research]] — the matched-pair
  rankings produced against the pre-vectorize code. *Not yet
  re-confirmed.*
- [[v2-algorithmic-solvers|v2-algorithmic-solvers]] — the solver
  family this rewrite preserves (no modes added or removed).
- [[v2-viewer|v2-viewer]] — the browser displayer, now reading v3.
- [[../decisions/v2-edge-pressure-state|v2-edge-pressure-state]] —
  the reducer rule the vectorized `apply_actions` preserves
  bit-exactly modulo the action-selection RNG.
