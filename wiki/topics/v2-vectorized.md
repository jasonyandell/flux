---
title: v2 vectorized (provisional)
kind: topic
first_seen: 2026-05-15
last_updated: 2026-05-15
status: provisional
---

## Status

**Provisional.** Lives on branch `worktree-v2-vectorize-compact`
(commit `194f29b`). All 141 Python tests + tsc + vite build pass, and
all 17 solver modes dispatch and play through a small smoke run, but
the matched-pair tournament numbers in
[[v2-overnight-research|v2-overnight-research]] have not yet been
re-checked under the new code. Until that happens, treat this page as
a separate track — the wiki's current rankings still describe the
pre-vectorize loop solvers.

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

## Measured numbers

| | before | after |
| --- | --- | --- |
| 6000-tick R=20 6-seat `lightning_sum_long`, single-thread | 9.7s | 6.0s (1.6×) |
| 6000-tick R=30 6-seat `lightning_sum_long`, single-thread | 31.3s | 25.2s (1.24×) |
| 18 same R=20 games, `--workers 18` | n/a (single-process baseline) | 19s wall (5,700 ticks/sec aggregate) |
| Replay file size (R=20 6000-tick stride-5) | ~20 MB | 468 KB (~43×) |
| Replay file size (R=20 6000-tick stride-25) | ~5 MB | 109 KB (~180×) |
| Python tests | 141 pass | 141 pass |
| `tsc --noEmit` | clean | clean |
| `vite build` | clean | clean |

The single-thread speedup shrinks on bigger boards (1.6× at R=20,
1.24× at R=30) because the picker-loop overhead I removed scaled
with the count of *owned cells*, while `compute_potential`'s
32-iteration `(N, K)` field iteration scales with *N* and was
already vectorized. As `N` grows, `compute_potential` becomes the
dominant cost. The bigger wins are shape, not raw throughput:
parallel-throughput on `--workers 18`, and the per-replay file size
dropping from MB to KB so the trainer-displayer can live on the
Cloudflare free tier and load on a phone.

Both R=30 runs stalemated at 6000 ticks under identical seeds with
different surviving-seat counts (old=4 alive, new=5 alive) — the
expected divergence from the two intentional semantic deltas
documented below. Macro behavior class is the same (same regime,
same dominance, same outcome).

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
