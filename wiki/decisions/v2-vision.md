---
title: v2 Vision (3-Hop Receptive Field)
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Choice

Run a second model class — **v2** — alongside v1. v2 widens the per-cell receptive field from 2-hop (18 neighbors) to 3-hop (36 neighbors). Same hidden width (32) and same output space (19 = 18 distance-2 action edges + noop). Input grows 91 → 181; total weights grow 3571 → 6451 (~1.8×).

v1 is **kept intact**. The training loop selects model class via `--model {v1,v2}` on `python/scripts/train.py`; checkpoints and champions live in separate directories (`python/checkpoints/{latest.npz, v2/latest.npz}` and `public/champions/{*.json, v2/*.json}`).

## Why

The 2-hop receptive field gives a cell its immediate neighborhood and one ring out. That hides reinforcements building two cells away — the NN can't see backlines forming until they're already adjacent. The 3-hop receptive field exposes that second ring explicitly: the NN can see incoming reinforcements before they arrive at the contact line.

The output space stays at 19. Flows still only travel over the game's distance-2 edges. v2 widens **vision** without widening **action**.

## Why keep v1

- v1 has gen 5372 of compute baked in. Throwing it out would discard a known-strong reference.
- Comparison signal. "Does the 3-hop receptive field actually accelerate evolution, or did we just retrain on a faster machine?" only has an answer if both models train in the same regime.
- Cost is low: a CLI flag and a parallel checkpoint dir.

## Cost

- ~1.8× weights per genome (3571 → 6451).
- Features tensor scales G × S × N × 181 instead of × 91, so memory bandwidth roughly doubles per AI tick. Doesn't change the qualitative G-throughput curve (G > 24 is still counterproductive on memory bandwidth) but pushes the inflection slightly lower.
- New file: `python/flux/vision.py` (3-hop neighbor table, `STRIDE_V2 = 36`). New constants module: `python/flux/mlx_genome_v2.py`.

## Early result (2026-05-11)

- v1 at gen 5372, all-time-best fitness 1540.50.
- v2 at gen 347 (started fresh same day), all-time-best fitness 1521.50.

v2 caught v1 in ~6.5% of v1's generation budget. Wider vision **appears** to massively accelerate evolution. Caveats: small sample of one run per model; v2 has run for far fewer generations, so we don't yet know whether v1's ceiling is higher, the same, or lower. The result is consistent with "more useful per-cell signal → faster credit assignment," but not proof of it.

## Files

- `python/flux/vision.py` — 3-hop neighbor table, `STRIDE_V2 = 36`.
- `python/flux/mlx_genome_v2.py` — v2 layout constants.
- `python/flux/mlx_batch.py` — `build_flows_batched_v2` is the v2-specific batched AI tick (counterpart to `build_flows_batched` for v1).
- `python/checkpoints/v2/latest.npz` — v2 checkpoint (auto-resumed by `train.py --model v2`).
- `public/champions/v2/*.json` — v2 champions for the browser.

## What's next

- Run v2 to a similar generation count as v1 and compare ceilings.
- If v2 wins decisively at saturation, retire v1 and consider a v3 with deeper hidden width (input is already wide; hidden = 32 may be the next bottleneck).
- If v2 plateaus near v1's ceiling, the win is "faster credit assignment, same skill ceiling" — useful for training velocity, not for max-skill matches.
