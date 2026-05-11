---
title: Python Port (Bit-Exact Parity Bridge)
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## What

A second implementation of flux lives at `python/`. It mirrors `src/game/` and the genome forward pass from `src/gpu/genome.ts` exactly. Same algorithm, same arithmetic, same iteration order.

The Python side is independent of `src/`. It imports nothing from TypeScript. The contract is **algorithm-level**, not module-level — both sides consume the same spec.

## Why this shape

[[neuroevolution]] tier 4 lives in WebGPU because the browser is the deployment target. But training is GPU-bound and laptops want MLX. The Python port is the on-ramp:

1. NumPy port today, bit-exact against JS. (This page.)
2. MLX-accelerated forward pass and step kernel later.
3. Champion genomes serialize to JSON; either side loads either side's champions and plays them in any runtime — browser, headless JS sim, Python sim, MLX trainer.

The bridge is filesystem JSON (the existing `public/champions/` format). No HTTP, no IPC, no shared memory. Cross-language reproducibility is the unit test.

## The parity invariant

**The Python sim must produce bit-identical state to the JS sim for any deterministic seeded scenario.** If they diverge, the Python side is wrong (JS is the reference of truth). The invariant covers:

- `make_initial_state` — cell ordering, perimeter polar-angle sort, seat placement, edge enumeration order.
- `step` — force accumulation order, capture-on-zero-crossing, `MIN_STRENGTH_TO_SEND`, `ATTACK_BONUS`.
- `apply_action` — toggle / add / remove flow semantics.
- `mulberry32` — JS `>>> 0` and `Math.imul` semantics replicated with explicit 32-bit masking.
- `nn_infer_cell` — Float32Array store rounding at each accumulation. Hidden and output buffers are NumPy `float32`; arithmetic per expression evaluates in Python `float` (IEEE 754 binary64), then quantizes back to f32 on store. Matches JS Float32Array semantics exactly.
- `build_neighbor_table` — per-cell sort by `(pos.x, pos.y)` for determinism.
- `ai_think` reconcile-flows logic — Python dict insertion order matches JS Map insertion order, so action emission order is identical.

The parity test at `python/tests/test_parity.py` (Python) and `python/tests/dump_reference.ts` (JS) runs the same 100-tick scenario on both sides and compares SHA-256 hashes every 10 ticks. All 11 hashes must match byte-for-byte.

## What's NOT in scope on this page

- Evolution loop (rtNEAT or batch — separate decision when it lands).
- Champion JSON read/write on the Python side.
- MLX kernels.
- HTTP / IPC bridges.

Those come later. This page documents only the foundation: parity-validated reimplementation of game + NN forward pass in Python.

## Tradeoffs

- **Cost:** double-implementation of step + NN. Means bugs need a fix on both sides. Mitigation: the parity test catches drift within one tick.
- **Strict float matching constrains design.** We can't trade arithmetic order for clarity in either implementation. Every loop and every store has to mirror the other side.
- **Worth it because:** the alternative (Python implementation that's "close enough") makes evolution non-reproducible across runtimes, which kills the offline-train / online-deploy bridge.

## Files

- `python/pyproject.toml` — `uv`-managed, Python 3.12+, NumPy 2.x.
- `python/flux/state.py` — dataclasses + constants.
- `python/flux/graph.py` — `make_initial_state`.
- `python/flux/step.py` — `step`, `apply_action`.
- `python/flux/rng.py` — `mulberry32`.
- `python/flux/genome.py` — `nn_infer_cell`, `build_neighbor_table`, `random_genome`, `ai_think`.
- `python/tests/test_parity.py` — Python side of the parity scenario.
- `python/tests/dump_reference.ts` — JS reference; `npx tsx python/tests/dump_reference.ts`.
