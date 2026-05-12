---
title: Replay Rendering (Python is the Lab, Web is the Player)
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Choice

The browser default mode is **replay playback**, not live evolution. Python (MLX) runs the training loop and writes binary `.flxr` files; the web app parses and plays them back. The lil-gui "watch replays" toggle is on by default; the top bar shows `[model] gen N · best F`.

The in-browser WebGPU evolution ([[webgpu-evolution]]) still exists and still works, but it is a side-quest. Champions on disk now come from Python.

## Why this shape

- **Training wants Metal directly.** MLX on Apple Silicon GPU outruns the WebGPU round-trip (no driver, no rebind, full unified memory, easy batching across games × seats × cells). See [[../topics/neuroevolution|neuroevolution]] Tier 5.
- **Recording is cheap on the writer side.** A sampled-frame dump every K ticks is small enough to ship over a static server. No live websocket, no IPC.
- **Web stays simple.** The browser keeps its renderer; the replay player feeds the same `GameState` shape that `updateScene` already consumes. No new rendering code path.
- **One contract: the `.flxr` file.** Both sides agree on the binary format. The codebase has two independent producers (Python now, JS pre-sim for the showcase demo could later target the same format) and one consumer (the player).

## Files

- `python/flux/replay.py` — writer. Binary header + sampled-frame body. Header carries model id, genome generation, board config; body is one record per recorded tick.
- `src/replay/format.ts` — TypeScript mirror of the binary layout (parser).
- `src/replay/player.ts` — playback loop. Maps wall-clock to frame index, hands the resulting `GameState` to the renderer.
- `public/replays/*.flxr` — produced by `python/scripts/train.py`. Index at `public/replays/index.json` is auto-pruned to 50 entries.

For the schema of the `.flxr` format itself, `python/flux/replay.py` is the source of truth; `src/replay/format.ts` is the consumer-side mirror and must match byte-for-byte.

## What "the web is a replay renderer" means

The frame-loop branch for replay mode:

1. Pick the latest `.flxr` from `public/replays/index.json` (or the user-selected one).
2. Stream + parse via `src/replay/format.ts`.
3. `src/replay/player.ts` walks the frames in wall-clock; the live sim path is dormant.
4. Top bar reads `[model] gen N · best F` from the replay header.

No `step()` is called in this mode. The browser is rendering Python's output, not advancing state.

## Tradeoffs

- **Replay is sampled.** A `.flxr` is not a bit-exact reproduction across renderers — it's the canonical sequence of `GameState` snapshots at the recording cadence. If a frame wasn't recorded, it's not in the replay.
- **Two consumers must stay in sync.** Any field added to the binary header has to land in both `python/flux/replay.py` and `src/replay/format.ts` in the same change.
- **WebGPU evolution drifts further from the deployed flow.** Worth accepting; it's a side-quest now, not the deployed product.

## What this is NOT

- Not a hot-reload bridge. The browser does not stream replays from a running trainer; it reads files written to disk and served statically. Cross-process synchronization is just "the file is there now."
- Not a substitute for the champion JSON contract. Champion weights still ship as `public/champions/**/*.json`; `.flxr` carries game state, not genome.

## Greatest hits cycle

A curated cycle mode exists alongside the live-newest mode. `python/scripts/build_greatest_hits.py` walks `public/replays/*.flxr`, parses the FLXR header for `num_frames` and `metadata.best_fitness`, filters to positive-fitness games with at least 200 recorded frames, sorts longest-first (fitness as tiebreaker), and writes the top 30 to `public/replays/greatest-hits.json`.

Browser side: a `greatestHits` lil-gui tunable calls `replayPlayer.setIndexUrl()` to swap between `replays/index.json` (live newest-first) and `replays/greatest-hits.json` (curated cycle). In greatest-hits mode the auto-speed targets ~2 seconds per replay for max-DPS phone viewing.

Camera resets to origin on every replay swap so panning from one game doesn't carry into the next.

## Playback cadence

Replays are recorded with a configurable `record_stride` (default 10) — each recorded frame represents `record_stride · 0.1s` of game time. The FLXR header's `tick_stride` stores this.

The browser's auto-speed targets **one recorded frame per browser frame at ~60Hz** regardless of stride. So a `tick_stride=1` (tick-by-tick) replay plays each game tick in one browser frame; a `tick_stride=10` replay plays 10 game ticks per browser frame. Same wall-clock per recorded frame either way — the user picks the visualization resolution by choosing stride at train time. Greatest-hits mode overrides this with its 2s-per-game max-DPS target.

The replay player **plays each game to its last frame before swapping**. If a newer replay drops mid-game, it's queued via `pendingFile` and loaded only at end-of-replay. This lets you watch full games instead of jumping to the newest drop the moment it lands.

## Flow arrows

Replays render directional arrows for each active flow at the rendering tick. Each flow is drawn as **three line segments per stack** (shaft + two arrowhead wings) at z=0.3 above the cell layer, with a gradient from a dim source-side colour to a brighter tip.

**Per-flow visual emphasis scales with source-cell strength** (a proxy for outgoing power):
- **Stacked thickness**: 1–5 perpendicularly-offset copies of each flow (WebGL line widths are uniformly ignored across browsers, so stacking is the practical thickness substitute). Strong flows render as a thick band, weak ones as a single line.
- **Reach**: weak flows shaft to the midpoint; strong flows push their tips closer to the destination (up to ~80% of the way).
- **Arrowhead size**: scales 0.7×–1.3× with source strength.
- **Brightness gradient**: dim at the source-side end, brighter at the tip, with overall brightness rising with strength.

Edge mesh contrast was also bumped (`0x1a1a1a → 0x2a3548`) so the graph topology is visible on a phone screen. Node base radius dropped 20% (`0.45 → 0.36`) to leave more visual room between cells now that flow arrows are emphasized.

## Scene rebuild on board-size change

`rebuildSceneGeometry` runs on **every replay swap** (not only when node count differs), and now also calls `nodeInstanced.dispose()` to release the per-instance attribute buffers. This eliminates a ghosting bug where switching between boards of the same size could leave stale edge geometry visible.
