---
title: Showcase Demo
kind: topic
first_seen: workspace
last_updated: workspace
status: active
---

## Goal

Tell the evolution-arc story in under a minute: hand-written heuristic versus naïve nets → swap in champions across generations → final champion dominates. The visual is the pitch; flow lines and ownership cascades carry the story without voiceover.

## The two frames that sold the idea

These are the reference frames the demo recreates. They came out of unscripted play and are why this work exists.

![scene 1 — baseline: the dumb heuristic beats untrained nets, flow lines visible](../media/scene-1-baseline.png)

*Baseline. The hand-written `aggressive` (seat 0) versus eleven untrained `evolved` nets. The shape is the point — strength-scaled nodes plus colored half-edges read as living vasculature.*

![scene 3 — trained: a champion net dominates, multi-color border skirmishes around the edges](../media/scene-3-trained.png)

*After evolution. One champion expands as a deliberate blue mass; the other seats are reduced to coloured perimeter skirmishes. Same engine, same rules — only the genome changed.*

## Two ways to ship it

| | (A) edited video | (B) in-browser scripted demo *(canonical)* |
|---|---|---|
| medium | mp4/webm | live URL — `/?demo=1` |
| dependency | ffmpeg + screen recorder | nothing new — runs the deployed app |
| size | ~5–10 MB | ~290 KB extra (four champion JSONs) |
| editable | re-record + re-edit per change | bump a constant or swap a champion JSON |
| works where? | everywhere, including social previews | anywhere JS runs |
| smoke-testable in CI | no | yes — Playwright walks the demo |
| canonical? | secondary | **primary** |

We ship (B). (A) stays as a derivative — point Playwright at the demo URL to record a reproducible video for places JS can't reach.

## The arc (~30s, five scenes + intro)

Seat 0 is always `aggressive`; the other eleven seats are `evolved` and all read the same module-global champion (set in `src/gpu/evolved.ts`). Each scene is pre-simmed off-screen under its own champion, then played back as a snapshot stream over 5s wall-clock — see "Pre-sim + playback architecture" below.

| scene | label | caption | duration | champion file |
|---|---|---|---|---|
| 1 | `gen0`    | "watch ai battle"            | 5s | `null` — fresh random genome via `ensureChampion()` |
| 2 | `gen100`  | "the blue one is code"       | 5s | `gen100.json` *(placeholder, see below)* |
| 3 | `gen200`  | "the others are neural nets" | 5s | `gen200.json` *(placeholder)* |
| 4 | `gen1000` | "watch them get smarter"     | 5s | `gen1000.json` *(placeholder)* |
| 5 | `gen20k`  | "watch them win"             | 5s | `strong.json` — real trained champion |

Captions are lowercase by design. The runner cycles back to scene 1 after scene 5, so the demo loops indefinitely.

### Pre-sim + playback architecture

The demo does **not** step the sim live. Doing so would couple frame rate to sim cost and stutter on slow devices. Instead each scene is a **pre-computed snapshot array**, played back frame-by-frame in exactly 5s wall-clock:

1. **Pre-sim phase (off-screen).** Set the champion via `setChampion()`. Build initial state with `makeInitialState(undefined, undefined, 12)`. Loop `step(s, 0.1)` + (every 5 ticks) the 12-seat AI thinks (`applyAction` for every action each seat returns) for `PRESIM_TICK_BUDGET = 300` ticks (or until a single owner remains). Push every tick into `snapshots: GameState[]`. Yield via `await new Promise(r => setTimeout(r, 0))` every 50 ticks so the main thread breathes.
2. **Playback phase (5s wall-clock).** Each frame compute `t = sceneElapsed / 5s`, pick `snapshots[Math.floor(t * expectedLength)]` (clamped to actual length so partial pre-sims gracefully stall on the latest available frame), hand it to `updateScene(scene, snap, null)`. No `step()` calls during playback. Frame rate is decoupled from sim cost.

Because `setChampion()` is module-global state in `src/gpu/evolved.ts`, the five scene pre-sims must run **sequentially** — kicked off as a chained promise during `enter()`. Scene 0's pre-sim is awaited before the intro animation begins; scenes 1–4 race against playback wall-clock and almost always finish in time (each pre-sim is ~300 cheap `step()` calls; the intro animation alone is ~3.2s). `expectedLength` locks in once a pre-sim finishes — if a scene ends in a single-owner win early, `expectedLength` shrinks accordingly so playback's `t→idx` mapping uses the true span. Memory cost is ~5 × ~301 snapshots × ~271 nodes; cheap because `step` doesn't mutate, so snapshots share most structure by reference.

### Intro framing

Before scene 1, the runner pre-sims an off-screen game under the gen0 champion (random genome) for 150 ticks via the same `presimGame()` helper used for the scenes. From the last snapshot it computes a **hot area** — the length-weighted centroid of *cross-owner* flow midpoints (attacks across borders; the visually interesting flows). Falls back to centroid of all flows, then to origin if there are no flows yet. The intro then displays that frozen last snapshot while the camera pans + zooms to the hot point over 1s, holds the "AI WARS" title card for 1.5s, and zooms back to the wide view (0.7s). Only the camera animates during the intro; the rendered game state stays still.

Phase sequence per cycle: `intro-pan` (1.0s) → `intro-title` (1.5s, title card "AI WARS") → `intro-zoom-out` (0.7s) → for each scene: `scene-caption-in` (0.6s) → `scene-hold` (5s − 2×0.6s) → `scene-caption-out` (0.6s) → next scene. The scene-phase trio share one monotonic `sceneElapsed` counter so snapshot sampling is continuous across all 5s.

## What landed

Files under `src/demo/` and `public/champions/`:

- **`src/demo/runner.ts`** — scene state machine + pre-sim/playback engine. Public API: `createRunner({scene, overlay})` returns `{enter, tick, isActive, currentSnapshot, currentScene}`. Exports `SCENES` (the data list above) and `pickHotArea(state)`. The runner owns champion loading and pre-sim; the host only feeds it `dt`. Internals: `presimGame()` runs `step` (`dt=0.1`) plus the 12-seat AI every 5 ticks into a caller-provided snapshot array, breaking out early on a single-owner win and yielding every 50 ticks; `sampleSnapshot()` maps playback `t` to an index using the scene's `expectedLength` (which locks in to the actual recorded length once pre-sim finishes). Camera moves via existing `setViewSize` / `panBy` / `clampCamera` from [[../entities/flux]] (`src/render/scene.ts`); lerp + ease-in-out is inlined, no animation framework.
- **`src/demo/overlay.ts`** — pure DOM. `createOverlay()` returns `{showTitle, hideTitle, showCaption, hideCaption, destroy}`. Title is centered mono (`clamp(40px, 8vw, 96px)`, letter-spacing 8); caption sits in the bottom third (`clamp(20px, 3.2vw, 40px)`, letter-spacing 3). Both fade via CSS opacity transitions; container is `pointer-events:none` so input still reaches the canvas.
- **`src/demo/champions.ts`** — `loadSceneChampion(label) → Promise<Float32Array | null>`. Reads `public/champions/index.json`, fetches the mapped file, parses `weights`. Returns `null` for `gen0` so `ensureChampion()` in `src/gpu/evolved.ts` mints a fresh random genome.
- **`src/main.ts`** — `?demo=1` URL trigger sets `DEMO_MODE`. In that mode lil-gui (`gui.hide?.()`), `#flux-topbar`, `#hud`, the hint, and `#install-banner` are all `display:none`. The frame loop's runner branch is dead-simple: `runner.tick(dt)` → `updateScene(scene, runner.currentSnapshot(), null)` → `render(scene)`. The normal sim path is skipped entirely while the runner is active, so the live `state` variable, winner detection, and stasis detection all sit dormant. (No banner suppression hack needed — the live path simply doesn't execute.)

Champions in `public/champions/`:

| file | size | source |
|---|---|---|
| `gen100.json`  | 75K | placeholder, `mulberry32(100)`  + Gaussian `std=0.05` |
| `gen200.json`  | 73K | placeholder, `mulberry32(200)`  + Gaussian `std=0.15` |
| `gen1000.json` | 72K | placeholder, `mulberry32(1000)` + Gaussian `std=0.30` |
| `strong.json`  | 68K | real saved champion — copy of `flux-champion-gen12228-fit215.75.json`, fitness 215.75 |
| `index.json`   | — | scene-label → filename map, with `gen0: null` |

**Honest caveat:** `gen100` / `gen200` / `gen1000` are *not* real intermediate training snapshots. They're random Gaussian genomes seeded for visual variety — increasing `std` across the three files makes successive scenes look subtly different from `gen0` without actually carrying any training signal. The five-scene arc reads as a progression, but mechanically only `gen0` (untrained) and `gen20k` (trained) reflect real artifacts. Regenerate the placeholders via `node scripts/gen-champions.mjs` — fully deterministic.

## Playwright's role

Two uses, both downstream of the demo URL existing:

- **Reproducible recording.** Headed Playwright opens `/?demo=1`, records one full loop (intro + 5 scenes ≈ 33s) via `page.video()` or an ffmpeg-driven `chromium --use-fake-ui-for-media-stream`. Output is a clean mp4 with no editor in the loop.
- **CI smoke test.** A test that opens the demo URL and asserts the five scene captions appear in order. If the runner ever breaks, CI catches it. (Not yet wired.)

## Open questions

- Replace the placeholder genomes with real intermediate snapshots? Would require checkpointing the evolution loop at gens 100 / 200 / 1000 and saving each as a JSON. Cheap once we re-run training.
- Cinema-mode toggle independent of `?demo=1`? Right now the demo is the only way to hide chrome. A `c` keybind or `?cinema=1` was floated but not built.
- Deterministic seed for the board layout? `?seed=N` was discussed; not implemented in this pass. The demo currently uses whatever `makeInitialState()` produces from `Math.random`.
- "Skip to live" button mid-demo? Not built. Today you exit by reloading without `?demo=1`.

## Status

Implementation landed. The five-scene runner exists in `src/demo/`, gated by `?demo=1`; champions sit in `public/champions/`. `npm run typecheck` clean. Browser/visual verification is on the user — agents did not poke at it.
