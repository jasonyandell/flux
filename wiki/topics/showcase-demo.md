---
title: Showcase Demo
kind: topic
first_seen: workspace
last_updated: workspace
status: active
---

## Goal

Tell the evolution-arc story in under two minutes: a hand-written heuristic crushes naïve nets → the lab cooks → a trained net crushes back. The visual is the pitch; flow lines and ownership cascades carry the story without voiceover.

## The two frames that sold the idea

These are the reference frames the demo has to recreate. They came out of unscripted play and are why this work exists.

![scene 1 — baseline: the dumb heuristic beats untrained nets, flow lines visible](../media/scene-1-baseline.png)

*Scene 1, baseline. The hand-written `aggressive` (seat 0) versus eleven untrained `evolved` nets. The shape is the point — strength-scaled nodes plus colored half-edges read as living vasculature.*

![scene 3 — trained: a champion net dominates, multi-color border skirmishes around the edges](../media/scene-3-trained.png)

*Scene 3, after evolution. One champion expands as a deliberate blue mass; the other seats are reduced to coloured perimeter skirmishes. Same engine, same rules — only the genome changed.*

## Two ways to ship it

| | (A) edited video | (B) in-browser scripted demo *(recommended)* |
|---|---|---|
| medium | mp4/webm | live URL, e.g. `/?demo=1` |
| dependency | ffmpeg + screen recorder | nothing new — runs the deployed app |
| size | ~5–10 MB | ~50 KB extra (champion JSON × 2–3) |
| editable | re-record + re-edit per change | bump a constant or swap a champion JSON |
| works where? | everywhere, including social previews | anywhere JS runs |
| smoke-testable in CI | no | yes — Playwright walks the demo |
| canonical? | secondary | **primary** |

We commit to (B). (A) stays in the toolbox as a derivative — once (B) exists, point Playwright at it to record a reproducible video for places JS can't reach.

## The arc (~75s, scripted scenes)

Three scenes, each a scripted state for the live sim. Each scene has: champion to load, seats, camera plan, speed, exit condition, next-scene trigger.

| scene | champion | seats | speed | camera | exits when |
|---|---|---|---|---|---|
| 1 — baseline | untrained / random genome | seat 0 = `aggressive`, others = `evolved` | 60–120× | wide; one push-in on the aggressive frontier | banner `AGGRESSIVE WINS` |
| 2 — training | n/a (montage) | n/a | n/a | full-canvas with a soft `gen N · best F` reveal; either time-lapse the real evolution loop or step through 3–4 canned generations (gen 5 → 20 → 50 → 200) | counter reaches a target gen / fixed timeout |
| 3 — emergence | fully-trained champion | same seat config as scene 1 | 60–120× | wide; one push-in on the evolved/aggressive border skirmish | banner `EVOLVED WINS` |

Target wall-clock: 5s per full sim run at 60×–120× speedup (`SPEED` constant in `main.ts` currently 5; a `demoSpeed` override raises it for the demo). A two-minute headless run compresses to ~3–4s on the screen.

## What this needs from the codebase

Small surface, can land incrementally:

1. **Deterministic seed.** `makeInitialState(...)` currently uses `Math.random` for board layout and seat placement. Accept a seed (already plumbed through `graph.ts` for some calls — verify and propagate). Surface as `?seed=N` URL param so a scene can pin its layout.
2. **Canned champions.** Add 2–3 saved `Float32Array` genomes to `public/champions/` (`untrained.json`, `mid.json`, `champion.json`). The "load champion" button already accepts this format. A scene loader calls `setChampion(new Float32Array(weights))` from `gpu/evolved.ts`.
3. **Demo speed override.** A `?demoSpeed=N` URL param (or scene field) that swaps `SPEED` for the duration of a scene. At 60×+ the existing flow-line rebuild per frame is still cheap (one `LineSegments` rewrite per tick — see [[../entities/flux]] render frontier).
4. **Cinema mode.** Toggle (key `c` or `?cinema=1`) that hides HUD, top bar, lil-gui. Optional: hide the banner too and use a custom on-canvas one for the demo.
5. **Scene runner.** New module `src/demo/runner.ts`. Pure data list of `Scene` objects (champion, seats, speed, seed, camera keyframes, exit condition); a tiny state machine in `main.ts` advances scenes on each scene's `exits when`. ~150 lines tops.
6. **Camera keyframes.** Reuse the existing `setViewSize` / `panBy` / `clampCamera` ([[../entities/flux]]). A scene's camera plan is just `(t_in_scene_seconds → {viewSize, x, y})` interpolated. Don't build an animation framework; lerp two keyframes and call it.

None of these block each other. Order of operations: (4) → (2) → (3) → (1) → (5) → (6).

## Playwright's role

Two uses, both downstream of (B) existing:

- **Reproducible recording.** Headed Playwright opens `/?demo=1&seed=42`, waits for scene 3 to finish, runs `cmd+shift+5`-equivalent via `page.video()` or `ffmpeg` driving a `chromium --use-fake-ui-for-media-stream` capture. Output is a clean mp4 with no editor in the loop.
- **CI smoke test.** A test that opens the demo URL, asserts the three scene transitions happen in order with the expected banners (`AGGRESSIVE WINS` then `EVOLVED WINS`). If a balance change ever breaks the arc, CI catches it.

## Open questions

- Champion-training cadence: do we re-train the canned champions on every release, or pin to a single curated trio? Probably pin; re-curate when balance constants change.
- Scene 2 (training montage): use the real GPU evolution loop *running live* for X seconds, or pre-rendered champion-stepping through gens? Live is more honest but won't work on devices without WebGPU; pre-rendered always plays. Probably pre-rendered for the demo URL, real GPU for the deployed app.
- Audio? Probably no. The visual is enough; audio is a production tax without a payoff.
- Does the demo URL also get a "skip to live" button? Yes — at any point, exit the demo state machine and hand control back to the user.

## Champions catalog

Files in `public/champions/` and their scene mappings (see `index.json`):

| scene label | file | source |
|---|---|---|
| `gen0` | `null` (no file) | runner calls `setChampion(null)`; `ensureChampion()` in `src/gpu/evolved.ts` mints a fresh random genome on first read |
| `gen100` | `gen100.json` | placeholder, `mulberry32(100)` + `std=0.05` |
| `gen200` | `gen200.json` | placeholder, `mulberry32(200)` + `std=0.15` |
| `gen1000` | `gen1000.json` | placeholder, `mulberry32(1000)` + `std=0.30` |
| `gen20k` | `strong.json` | real saved champion (copy of `flux-champion-gen12228-fit215.75.json`, fitness 215.75) |

Placeholders carry `"note": "placeholder, random-seeded"` in the payload. Regenerate via `node scripts/gen-champions.mjs` — fully deterministic.

Loader: `src/demo/champions.ts` exports `loadSceneChampion(label)` → `Promise<Float32Array | null>`. Pure module, no `three` / `lil-gui` / DOM. Scene runner feeds the result into `setChampion` from `src/gpu/evolved.ts`.

## Status

Plan only. No code yet. Next step is one PR landing items (4) cinema mode and (2) canned champions — those are the highest-information, lowest-risk pieces. Everything after is iteration on top.
