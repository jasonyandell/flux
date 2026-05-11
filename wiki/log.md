---
title: flux Wiki Log
kind: log
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## [2026-05-11 | workspace | five-scene showcase demo lands (pre-sim + playback)]

**Touched pages:** [[topics/showcase-demo]] [[index]] [[log]]
**Added:**
- `src/demo/runner.ts` — scene state machine with **pre-sim + snapshot playback**. Each scene runs an off-screen game once via `presimGame()` (300 ticks of `step(s, 0.1)` + the 12-seat AI applying actions every 5 ticks; breaks out early on a single-owner win; yields every 50 ticks via `setTimeout(0)`), recording every tick into a `GameState[]`. Playback maps wall-clock `t = sceneElapsed / 5s` to `snapshots[Math.floor(t * expectedLength)]` and hands the result to `updateScene`. No `step()` calls during playback — frame rate is decoupled from sim cost. `expectedLength` locks in to the actual recorded length once pre-sim finishes, so winner-early-exits compress correctly and partial pre-sims clamp gracefully to the latest available frame. `SCENES` is a pure data list of `{label, caption, durationSec}` for `gen0`/`gen100`/`gen200`/`gen1000`/`gen20k`, 5s each, lowercase captions. Public API: `createRunner({scene, overlay})` returns `{enter, tick, isActive, currentSnapshot, currentScene}`. Also exports `pickHotArea(state)` — length-weighted centroid of cross-owner flow midpoints, falling back to all-flow centroid, then origin. Phases: `intro-pan` (1.0s) → `intro-title` ("AI WARS", 1.5s) → `intro-zoom-out` (0.7s) → per scene: `scene-caption-in` (0.6s) → `scene-hold` (3.8s) → `scene-caption-out` (0.6s); loops scene 0 after scene 4. Scene-phase trio share a single monotonic `sceneElapsed` so snapshot sampling is continuous.
- `src/demo/overlay.ts` — pure DOM. `createOverlay()` returns `{showTitle, hideTitle, showCaption, hideCaption, destroy}`. Fixed-inset, `pointer-events:none` container; centered mono title (`clamp(40px,8vw,96px)`, letter-spacing 8) and bottom-third caption (`clamp(20px,3.2vw,40px)`, letter-spacing 3). Both fade via CSS opacity transitions.
- `src/demo/champions.ts` — pure fetch helper. `loadSceneChampion(label) → Promise<Float32Array | null>` reads `public/champions/index.json`, fetches the mapped file, parses `weights`. Returns `null` for `gen0` so `ensureChampion()` in `src/gpu/evolved.ts` mints a fresh random genome. The runner owns calling this (no callback contract with `main.ts`).
- `public/champions/strong.json` — copy of `flux-champion-gen12228-fit215.75.json` (fitness 215.75). Maps to the `gen20k` scene; the only real trained genome in the catalog.
- `public/champions/gen100.json` / `gen200.json` / `gen1000.json` — deterministic *placeholder* genomes seeded by `mulberry32(100|200|1000)` with Gaussian `std` 0.05 / 0.15 / 0.30 respectively. Marked `"note": "placeholder, random-seeded"`. **Not** real intermediate training snapshots — they exist purely for visual variety across the five scenes. Regenerate via `node scripts/gen-champions.mjs`.
- `public/champions/index.json` — scene-label → filename map (`gen0: null`, rest point at the JSONs above).
- `scripts/gen-champions.mjs` — Node-runnable generator. Inlines `mulberry32` + Gaussian sampling so it needs no TS toolchain.
**Updated:**
- `src/main.ts` — `?demo=1` URL trigger sets `DEMO_MODE`. In that mode lil-gui (`gui.hide?.()`), `#flux-topbar`, `#hud`, the hint, and `#install-banner` are all `display:none`. The frame loop's runner branch is dead-simple: `runner.tick(dt)` → `updateScene(scene, runner.currentSnapshot(), null)` → `render(scene)`. The normal sim path is skipped entirely while the runner is active, so the live `state` variable, winner detection, and stasis detection all sit dormant — no banner suppression hack required.
- [[topics/showcase-demo]] rewritten end-to-end: replaces the old 3-scene baseline/training/emergence plan with the shipped 5-scene gen-progression arc, documents the pre-sim + playback architecture, drops the speculative "what this needs from the codebase" list in favor of a "what landed" section, adds an honest caveat that `gen100`/`gen200`/`gen1000` are random-seeded placeholders rather than real intermediate snapshots, and consolidates the champions catalog with file sizes + sources.
- [[index]] — showcase-demo route-map line updated to reflect the shipped reality (`?demo=1`, five 5s scenes, hot-area intro + "AI WARS" title card).
**Mechanics worth pinning:**
- Pre-sims must run **sequentially** because `setChampion()` is module-global state in `src/gpu/evolved.ts` — they're chained as a promise during `enter()`. Scene 0's pre-sim is awaited before the intro begins; scenes 1–4 race against playback wall-clock and almost always finish in time (each is ~300 cheap `step()` calls; the intro animation alone is ~3.2s).
- Memory: 5 × ~301 snapshots × ~271 nodes. Cheap because `step` doesn't mutate, so snapshots share most structure by reference.
**Retired:**
- The old 3-scene plan (baseline → training montage → emergence). Lives in git; the topic page now describes the shipped flow.
- An earlier version of the runner that stepped the live sim at `DEMO_SPEED = 100` and used `getState`/`loadScene` callbacks from `main.ts`. Replaced mid-iteration by the pre-sim + playback architecture — frame rate is now decoupled from sim cost, which matters on slow devices and makes the 5s-per-scene budget exact.
**Questions opened:**
- Replace placeholder genomes with real checkpoints if/when we re-run training and snapshot gens 100/200/1000.
- Cinema-mode toggle independent of `?demo=1` (not built; was floated as `c` keybind / `?cinema=1`).
- Deterministic `?seed=N` for board layout (not built in this pass; `makeInitialState()` still uses `Math.random`).
- "Skip to live" button mid-demo (not built; reload without `?demo=1` is the current exit).
**Verification:** `npm run typecheck` clean. Browser/visual verification deferred to the user — no agent loaded the demo URL.

## [2026-05-11 | workspace | showcase-demo topic page]

**Touched pages:** [[topics/showcase-demo]] [[index]] [[log]]
**Added:** [[topics/showcase-demo]] — first-class plan for the evolution-arc demo. Compares an edited video (A) against an in-browser scripted demo (B), recommends (B), and lists the small surface the codebase needs: deterministic seed, canned champions in `public/champions/`, `?demoSpeed=N` override, cinema mode (hide all chrome), a `src/demo/runner.ts` scene state machine, and lerp'd camera keyframes reusing the existing clamp. Playwright is used downstream for reproducible video capture and as a CI smoke test of the scene transitions. First entries in `wiki/media/`: `scene-1-baseline.png` (the heuristic-dominates frame) and `scene-3-trained.png` (the champion-dominates frame), embedded at the top of the page as reference frames the scene runner has to recreate.
**Updated:** [[index]] now lists `showcase-demo` under topics.
**Retired:** none.
**Questions opened:** training cadence for canned champions (per-release re-train vs pinned trio); whether scene 2's training montage uses live GPU evolution or pre-rendered stepping.

## [2026-05-11 | workspace | top-bar jitter fix + HUD expando]

**Touched pages:** [[entities/flux]] [[log]]
**Added:** none new.
**Updated:**
- `src/render/gameui.ts`: stats slot in the top bar now has `min-width:22ch` + `font-variant-numeric:tabular-nums`, and the evolve subscript has `min-width:14ch`. The top bar is `transform:translateX(-50%)` centered, so any width change in stats or sub used to slide the evolve button each generation — locking those slots removes the jitter.
- `src/render/gameui.ts` + `src/main.ts` + `index.html`: the HUD became a compact expando (`▸ tick N · X/Y alive` collapsed, full per-seat list on tap) so the 12-row listing no longer collides with the centered top bar on narrow viewports. `white-space:nowrap` on the top bar keeps `drains battery` from wrapping; the evolve button now stacks `drains battery` as a small subscript under the label.
**Retired:** none.
**Questions opened:** none.

## [2026-05-10 | workspace | top-bar casual-demo UI]

**Touched pages:** [[entities/flux]] [[log]]
**Added:** none new.
**Updated:**
- `src/render/gameui.ts` exports `createTopBar()` — a centered pill at `top:env(safe-area-inset-top)` with `↻ Restart`, an `Evolve` toggle (subtitle "drains battery"; auto-disables to "no WebGPU" when `initGPU()` returns null), and a live `gen N · best F` readout.
- `src/main.ts` wires the top bar to `respawn()` and to the existing evolve toggle (mirrors `saveEvolveEnabled` + `startEvolution`), tracks last-rendered values to avoid per-frame DOM thrash, and calls `gui.close()` so lil-gui starts collapsed and the demo's two primary buttons dominate.
- [[entities/flux]] gameui bullet documents the top bar role.
**Retired:** none.
**Questions opened:** none.

## [2026-05-10 | workspace | map-style touch input + camera clamp]

**Touched pages:** [[entities/flux]] [[decisions/multi-player-free-for-all]] [[log]]
**Added:** none new.
**Updated:**
- `src/main.ts` now drives camera through a shared `zoomAndPanAt(before, after, factor)` helper. Wheel zoom is unchanged in feel; new pointer handlers track active touches in a flat list, derive centroid + mean spread, and apply pan + pinch in one step. One finger pans, two fingers pinch + pan anchored on the centroid.
- `src/render/scene.ts` records per-axis `worldHalfWidth`/`worldHalfHeight` at scene creation and exports `clampCamera`. `panBy`, `setViewSize`, and `resizeRenderer` all clamp afterwards. At max zoom out (or when the world fully fits in either axis) the camera is forced to 0 on that axis, which fixes the "zoom in then out leaves the board shifted" bug.
- [[entities/flux]] input/scene descriptions reflect the pan + pinch model and the bounds clamp.
- [[decisions/multi-player-free-for-all]] decision body now lists the full camera-input set; the re-enabling-human-play note flags that pointerdown is no longer free for clicks.
**Retired:** none.
**Questions opened:** none.

## [2026-05-10 | workspace | wiki cleanup + session catchup]

**Touched pages:** [[decisions/inbound-bonus]] [[decisions/stasis-detection]] [[todo]] [[questions/open]] [[index]] [[log]]
**Added:** none new.
**Updated:**
- [[decisions/inbound-bonus]] trimmed to a one-paragraph retired marker.
- [[decisions/stasis-detection]] documents all three suppression rules — buffer-not-full, 1v1 endgame, and cleanup phase (runner-up < 5 cells, absolute not percentage).
- [[todo]] drops the resolved stasis-screenshot item and refreshes the constant-tuning note (`REGEN_PER_SEC` 1.0 → 1.1 already shipped).
- [[questions/open]] mitigation note describes both stasis suppression rules.
- [[index]] drops the retired-pointer line for `inbound-bonus`.
**Retired:** none (`inbound-bonus` was retired earlier this session).
**Questions opened:** none.

**Session catchup** — code changes shipped earlier without their own log entries, summarized here:

- `SPEED = 5` scaler in `main.ts` for 5× spectator playback.
- `REGEN_PER_SEC` bumped 1.0 → 1.1 for pacing.
- Per-seat AI dropdowns in lil-gui; default board is 12 seats, seat 0 = `aggressive`, rest = `evolved`.
- Capture folder in lil-gui: PNG snapshots and WebM recordings via `MediaRecorder` on the canvas stream.
- Champion save/load buttons writing/reading `flux-champion-genN-fitF.json`. Sim runner reads `FLUX_CHAMPION_JSON=path` for head-to-head tests.
- `localStorage` backup of full evolution state with auto-resume on reload; toggle state persists; clear-save button in lil-gui.
- Sim `tournament` subcommand cells print `P0/P1/draws` instead of just P0 wins.
- Evolution fitness gains `lingerPenalty = 0.5` per remaining opponent cell.
- Evolution `boardRadius` 8 → 9 (~271 cells) per user spec.
- Stasis exemptions added in steps: first 1v1 (alive ≤ 2), then cleanup (runner-up < 5). Consolidated in [[decisions/stasis-detection]].
- `INBOUND_BONUS` shipped then reverted within the session — see retired [[decisions/inbound-bonus]] for the math walk.
- `wiki/todo.md` added as an active-threads holding pen.

## [2026-05-10 | workspace | webgpu neuroevolution MVP]

**Touched pages:** [[decisions/webgpu-evolution]] [[topics/neuroevolution]] [[entities/flux]] [[index]]
**Added:** [[decisions/webgpu-evolution]] capturing the MVP architecture (population eval on GPU, JS evolution loop, champion as `evolved` seat), the 91→32→19 controller shape, hyperparameters (P=12, σ=0.05, elites=3, tournament=3, 500 ticks, board radius 8 during evolution), the parity invariant (`WGSL step ≡ JS step`), and the WebGPU-missing fallback. `src/gpu/` added with `runtime.ts` (init), `shaders/step.wgsl` (port of `step.ts` over a games-batch), `shaders/nn.wgsl` (per-cell forward + flow rebuilder), `step.ts` (driver), `evolution.ts` (generation loop), `genome.ts` (layout + matching JS forward pass), `evolved.ts` (the `aiThink` registered in `src/ai/index.ts`), `parity.ts` (parity test exposed via lil-gui). `@webgpu/types` added to devDeps and tsconfig.
**Updated:** `src/main.ts` exposes an "evolution" folder with `evolve` toggle, `generation`/`bestFitness` displays, and "run parity test"; the `evolved` AI joins the registry; `src/ai/index.ts` adds the `evolved` entry; [[topics/neuroevolution]] marks tier 1 + tier 4 as in progress; [[entities/flux]] lists the new `src/gpu/` frontier; [[index]] adds the decision.
**Retired:** none.
**Questions opened:** none new.

## [2026-05-10 | workspace | stasis detection]

**Touched pages:** [[decisions/stasis-detection]] [[entities/flux]] [[index]] [[questions/open]]
**Added:** [[decisions/stasis-detection]] documenting the variance-window detector (`STASIS_SAMPLE_PERIOD_TICKS = 5`, `STASIS_WINDOW = 50`, `STASIS_EPSILON = 1.0`), the pure `detectStasis` in `src/sim/stasis.ts`, and the `showStasisBanner` sibling in `src/render/gameui.ts`. `src/main.ts` now keeps a ring buffer of per-player cell counts and freezes `step`/AI when stasis fires; `respawn()` resets the flag and buffer.
**Updated:** [[entities/flux]] lists `src/sim/stasis.ts` in the implementation frontier; [[index]] adds the decision; [[questions/open]] links to it as a mitigation for the four-AI stalemate.
**Retired:** none.
**Questions opened:** none new.

## [2026-05-10 | workspace | record neuroevolution as the planned next step]

**Touched pages:** [[topics/neuroevolution]] [[questions/open]] [[index]] [[entities/flux]]
**Added:** [[topics/neuroevolution]] capturing the rtNEAT direction — Stanley/Bryant/Miikkulainen lineage, NERO precedent, OpenNERO, controller shape (per-cell shared network, ~91→32→19 dense), evolution loop, four effort tiers, open implementation questions.
**Updated:** [[questions/open]] now points at [[topics/neuroevolution]] as the planned answer to the AI stalemate; [[index]] adds the topic; [[entities/flux]] mentions it as the next step.
**Retired:** none.
**Questions opened:** none new; the existing stalemate question now has a planned answer rather than just possible directions.

## [2026-05-10 | workspace | ai zoo + multi-player spectator]

**Touched pages:** [[decisions/ai-zoo]] [[decisions/multi-player-free-for-all]] [[entities/flux]] [[index]] [[questions/open]]
**Added:** [[decisions/ai-zoo]] documenting six pure heuristics (`aggressive`, `random`, `defensive`, `greedy-neutral`, `opportunist`, `cluster`) under `src/ai/` with shared `utils.ts`/`rng.ts` and an `index.ts` registry, plus the sim `pair` and `tournament` subcommands. [[decisions/multi-player-free-for-all]] documenting the spectator-mode pivot: 2/4/6/8/12 perimeter-spaced AI seats, no human, wheel zoom only, `SPEED = 5` scaler. `REGEN_PER_SEC` bumped from 1.0 to 1.1 for pacing.
**Updated:** [[entities/flux]] rewritten to describe the multi-player spectator mode, the AI zoo, the 12-color palette, and the `SPEED` scaler; [[index]] adds the two new decision pages; [[questions/open]] narrows the AI stalemate to the four-AI "weakest-local-neighbor" attractor.
**Retired:** `src/ai/dumb.ts` (renamed to `src/ai/aggressive.ts`).
**Questions opened:** none new. Tournament confirmed defensive/opportunist lose to all four "active" heuristics, which stalemate against each other on the 1000-cell board.

## [2026-05-10 | workspace | attack bonus replaces loop bonus]

**Touched pages:** [[attack-bonus]] [[loop-bonus]] [[continuous-flow-model]] [[index]] [[questions/open]]
**Added:** [[attack-bonus]] decision page; `ATTACK_BONUS = 0.5` constant in `state.ts`; non-friendly destination multiplier in `step.ts`. Friendly transfer is back to a wash.
**Updated:** [[continuous-flow-model]] now references [[attack-bonus]] instead of loop bonus; [[index]] route map; [[questions/open]] notes the stalemate persists even with combat acceleration — confirming the failure mode is the heuristic, not the combat constants.
**Retired:** [[loop-bonus]] (status: retired; one-line pointer to [[attack-bonus]]).
**Questions opened:** none new.
**Verification:** by hand. One-sided attack: attacker drains at 2/sec (regen 1 − k 3); passive defender drains at 3.5/sec (regen 1 − k·1.5 = 4.5). Mutual fire: both drain at 6.5/sec (regen 1 − k 3 source − k·1.5 inbound). Sim outcome: still 5/5 draws — the symmetric mirror match isn't broken by combat acceleration alone.

## [2026-05-10 | workspace | loop bonus on friendly flows]

**Touched pages:** [[loop-bonus]] [[continuous-flow-model]] [[index]]
**Added:** [[loop-bonus]] decision page; `LOOP_BONUS = 0.5` constant in `state.ts`; friendly-destination multiplier in `step.ts`.
**Updated:** [[continuous-flow-model]] now describes the friendly-destination multiplier and backlinks to [[loop-bonus]]; [[index]] route map.
**Retired:** none.
**Questions opened:** none new. Stalemate sim outcome unchanged — dumb AI doesn't build chains, so the bonus has no effect under it.
**Verification:** by hand from constants. 3-cycle of friendly-owned cells with three active flows nets `+2.5·dt` per node per tick versus idle baseline of `+1.0·dt` — circulation now grows 2.5× faster than idle.

## [2026-05-10 | workspace | hex grid default + instanced renderer]

**Touched pages:** [[hex-grid-default]] [[flux]] [[index]] [[questions/open]]
**Added:** [[hex-grid-default]] decision page covering the new ~1000-cell hex board, renderer batching, the `WeakMap`-backed adjacency cache in `applyAction`, and the per-call adjacency list in `aiThink`.
**Updated:** [[flux]] reflects the hex board, instanced renderer, drag-input model, and slower-per-run sim; [[index]] route map; [[questions/open]] notes the stalemate persists at hex scale.
**Retired:** none. Old 7-node hand-laid graph is gone but lives in git.
**Questions opened:** none new. Browser was not tested from this session — flagged in commit.

## [2026-05-10 | workspace | wiki audit against idle-tower]

**Touched pages:** none in wiki body; added top-level `AGENTS.md`, reduced `CLAUDE.md` to a pointer.
**Added:** top-level `AGENTS.md` matching idle-tower's convention (project notes + pointer to wiki).
**Updated:** `CLAUDE.md` now defers to `AGENTS.md`.
**Retired:** none.
**Questions opened:** none.
**Audit notes:** schema (frontmatter, log format, page conventions, filenames) matches idle-tower's `wiki/AGENTS.md`. Directory layout is an intentional subset — flux omits `trails/` and `playbooks/`, consistent with "do not create pages speculatively". `kind` enum is narrower (no `experiment`, `trail`, `playbook`); fine until those page types are needed.

## [2026-05-10 | workspace | one flow per edge]

**Touched pages:** [[one-flow-per-edge]] [[continuous-flow-model]] [[index]]
**Added:** [[one-flow-per-edge]] decision capturing the new `applyAction` rule that at most one flow may exist per undirected edge, with reverse-as-flip semantics.
**Updated:** [[continuous-flow-model]] to reference the new per-edge constraint; [[index]] route map.
**Retired:** none.
**Questions opened:** none.

## [2026-05-10 | bootstrap | seed flux]

**Touched pages:** [[flux]] [[continuous-flow-model]] [[pure-step-function]] [[galcon-like]] [[questions/open]]
**Added:** initial wiki schema, route map, entity page, two decision pages, genre topic page; open question recording the dumb-AI stalemate observed in `npm run sim`.
**Updated:** none.
**Retired:** none.
**Questions opened:** dumb AI stalemates against itself.
