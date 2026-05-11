---
title: flux Wiki Log
kind: log
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## [2026-05-11 | workspace | demo script tightened: stalemate genome + adaptive churn truncation]

**Touched pages:** [[topics/showcase-demo]] [[log]]
**Added:**
- `public/champions/stalemate.json` — user-saved gen 153 from in-app GPU evolution in a different browser; now drives the `gen2000` scene via live presim instead of the trainer's still.
**Updated:**
- `src/demo/runner.ts` — final script + per-scene knobs:
  - Captions: scene 4 "they start to learn (gen 150)" and scene 5 "they win (gen 12k)" replace the older "watch them get smarter" / "watch them win" — gen numbers now reference what the underlying genome actually is (stalemate.json is gen 153 rounded; strong.json is gen 12228 rounded). Scene 3 caption "the others are neural nets" now lands on `gen50` instead of `gen200` so the gen label matches the sluggish-net look.
  - Per-scene `SceneSpec` knobs: `tickBudget`, `stopOnWinner`, `stillUrl`.
  - Adaptive churn truncation replaces a brief per-player variance check. Inter-sample Σ|Δcount|, rolling 50-sample mean, peak-tracking — when current churn drops under 15% of the peak window mean, declare stalemate and continue only until the flat tail is ≤30% of total kept ticks. Catches oscillating equilibria the per-player variance check missed (seats swapping cells around a frozen system). Suppression rules in `src/sim/stasis.ts` deliberately *not* reused — they're for in-game UX, not playback truncation.
  - `stillUrl` loader (`loadStill()`) rebuilds GameState from a baked `{boardConfig, owners[], strengths[], flows[]}` JSON. Was wired briefly to bypass a sim-vs-render divergence on `gen2000`; currently unused (stalemate.json live presim works) but kept for future single-frame scenes.
- `scripts/train-stalemate.ts` — after solving, re-runs the winner with recording on and dumps `public/champions/gen2000-still.json` (the trainer's final-state ground truth).
- `public/champions/index.json` — `gen2000 → "stalemate.json"`; old `gen200` / `gen1000` entries dropped in favor of `gen50`.
- `public/champions/gen50.json` (new) replaces `gen200.json` / `gen1000.json` (deleted). Regenerated via `scripts/gen-champions.mjs` with updated specs (std 0.03 / 0.08 / 0.40).
- [[topics/showcase-demo]] arc table, pre-sim architecture, champions catalog, and open questions all reflect the above.
**Retired:** the gen 0 → 100 → 200 → 1000 → 20k label sequence (replaced by 0 → 100 → 50 → 2000 → 20k narrative ordering); the per-player variance stalemate detector (replaced by adaptive churn); the original "watch them get smarter" / "watch them win" captions.
**Questions opened:** Sim-vs-render parity validation framework — champion JSONs carrying `expected: {atTick, alive, maxShare}` and a `npm run sanity` script. Discussed in detail mid-session but not built. The session's actual workaround for one specific divergence was the `stillUrl` still-frame mechanism — that's tactical, not the general fix.

## [2026-05-11 | champion-curator | gen2000 trained for stalemate at tick 4000]

**Touched pages:** [[topics/showcase-demo]] [[log]]
**Added:**
- `scripts/train-stalemate.ts` — CPU-only mini-evolver (no GPU). Warm-starts from `public/champions/strong.json` with σ=0.4, pop=16, ~30 gen / 10 min hard cap, mirrors `presimGame()`'s 10Hz step + AI-every-5-ticks loop from `src/demo/runner.ts`. Fitness composite rewards `aliveCount ≥ 2` and `max_share ≤ 60%` at tick 4000.
- `public/champions/gen2000.json` overwritten — solved at gen 1 (40.5s wall-clock): alive=4, max_share=48.5%, fitness=34.00. Same `{weights, generation, bestFitness, savedAt, note}` payload shape as the other champions.
**Updated:** [[topics/showcase-demo]] Champions catalog now lists `gen2000.json` as a real CPU-evolved artifact; the "honest caveat" paragraph distinguishes it from the random-seeded placeholders.
**Retired:** the prior `gen2000.json` placeholder (random `mulberry32(2000)` + `std=0.40`) — fully replaced.
**Questions opened:** none — the trained genome is deterministic per seed but warm-start mutation paths through `gaussian()` mean reruns differ slightly; if the user wants a reproducible regen, the seed `0xC0FFEE_2000` is in the script.

## [2026-05-11 | workspace | python evolution direction: MLX from the jump]

**Touched pages:** [[decisions/python-port]] [[topics/neuroevolution]] [[todo]] [[log]]
**Added:** none new.
**Updated:**
- [[decisions/python-port]] reframed: NumPy parity is the algorithm reference (done); MLX is the compute backend for the evolution loop from the start.
- [[topics/neuroevolution]] Tier 5 / Python bridge subsections drop the "NumPy first, MLX later" framing.
- [[todo]] MLX evolution loop is the active next thread.
**Retired:** none.
**Questions opened:** MLX float32 vs JS float64 — bit-exact parity won't hold; tolerance-based parity is the new invariant for the MLX side.

## [2026-05-11 | workspace | Python port lands (bit-exact JS parity, foundation only)]

**Touched pages:** [[decisions/python-port]] [[topics/neuroevolution]] [[entities/flux]] [[index]] [[log]]
**Added:**
- `python/` — independent reimplementation of the game core and the NN forward pass, managed by `uv`. Modules: `flux/state.py`, `flux/graph.py`, `flux/step.py`, `flux/rng.py`, `flux/genome.py`. NumPy-only; no MLX yet. Parity scenario at `python/tests/test_parity.py` + JS counterpart `python/tests/dump_reference.ts`. Both produce identical SHA-256 hashes every 10 ticks across a deterministic 100-tick run (seeded `mulberry32(42)`, `random_genome(rng, std=2.0)`, `make_initial_state(radius=9, num_players=4)`, two seed `toggleFlow` actions before the loop, a manual toggle at tick 50, NN-driven actions every tick).
- [[decisions/python-port]] — explicit decision page for the bit-exact-parity bridge: filesystem-JSON champion handoff, NumPy-then-MLX strategy, parity invariant covering `step`, `apply_action`, `mulberry32`, `nn_infer_cell` (Float32Array semantics replicated via NumPy `float32` storage + Python `float` arithmetic), `build_neighbor_table` sort, and `ai_think` flow-reconcile order.
**Updated:**
- [[topics/neuroevolution]] — added Tier 5 (offline Python training) and a "Python bridge" section listing what's in `python/` today vs explicit follow-ups (evolution loop, champion JSON I/O, MLX kernels, hot-reload).
- [[entities/flux]] — added `python/` to the implementation frontier.
- [[index]] — added [[decisions/python-port]] under Decisions.
**Mechanics worth pinning:**
- The trickiest port is `nn_infer_cell`. JS stores `h` and `out` as `Float32Array`, which means every `h[j] +=` reads f32 → up-casts to f64 → adds an f64 RHS expression → stores back as f32. The NumPy port matches this by holding `h` / `out` as `np.float32`, but reading each element with `float(...)` (yields Python `float`, IEEE 754 binary64) and writing back as scalars (NumPy quantizes on store). Each accumulation line must mirror JS sum order exactly because float addition is not associative at f64 precision.
- `mulberry32` needs `Math.imul` semantics (32-bit signed multiplication). Python replicates with `(a * b) & 0xFFFFFFFF` then reinterpreting bit 31 as sign. JS `>>> 0` is just `& 0xFFFFFFFF`. Validated indirectly by the parity test — if RNG drifts even once the genome differs and all downstream hashes diverge.
- `build_neighbor_table` JS sort comparator is `(a, b) => pa.x - pb.x || pa.y - pb.y`. Python's `sort(key=lambda nid: (pos.x, pos.y))` is equivalent and stable on the same float tuples.
- `make_initial_state` perimeter seat placement sorts by `Math.atan2(pos.y, pos.x)`. Python `math.atan2(y, x)` matches bit-for-bit on this hardware (libm has been stable for these ranges across CPython 3.12 / V8).
- `ai_think` flow reconcile uses a `Map<src, dst>` in JS and a Python `dict` here. Both preserve insertion order (V8 stable since 7.0, CPython since 3.7), so the action emission order matches without sorting.
**Out of scope / explicit follow-ups:**
- Evolution loop (rtNEAT continuous or batch).
- Champion JSON read/write on the Python side.
- MLX kernels for batched forward pass and step.
- HTTP / filesystem hot-reload of Python-trained champions into the browser.
**Verification:** Parity test passes — all 11 hashes byte-identical between Python and JS. `npm run typecheck` clean. `uv sync` produces a working venv on first run.
**Retired:** none.
**Questions opened:** what does the Python evolution loop look like (rtNEAT in NumPy first then MLX kernels? or straight to MLX with a NumPy fallback?). Deferred until parity foundation is in use.

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
