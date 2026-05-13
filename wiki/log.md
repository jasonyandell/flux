---
title: flux Wiki Log
kind: log
first_seen: bootstrap
last_updated: workspace
status: active
---

## [2026-05-13 | workspace | v2 PRD: design phase done]

**Touched pages:** [[v2-prd]] [[todo]] [[log]]

Pinned the three remaining open questions and the delivery shape; PRD is
implementation-ready.

- **Action encoding: Set/Clear, 13 actions.** K=6 direct hex neighbors.
  Action space = 6 set + 6 clear + 1 no-op. Idempotent, state-independent
  semantics — network doesn't need the current outflow vector as input.
  Chose this over toggle (7 actions) because with K=6 the output-layer
  delta is tiny and the state-independence is the bigger win.

- **CAPTURE_STRENGTH = 50.** Raised from 1 to ~half MAX_STRENGTH. Fixes
  the whip-back where a captured cell with HP=1 dies instantly to the
  previous owner's residual edge pressure. Sized so the new owner has
  a tick or two of breathing room without making captures un-recapturable.

- **Stale targets stay on.** Outflow pointing at a captured friend now
  delivers damage to the enemy receiver. Pure scalar semantics — pressure
  meaning is decided by current ownership of the receiver, not by intent
  tags at the source. Livable only because CAPTURE_STRENGTH was raised.

- **Reward shape, three terms.** `+power_coef·Δ(Σ strength) - waste_coef·waste -
  time_coef`, plus a terminal win bonus. v1's engagement/activity coefs
  are gone — under persistent state a stable loop has every cell active
  permanently and shouldn't be rewarded extra for that. Overkill (excess
  attacker pressure past what was needed to capture) is **not** counted as
  waste; attacker can't know defender strength at commit time. Flagged
  for revisit if overkill dominates observed waste during training.

- **UI is a trainer-displayer, not a simulator.** Plays back `.flxr`
  replays the v2 trainer writes; no in-browser game logic. Same colors
  and layout as the current v1 page, but stripped of the three.js debug
  dropdown. Top bar shows iter/gen and the last ~3 incoming playbacks
  as they arrive. Lives in `src_v2/` so v1's already-packed page stays
  untouched.

Next: implement the pure reducer in `python/flux_v2/` with unit tests
(loops persist; captures respect CAPTURE_STRENGTH; waste accounting
matches algorithm spec). Trainer (`train_v2.py`) and UI (`src_v2/`)
come after the reducer is locked.

---

## [2026-05-12 | workspace | v2 PRD: pressure on first-class edges]

**Touched pages:** [[v2-prd]] [[log]]

After a long day pushing v1 — lookahead-k4, fanout rule, waste penalty,
bidirectional override — the underlying failure mode crystallized:
v1's simulation has **no persistent edge state**. Every AI tick rebuilds
flows from fresh actions, so a loop strategy requires N cells × hundreds
of consecutive correct decisions. PPO can't ladder up to emergent
structures through that combinatorial wall. (Insight credit: "the
spazzy policies weren't doing cancel/re-add, they were just failing to
add every single time.")

The v2 model, pinned in `wiki/v2-prd.md`:

- **Edges are first-class state.** Each directed edge has a `pressure` value
  that persists tick-to-tick. Read last-tick, write new-tick. One-tick lag
  per hop is the propagation mechanism — eliminates same-tick recursion.
- **Multi-outflow per cell.** Cells configure a *set* of active outflows.
  Overflow at MAX splits evenly across the active set, capped per edge.
- **Fill-then-overflow rule, single branch.** Friendly inflow + regen grow
  the cell up to MAX. Any excess overflows out the active outflows.
  Strength only shrinks from enemy damage. No special-case for "maxed"; the
  fanout v1 needed becomes the natural consequence of the same rule.
- **Bidirectional friendly flow impossible by construction.** Mutation
  invariants resolve at AI-tick time (override + higher-index tiebreaker
  for simultaneous bidir).
- **Closed loops correctly leak.** Per-edge cap binds; Σ regen per tick
  becomes waste once edges saturate. Loops with no exit are a player-error
  pattern that the simulation punishes naturally.

Open questions left in the PRD: action encoding (toggle vs. set/clear),
stale-target behaviour, reward shaping under persistent state.

No code touched. v1 training is still running passively in the background
as a baseline; v2 is a fresh codebase track when we pick it up.

---

## [2026-05-12 | workspace | regen-flow gets passthrough + dense shaping + board randomization]

**Touched pages:** [[decisions/regen-flow-rules]] [[decisions/replay-rendering]] [[todo]] [[log]]

The regen-flow ruleset evolved substantially during a live design + training session. Major adds:

**Passthrough (1-tick-lagged friendly inflow → output):**
- A sending cell's *output_capacity* = `regen(s) + passthrough_carry`, where `passthrough_carry` is the friendly support received the *previous* tick (only while the cell was sending; idle cells bank support directly).
- Capped at `MAX_OUTPUT_PER_SEC = 100` per outflow as an insta-kill ceiling.
- Captured cells reset passthrough to 0.
- Implementation: extra `(G, N)` array threaded through `step_batched_regen`. Step now returns 4-tuple. `train_ppo.collect_rollout` uses an adapter so the transfer-flow step keeps its 3-tuple signature.
- Effect: loops become self-amplifying. A chain of sending friendlies pumps strength forward; visualizable as **bright pulses traveling along the chain** (at low playback speeds).

**Dense reward shaping** layered on top of `cell_delta_reward`:
- `engagement_coef = 0.01 × (cells_sending / cells_owned)` — push idle cells off the bench.
- `idle_capped_coef = 0.02 × (idle_near_cap / cells_owned)` — penalize the specific waste of sitting capped while not projecting.
- `output_boost_coef = 0.05 × avg(output_rate / MAX_OUTPUT_PER_SEC of sending cells)` — directly reward "configurations where my output is high", which is exactly what passthrough amplifies. This closes the loop-formation credit-assignment gap that pure cell-delta couldn't.
- Empirically: `mean_total_R` 31 → 44 in 300 iters at r=5/P=3/d=1 with all four terms vs 27 → 27 stuck under cell-delta-only. Entropy commits ~30% faster.

**Distance=1 graph connectivity** as a CLI flag (`--distance 1`, default still 2). Each hex connects only to its 6 immediate neighbors instead of 18. Replay header carries it so the browser rebuilds the same sparse mesh. The GNN still emits 19 action logits per cell; the 12 vestigial slots produce `-1 = no neighbor` and are auto-invalidated. Way cleaner visual, much faster updates.

**Configurable record_stride** (`--record-stride`, default 10). Each replay frame represents N game ticks. Lower = finer playback resolution but more Python object allocation per rollout. Header `tick_stride` matches. The browser auto-speed now adapts to any stride — one recorded frame per browser frame at 60Hz, so `stride=1` is tick-by-tick playback and `stride=10` is a snappier 10× compression.

**Board randomization** for robustness:
- `--num-dead-cells N` marks N random cells as **dead** per game (independent per game in the rollout). Dead cells are untouchable obstacles — flows targeting them are dropped at build time. Replay metadata carries the game-0 dead set in `metadata.dead_cells`.
- `--randomize-starts` puts the P seats at random distinct non-dead cells per game instead of evenly-spaced perimeter.
- GNN input dimension extended **4 → 5 channels**: added `is_dead` so the policy can distinguish "untouchable obstacle" from "capturable empty cell." Breaks checkpoint compatibility with the old 4-channel net — fresh start.

**Render polish:**
- Flow arrows scale visual emphasis by source-cell strength: 1–5 perpendicularly-offset stacked lines (WebGL ignores `linewidth` so stacking is the actual thickness mechanism), shaft reach 50–80% toward dst, arrowhead size 0.7–1.3×, brightness gradient with strength.
- Node base radius dropped 20% (`0.45 → 0.36`) for more visual breathing room.
- Edge contrast bumped (`0x1a1a1a → 0x2a3548`) for phone-screen visibility.

**Player change:** plays current replay to the last frame before swapping. Newer replays are queued via `pendingFile` and loaded at end-of-replay. Lets you watch full games end-to-end.

**Files touched (uncommitted before this entry):**
- `python/flux/ppo.py` — `IN_DIM 4→5`, `build_features` takes optional `dead_mask`, `forward` accepts `dead_mask`.
- `python/flux/mlx_step_regen.py` — passthrough threading + `MAX_OUTPUT_PER_SEC` cap.
- `python/flux/mlx_batch.py` — `build_flows_from_actions` accepts `dead_mask` and drops flows landing on or originating from dead cells.
- `python/scripts/train_ppo.py` — `--distance`, `--record-stride`, `--engagement-coef`, `--idle-capped-coef`, `--output-boost-coef`, `--num-dead-cells`, `--randomize-starts` flags. `collect_rollout` returns `(rollout, frames, rng_key, dead_mask_np)`. `Rollout` dataclass gains `dead_mask: (G, N)`. PPO update tiles dead_mask across T for minibatch slicing.
- `src/render/scene.ts` — stacked-line flow rendering, smaller node radius.
- `src/replay/player.ts` — full-game playback, `setIndexUrl`, `tickStride()/dtPerTickMs()` accessors.
- `src/main.ts` — generalized auto-speed for any tick_stride.

**Active training run** at end of session: `ppo-regen-r5-p3-d1-rand5dead` (r=5, P=3, d=1, G=8, max_ticks=3000, 5 dead cells per game, random starts, all dense-shaping terms active, regen-flow + passthrough). Fresh policy.

**Retired:** none. Transfer-flow ruleset still works under `--ruleset transfer`.

**Questions opened / still open** (see [[todo]]):
- Dead-cell visualization in the browser (data is in metadata; renderer isn't reading it).
- True overage cap beyond `MAX_OUTPUT_PER_SEC` per outflow.
- Browser live-play wiring for regen-flow.

## [2026-05-12 | workspace | regen-flow ruleset shipped as a second game]

**Touched pages:** [[decisions/regen-flow-rules]] [[index]] [[todo]] [[log]]

**Added:**
- `src/game/step_regen.ts` — TS reference implementation. Same `GameState` / `Flow` shape; new step semantics. Linear regen scaling, sender forfeits regen, symmetric damage, deterministic capture at strength=1.
- `python/flux/step_regen.py` — Python parity mirror.
- `python/flux/mlx_step_regen.py` — batched MLX kernel for training. Reuses `build_flows_from_actions` (flow tensor shape unchanged).
- [[decisions/regen-flow-rules]] — new decision page covering mechanics, strategic consequences, file map, and what still needs deciding (overage propagation through caps, K>1 in training, reward shaping).

**Updated:**
- `python/scripts/train_ppo.py` — accepts `--ruleset {transfer, regen-flow}`. Dispatches `step_fn` based on the flag; default checkpoint becomes `python/checkpoints/ppo-regen/latest.npz` when the new ruleset is selected. Replays land at `public/replays/train_ppo_regen_*.flxr` and the metadata gains `ruleset: "regen-flow"`. Index entries gain the same field so the browser can distinguish.

**Why:** Across v1/v2/v3/PPO it became clear the transfer-flow model conflates "I'm sending strength" with "I'm losing strength." That made loops self-defeating (every member bleeds) and made symmetric attack/defense ambiguous. Regen-flow separates the two: sending forfeits regen but doesn't drain health, damage is symmetric, and `regen(s)` scales linearly with strength (slope 2.0) so big idle cells fatten faster than they can project. Loops with heterogeneous member strengths pump strength downstream — the "loops gain energy" insight is now actually true.

**Initial numbers** — `ppo-regen-r5-p3` (radius 5, 3 seats, G=8, max_ticks=3000, tick-by-tick replays):
- Iter time ~3.5s (vs ~5s at radius 9). Smaller board, faster.
- `mean_total_reward = 29.33` (vs the structural 21.58 on transfer-flow self-play).
- `explained_variance` climbs `0.20 → 0.78` in 8 iters. The value head converges fast at this scale.
- `pol_loss` consistently negative (~-0.1) — policy moving in the advantage direction.

**Mechanics worth pinning:**
- `regen(s) = 0.5 · (1 + 2·(s − 1))` — linear in strength. At s=1, regen=0.5; at s=10, regen=9.5.
- A sending cell with K outflows: each outflow delivers `regen(s)/K · dt`. K>1 supported by the TS path but never produced by PPO training (one outflow per cell per action).
- Capture is deterministic strength=1, no inheritance of attacker overage.
- Overage propagation through caps not yet implemented — currently discarded.
- Replay binary format unchanged. Only metadata JSON gains the `ruleset` field. Older replays implicitly = `"transfer"`.

**Retired:** none. Transfer-flow stays the live-browser default and the historical replay record.

**Questions opened:** see [[todo]] § Open AI/evolution — overage propagation, browser live-play wiring, retirement of the transfer-flow PPO path.

## [2026-05-11 | workspace | PPO + GNN trains end-to-end; greatest-hits replay cycle; full wandb instrumentation]

**Touched pages:** [[decisions/ppo-gnn]] [[decisions/replay-rendering]] [[todo]] [[index]] [[log]]

**Added:**
- `python/flux/ppo.py` — `GNNActorCritic` (2-layer GCN policy + value head). Constants: `NEIGHBOR_STRIDE=18`, `IN_DIM=4`, `HIDDEN=32`, `POLICY_OUT=19`, `VALUE_HIDDEN=16`. Value pools second MP layer's activations over each seat's owned cells.
- `python/scripts/train_ppo.py` — main entry. PPO rollout collection (G parallel games, T AI ticks per rollout), GAE-λ advantages, clipped-surrogate + value MSE + entropy loss, Adam autograd via MLX. Auto-resume from `python/checkpoints/ppo/latest.npz`.
- `python/scripts/build_greatest_hits.py` — scans `.flxr` headers, filters fitness > 0 and `num_frames ≥ 200`, writes `public/replays/greatest-hits.json` (top 30, longest-first).
- `public/replays/greatest-hits.json` — curated list for the browser cycle.
- [[decisions/ppo-gnn]] flipped from `planning` to `active` with the actual file map, perf numbers, instrumentation panel, and current observation.

**Updated:**
- `python/flux/mlx_batch.py` — added `build_flows_from_actions(actions_all, owner, graph_neighbors)` for rollout-mode flow construction from sampled actions.
- `python/scripts/train_ppo.py` — Frame-recording site now pulls game-0 `flow_src/dst/player/valid`, builds Python `Flow` objects, and stores them in the recorded `GameState`. Replays render directional arrows.
- `src/replay/player.ts` — `setIndexUrl(url)` swaps the active index (e.g. `replays/index.json` ↔ `replays/greatest-hits.json`) on the fly; resets `entriesCache`, `replayName`, `replay`, `pendingFile`, `frameIdx`, `frameAccSec`, `lastPoll`.
- `src/main.ts` — `greatestHits` tunable + lil-gui toggle. In greatest-hits mode, auto-speed targets ~2s per replay (capped at 500×). Camera snaps to origin on every replay swap. `rebuildSceneGeometry` now runs on every replay swap (not only on node-count change) so stale geometry can't leak across boards.
- `src/render/scene.ts` — `rebuildSceneGeometry` early-return guard removed; also calls `nodeInstanced.dispose()` to release per-instance attribute buffers. Edge color bumped `0x1a1a1a → 0x2a3548` (visible on phone screens). Flow rendering: 3 line segments per flow (shaft + two arrowhead wings) at z=0.3 with gradient — 25% brightness at the source cell, full brightness at the arrowhead tip.
- [[decisions/replay-rendering]] gained sections on the greatest-hits cycle, flow-arrow render shape, and the scene-rebuild bug fix.

**Performance — PPO iter time:**

| stage    | baseline | post-fix |
|----------|----------|----------|
| rollout  | 4.5s     | 2.5s     |
| update   | 17.9s    | 2.5s     |
| total    | ~22s     | ~5s      |

**4.7× speedup.** Wins:
- `--update-epochs 4 → 2` (the bulk of it).
- Coalesced `mx.eval` calls in `collect_rollout`: single `mx.eval(logits, value, actions, owner, strength)` per AI tick instead of four separate evals.
- Hoisted `seat_mask` onto GPU once per update (was rebuilt per minibatch).
- Metric side-channel: `loss_fn` appends `(policy_loss, value_loss, entropy, approx_kl, clip_fraction, ratio_mean, ratio_max)` to a Python list; main loop pops them after `grad_fn` returns and evaluates alongside the gradients. No redundant forward pass.

`mx.compile` of the train step measured ~30% more in isolation by a perf subagent — not landed yet, current speed is acceptable.

**Wandb instrumentation (full panel):**
- PPO update health: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `ratio_mean`, `ratio_max`, `grad_norm`, `weight_norm`.
- Value head: `explained_variance` (key metric), `value_mean`/`std`, `return_mean`/`std`.
- Raw reward: `reward_step_{mean,std,max,min}`.
- Behaviour: `action_entropy`, `action_pick_top_frac`, `action_self_frac` (fraction picking the "no flow" action 18).
- Outcomes: `cells_{max,min}_end`, `dominance`, `alive_seats_end`, `neutral_frac_end`.
- Image every 20 iters: `end_state` ownership grid (one row per game in the rollout).

Pillow added as a dep for `wandb.Image`. The image emission is wrapped in `try/except` so a missing PIL won't kill training.

**Numbers worth pinning:**
- At iter ~196, `ppo-r9-ep2-instrumented` run: `explained_variance ≈ 0.74`, `entropy ≈ 2.92` (max log(19) ≈ 2.94), `mean_total_reward = 21.58` (pinned to a structural symmetry constant in self-play). Value head IS learning; policy hasn't started committing yet.
- Older PPO replays (before flow-recording fix) have no arrows when rendered. The greatest-hits.json from before the fix is in this state. Future replays will carry flows.

**Mechanics worth pinning:**
- MLX first-iter kernel compile at `max_ticks=5000` is **substantial** (1–2 min). Don't kill a hung-looking PPO process under 2 minutes.
- Under `uv run python ...` with redirected stdout, the script's `print()` is block-buffered. Set `PYTHONUNBUFFERED=1` (or use `python -u`) to see live iter timings.
- `bash` cwd drifts to `flux/` repo root between commands; PPO must be launched from `python/` — prefix with `cd /Users/jason/code/flux/python &&` to be safe.
- Greatest-hits is not a separate UI mode beyond the toggle — the player walks `entries[curIdx+1] % len` regardless of which index it loaded. The toggle just swaps the URL.

**Retired:** none.

**Questions opened:** none new. The "PPO policy commitment" entry on [[todo]] is the active open thread.

## [2026-05-11 | workspace | MLX training pipeline ships end-to-end; replay rendering becomes the browser default; v2 wider-vision model joins v1]

**Touched pages:** [[topics/neuroevolution]] [[decisions/python-port]] [[decisions/replay-rendering]] [[decisions/v2-vision]] [[index]] [[todo]] [[log]]
**Added:**
- `python/flux/mlx_step.py` — single-game MLX `step` + `apply_action`.
- `python/flux/mlx_batch.py` — batched (G × S × N) MLX step + NN forward + vectorized AI tick. `build_flows_batched` (v1) / `build_flows_batched_v2` (v2) build dense (G, N) flow tensors on GPU per AI tick — one batched NN forward → per-cell owner-action argmax → aggressive overlay → flow tensor. No per-game Python flow-reconcile loop.
- `python/flux/mlx_genome.py` — v1 layout constants (`IN=91`, `HID=32`, `OUT=19`, 3571 weights).
- `python/flux/mlx_genome_v2.py` — v2 layout constants (`IN=181`, `HID=32`, `OUT=19`, 6451 weights).
- `python/flux/vision.py` — 3-hop neighbor table for v2, `STRIDE_V2 = 36`.
- `python/flux/game_loop.py` — `play_batch_games`. Runs G games in parallel under MLX; per-game terminate when alive ≤ 1; 10k tick hard cap.
- `python/flux/evolve_mlx.py` — `run_one_batch`, tournament selection, gaussian mutation, checkpoint, champion JSON writeout.
- `python/flux/replay.py` — `.flxr` binary writer (header + sampled-frame body).
- `python/scripts/train.py` — CLI. `uv run python scripts/train.py --model {v1,v2} --games-per-batch N`. Other flags: `--pop`, `--ticks`, `--ai-period-ticks`, `--checkpoint`, `--champion-dir`, `--fresh`, `--aggressive-seat`. Auto-resume on startup from `python/checkpoints/{latest.npz | v2/latest.npz}` unless `--fresh`.
- `src/replay/format.ts` — TS-side `.flxr` parser (mirrors `python/flux/replay.py`).
- `src/replay/player.ts` — browser replay playback. Wall-clock → frame index; live sim path is dormant in replay mode.
- `public/replays/*.flxr` — written by `train.py`. `public/replays/index.json` auto-prunes to 50 entries.
- `python/checkpoints/{latest.npz, v2/latest.npz}` — checkpoint state per model. Restarting `train.py` resumes from there.
- `public/champions/v2/*.json` — v2 champion JSONs for the browser (all-time bests only).
- [[decisions/replay-rendering]] — new decision page: Python is the lab, web is the replay player, `.flxr` is the contract. Browser default mode is "watch replays" via lil-gui toggle; top bar reads `[model] gen N · best F`. The in-browser WebGPU evolution coexists as a side-quest but is no longer the training path.
- [[decisions/v2-vision]] — new decision page: 3-hop receptive field experiment. v1 kept intact; v2 trains in parallel. Cost ~1.8× weights; early result is v2 caught v1's fitness in ~6.5% of v1's generation budget.

**Updated:**
- [[topics/neuroevolution]] — heavy revision. Tier 5 is no longer "next thread"; it's *the* training path. New file map under `python/flux/`. Documents v1 vs v2, win + tick-cap termination, the vectorized AI tick win, the G knob with bench numbers.
- [[decisions/python-port]] — "what's NOT in scope" section cross-references the landed MLX pipeline (`evolve_mlx.py`, champion JSON writes, `.flxr` bridge). The page stays scoped to the parity foundation.
- [[index]] — new pointers for [[decisions/replay-rendering]] and [[decisions/v2-vision]]; the neuroevolution one-liner now says MLX is the training path; the webgpu-evolution line notes the coexistence.
- [[todo]] — MLX-evolution-loop and Python-pipeline-forks items ticked off into a new "Done — landed" section. New active threads: v2 saturation run, league/pool sampling. ANE-for-deployed-champions stays pending; gating item is now "champion worth packaging" (which v2 saturation will tell us).

**Numbers worth pinning:**
- v1: gen 5372, all-time-best fitness 1540.50. Beats hand-coded aggressive consistently.
- v2: gen 347 (started fresh today), all-time-best fitness 1521.50. Caught v1 in ~6.5% of v1's generation budget.
- Perf (today): G=1 went from ~12 gens/min → ~22 gens/min (~80% speedup) after vectorized AI tick + win-cadence relax (50 → 250 ticks) + deferred frame recording (stash `mx.array` refs, bulk-eval at end — avoids per-frame `mx.eval` sync barriers) + async I/O for checkpoint/replay/index writes via `ThreadPoolExecutor(1)`.

**G knob bench (radius 18, 12 seats, 10k tick cap, ai_period 5):**

| G | per-batch | gens/min | samples/genome/batch |
|---|-----------|----------|---------------------|
| 1 | 0.65s (loop) / 2.6s (train.py with writes) | 22 (new) | 0.46 |
| 2 | 1.0s | 60 | 0.92 |
| 4 | 1.4s | 43 | 1.83 |
| 8 | 2.3s | 26 | 3.67 |
| 24 | 11.0s | 5 | ~11 |
| 128 | 113.6s | 0.5 | ~59 |

G=4 was the production sweet spot pre-optimizations. G=1 is the current choice with the sync+async wins (max generation rate, accept noisier fitness signal). Memory bandwidth (features tensor scales G × S × N × 91 for v1, × 181 for v2) makes G > 24 throughput-counterproductive.

**Mechanics worth pinning:**
- The vectorized AI tick is the main perf win. The JS-style per-game Python flow-reconcile loop is eliminated: each AI tick is one batched NN forward → owner's action chosen per cell → aggressive overlay → dense flow tensors built on GPU. Lives in `build_flows_batched` / `build_flows_batched_v2`.
- The aggressive seat is hand-coded in MLX (vectorized argmin over non-friendly neighbor strength). Narrative anchor: every batch contains aggressive opponents the population must beat.
- Win-based termination per game (alive ≤ 1 → freeze the game in the batched tensor; other games keep ticking). 10k tick hard cap.
- v2 only widens **vision** (181 inputs, 36 neighbors via 3-hop). Output stays 19 — flows still travel only over distance-2 edges.

**Retired:** none. WebGPU evolution still works; it's a side-quest now, not deprecated.

**Questions opened:**
- v2 saturation behavior. Caught v1 fast — does it pass v1, match it, or plateau below? Open.
- Whether league-style champion pool sampling is worth wiring into `evolve_mlx.py` (per-model vs shared pool; aggressive-seat interaction).
- Whether a v3 with deeper hidden width is the next step once vision is wide.

## [2026-05-11 | workspace | three skills for next-session iteration]

**Touched pages:** [[topics/showcase-demo]] [[log]]
**Added:** none in-repo. Three Claude Code skills landed in `~/.claude/skills/` (cross-project, not committed here):
- `flux-demo` — project-specific. The 4-file ripple, caption conventions, churn knobs, cinema chrome, team patterns for editing this demo.
- `presim-playback` — pattern. The pre-sim → snapshot → wall-clock playback architecture, sequential pre-sim discipline, hot-area framing, adaptive churn truncation, `stillUrl` escape hatch. Reusable for any pure-`step()` simulator.
- `mini-trainer` — pattern. CPU-only headless evolution trainer scaffold (tsx) targeting a custom fitness function, warm-started from a saved genome.
**Updated:** [[topics/showcase-demo]] gains an "Iteration shortcuts" section pointing at the three skills so future-Claude knows the shortcuts exist before re-deriving them.
**Retired:** none.
**Questions opened:** none. The skills encode the sim-vs-render parity discussion as an open question but don't yet implement it — that's still worth doing as `expected: {atTick, alive, maxShare}` JSON contract + `npm run sanity` validator.

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
