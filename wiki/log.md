---
title: flux Wiki Log
kind: log
first_seen: bootstrap
last_updated: 2026-05-16
status: active
---

## [2026-05-16 | workspace | v2 viewer pinch zoom, pan, and wheel speed]

**Touched pages:** [[topics/v2-viewer]] [[index]] [[log]]

Added Mac-friendly replay inspection controls to `/index-v2.html`: trackpad
pinch (`ctrl+wheel` in Chromium/Electron) zooms the orthographic camera around
the cursor, smooth two-finger trackpad scroll pans the map with camera clamps,
and Shift-scroll steps playback speed up or down through forward, slow, `0×`,
and reverse multipliers. `0×` is a stop barrier: scrolling down from positive
speed pauses at zero; a separate follow-up scroll crosses into reverse, and
vice versa. Negative speed plays the current replay backward until frame 0.
The replay drawer keeps normal scroll behavior because the wheel handler is
canvas-scoped.

## [2026-05-16 | workspace | r40 smooth replay probes]

**Touched pages:** [[topics/v2-viewer]] [[log]]

Launched and verified two large solver replay probes for the upgraded v2
viewer: `radius=40`, `num_players=12`, `num_dead_cells=720`,
`edge_alpha=0.05`, `max_ticks=9000`, `record_stride=1`, carved connectivity.
The frontier duel
`solver_v2_lightning_sum_throttled+lightning_wave_long_ea0p05_20260516T210154.flxr`
ended with `lightning_sum_throttled` seat 8 winning at tick 6794. The
morphology mix
`solver_v2_lightning_chase+lightning_sum_wave+lightning_vortex_ea0p05_20260516T210154.flxr`
ended with `lightning_sum_wave` seat 11 winning at tick 4921.

## [2026-05-16 | workspace | v2 viewer run-management cleanup]

**Touched pages:** [[topics/v2-viewer]] [[index]] [[log]]

Upgraded `/index-v2.html` from a simple replay playlist into an
agent-friendly run viewer. Added URL-addressable runs via
`?replay=<file.flxr>`, keeping the URL synchronized when the current replay
changes. Replaced the drawer with a searchable/filterable replay browser
(All / Solver / Train / Fluid / New / Current), per-row copy-link buttons, a
new-arrival badge driven by `public/v2/replays/events.jsonl`, and inert
closed-state behavior so hidden rows cannot catch clicks. Added a compact
current-run header with metadata and a current-link button.

Solver replay publishing now records richer metadata in both the `.flxr`
header and index entry: seed, board/run shape, seat solvers, edge alpha,
winner/leader, final ticks, alive count, dominance, and final cell counts.
`append_index` now also appends replay-added events to `events.jsonl`.

## [2026-05-16 | workspace | fluid EDGE_ALPHA keeps champ and makes loop-release real]

**Touched pages:** [[topics/v2-edge-loop-emergence]] [[topics/v2-ml-gameplay-opportunities]] [[log]]

Added `scripts/loop_release_probe.py`, a tiny A/B/C friendly triangle plus
enemy target micro-scenario that compares direct feed against charged-loop
release while sweeping `EDGE_ALPHA`, `MAX_EDGE`, regen, max strength, target
strength, charge duration, and release mode. Result: the current rules already
contain a loop-release advantage, but it is much clearer when pressure has
edge memory. At `EDGE_ALPHA=0.05`, charged-loop divert captures targets that
direct pressure has only partially damaged within the same release horizon.
`MAX_EDGE=50-100` is enough to preserve the effect; no evidence yet says to
change regen or max strength.

Plumbed `--edge-alpha` into `eval_solvers.py` and added the alpha value to
solver replay filenames so same-seed alpha sweeps do not overwrite each other.
Compact R=18 P=6 dead=60 sweeps kept `lightning_sum_throttled` decisive at
all tested alphas (`1.0`, `0.2`, `0.1`, `0.05`) and ahead in matched pairs
against `lightning_wave_long`, `lightning_sum`, and `lightning_live`.
Lower alpha slows games and gives better temporal morphology without
invalidating the solver frontier.

## [2026-05-16 | workspace | throttled-sum clone/probe fails as full policy]

**Touched pages:** [[topics/v2-training-runs]] [[topics/v2-ml-gameplay-opportunities]] [[topics/v2-temporal-strategy]] [[log]]

Confirmed the next teacher should be `lightning_sum_throttled`, not
`sum_long`: vectorized matched-pair rechecks kept throttled-sum ahead of
vanilla `sum` and `bfs`, while `sum_long` leaned worse than vanilla `sum`.
Added `lightning_sum_long` and `lightning_sum_throttled` to
`train_v2.py` warmstart/opponent solver choices, added `--pretrain-only`
and `--pretrain-nonnoop-weight`, fixed warmstart collection for the current
`tick_batched` return, and let `run_v2_solver.py` evaluate edge-model
checkpoints plus argmax decoding. Also added `scripts/target_spell.py` for
coarse target-spell completion/abandonment diagnostics.

Negative result: GNN and edge-aware clones of `lightning_sum_throttled`
failed as deployed policies on R=5 P=6 dead=10. The best BC probes still got
0 wins against the teacher in 24-game alternating-seat evals; PPO fine-tunes
from those clones collapsed entropy near zero and produced huge KL spikes.
Conclusion: do not keep nudging vanilla PPO here. Next training surface should
supervise solver intent/desired edge masks directly, use DAgger-style
teacher labels on clone-induced states, or learn a residual over
`lightning_sum_throttled`.

## [2026-05-16 | workspace | hard target-hysteresis prototype loses to throttled-sum]

**Touched pages:** [[topics/v2-temporal-strategy]] [[topics/v2-algorithmic-solvers]] [[log]]

Registered two stateful experiment solvers in
`python/scripts/run_v2_solver.py`: `lightning_sum_throttled_sticky` and
`lightning_sum_long_sticky`. Each wraps the base Lightning solver with
per-seat hard target hysteresis: latch the first modal enemy receiving
outgoing attack pressure, then mask other enemy seats as blocked until the
target has no cells. Smoke compile and run passed, but the result is a
negative control. At R=20 P=8 dead=80 max_ticks=6000, 12 matched pairs,
`lightning_sum_throttled` beat `lightning_sum_throttled_sticky` 8/8
coherent pairs (p≈0.0078, 20-4 raw games). The smaller R=12 P=6 smoke
also did not favor sticky variants. The next version should be a soft
target bias or switching cost, not an all-or-nothing mask.

## [2026-05-16 | workspace | v2 ML gameplay opportunities synthesis]

**Touched pages:** [[topics/v2-ml-gameplay-opportunities]] [[index]] [[log]]

Added an ML-scientist synthesis page for the current v2 gameplay frontier.
The read: the strongest near-term ML role is to exploit the mechanistic
solver discoveries rather than replace them from scratch. Use
`lightning_sum_long` / `wave_long` as teachers and baselines, learn residuals
over long-field flow, make edge semantics the representation surface, and
turn cheap matched-pair tournaments into preference data. The page also calls
out risk areas: single-direction tournament noise, terminal reward visibility,
activity rewards, capacity-first PPO changes, and stale todo framing.

## [2026-05-16 | workspace | v2 vectorized lands on main — ~24× at R=30, ~36× at R=100]

**Touched pages:** [[topics/v2-vectorized]] [[index]] [[log]]

Branch `worktree-v2-vectorize-compact` merged. The v2 hot path is now
Numba-JIT'd top-to-bottom: vectorized solver pipeline (all 17 lightning
modes + BFS in one (N, K) shape), JIT'd `step.tick` /
`step.apply_actions`, JIT'd `compute_potential` Bellman value iteration
with warm-start across AI ticks under fluid `EDGE_ALPHA`, JIT'd
board-setup BFS (the unexpected ~1.16 s/game pure-Python culprit), and
a batched per-AI-tick solver call. Plus FLXR v3 replay format —
gzip-compressed dense per-frame, ~43–180× smaller than v2 — and an
`EDGE_ALPHA` momentum knob on the reducer that gives edges memory and
roughly halves the stalemate rate at `alpha=0.05`.

Headline numbers, R 6-seat all-`lightning_sum_long` fluid:

  R=20 (decisive)   pre-vec ~9.7s   new 0.7s     ~14×
  R=30 6000-tick    pre-vec 31.3s   new 1.3s     ~24×
  R=60 6000-tick    pre-vec ~140s   new 5.1s     ~27×
  R=80 6000-tick    pre-vec ~280s   new 8.8s     ~32×
  R=100 6000-tick   pre-vec ~500s   new 13.9s    ~36×

Pre-vec column is measured for R≤30 and extrapolated for R≥60 (game
loop linear in N, pre-vec board setup O(N²) Python BFS — the speedup
*grows* with N because the JIT'd path stays linear on both).

Parallel: 10× R=100 6000-tick games in 15.3 s on `--workers 10`
(1.53 s/game amortized; M5 Max 12 P-cores saturated).

**Added/updated:**
- [[topics/v2-vectorized]] — the canonical writeup. Status flipped to
  active. Contains the full perf table, the EDGE_ALPHA pilot, the
  Numba pass, the warm-start trick, the batched pipeline, the JIT
  board-setup, and a frank "MLX wired in lost vs Numba" finding so
  future tinkerers don't re-invent the dead end.
- [[index]] — `v2-vectorized` is now the current hot path entry in
  the "v2 research" list, no longer flagged provisional.

**Open item:**
- [[topics/v2-overnight-research]] rankings (`wave_long > sum > bfs ≈
  max >> attn …`) were produced against the per-cell-loop solvers.
  The two intentional semantic deltas in the new pipeline (RNG draw
  schedule via per-cell rotation offset; ε-tie relay rule now
  globally "within `fanout_eps` of max friendly pot AND above
  pot[c]") are individually small but the rankings live inside a
  ~6 pp seat-bias noise floor. A matched-pair rerun under the new
  code is the next thing to do before that page's official rankings
  are refreshed.

## [2026-05-15 | workspace | throttle hypothesis validated — lightning_sum_throttled dominates both bfs and sum]

**Touched pages:** [[topics/v2-temporal-strategy]] [[log]]

Direct test of the throttle hypothesis from the morning synthesis.
Built three pieces: `python/scripts/eval_solvers.py` (matched-pair
tournament + Wilson CI + sign test), `python/scripts/switch_rate.py`
(modal-target switch-rate diagnostic), and a `throttle` kwarg on
`lightning_solver_actions` that caps desired outflow slots to top-N by
`pot[d]` (attack tier > relay tier). New solvers wired in:
`lightning_sum_throttled` and `lightning_max_throttled`
(throttle=1 on sum and max potentials respectively).

**Matched-pair results at R=25 P=12 dead=200 max_ticks=12000, 20 pairs each:**

- `lightning_sum_throttled` vs `lightning_sum`: 9/9 coherent
  pairs, p ≈ 0.0039, +100pp coherent advantage. Raw 29-11 (72.5%).
- `lightning_sum_throttled` vs `bfs`: 7/7 coherent, p ≈ 0.0156,
  +100pp. Raw 27-13 (67.5%).
- `lightning_sum` vs `bfs`: 5/5 coherent, p ≈ 0.0625, +100pp
  (borderline). Raw 25-15 (62.5%).

**Headline:** throttled-sum dominates both alternatives. The
"commitment by construction" prior from bfs *combined with*
sum-mode's loop-aware potential is a real improvement over either
component alone.

**Caveat / inversions from prior memory:** at this config (40% dead,
6-vs-6 head-to-head) lightning_sum beats bfs — the *opposite* of the
multi-strategy FFA finding. So bfs's "only solver that converts" was
a 21-strat-FFA property, not a head-to-head property; in 2-strategy
6v6 splits each side has clear directional targets and lightning_sum's
loop-friendly field wins. This narrows the regime where bfs is
distinctively good (multi-target asymmetric cleanup) and broadens the
regime where throttle adds value (apparently everywhere tested).

**Updated:** [[topics/v2-temporal-strategy]] now has a "Validation
experiments (2026-05-15)" section recording these results. Three
follow-up directions flagged: multi-strategy FFA test (does throttle
become the new big-zoo converter?), throttle sweep (1/2/3/6, sharp
cliff or smooth gradient?), waste-attribution diagnostic (separate
the throttle effect into its components).

**Switch-rate diagnostic was null at self-play.** At R=20 P=8 and
R=25 P=12, bfs and lightning_sum both show ~0.55-0.66
switches/100t — no gap. Throttled is *higher* (0.88-0.93), not
lower. Re-interpretation: the user-observed dithering plays out at
500-1000-tick strategic timescale (kill, switch, regen, return), not
AI-tick scale. The right diagnostic is "target-spell-completion
ratio" not switch rate. Also, symmetric self-play cannot reproduce
the asymmetric "one dominant seat vs many fragments" condition where
the pathology was actually seen.

**Revised recommendation:** BC target should be
`lightning_sum_throttled`, not bfs. (Updated in
[[topics/v2-temporal-strategy]].)

## [2026-05-15 | workspace | ML-scientist synthesis — throttle + targeting + temporal as one options problem]

**Touched pages:** [[topics/v2-temporal-strategy]] [[topics/v2-edge-voting-policy]] [[index]] [[log]]

Discussion-driven synthesis. Three v2 pathologies — pressure/waste
cliff (too little pressure stalls, too much wastes — bfs converts
where lightning variants stall in multi-strategy FFAs), multi-target
dithering observed
live in an R=30 P=21 game (dominant orange seat cycling across 5+
fragmented enemies and finishing none), and the user's strategic
vocabulary (migrate / spread / flank / chase / corner) — are the same
problem in three costumes: the policy reacts to instantaneous state
when it should commit to a multi-tick plan.

**Added:** [[topics/v2-temporal-strategy]] frames this as the options
framework (Sutton-Precup 1999) / goal-conditioned RL (UVFA, HER) /
FeUdal Networks shape that AlphaStar and OpenAI Five solved. Maps each
strategic concept to a measurable graph quantity (centroid shift,
edge-activation variance, attack-vector angle, pressure-on-enemy
tracking, neutral-frontier perimeter delta) usable simultaneously as
auxiliary task, intrinsic reward, and diagnostic. Names the
multi-target failure as bandit-with-switching-costs (Banks-Sundaram)
where the optimal policy is piecewise constant and greedy is provably
suboptimal at K≥3. Architectural prescription: slow recurrent manager
emits `(target, throttle, intent_vector)` with KL-regularized
self-hysteresis; fast worker is goal-parameterized `lightning_sum` or
a learned edge-voting head. Concrete first moves in cheapness order:
log modal-target switch rate (zero-model diagnostic), hard-cap
`lightning_sum` to one open slot per cell (tests whether bfs's edge
is throttling not targeting), behavior-clone *bfs* (not
`lightning_sum`) since bfs has commitment + throttle by construction,
then KL-regularized PPO from that warm start.

**Updated:** [[topics/v2-edge-voting-policy]] now scoped explicitly to
the spatial layer with a pointer up to [[v2-temporal-strategy]] for
the temporal layer they share. [[index]] adds the new topic.

**Retired:** none.

**Questions opened:** none new; the multi-enemy dithering pathology
gets a named diagnosis (bandit-with-switching-costs) and an
architectural answer (manager/worker with hysteresis), pending
implementation.

## [2026-05-15 | workspace | v2 viewer — wall-clock smoothing + 80/20 fade/pressure mix on node brightness]

**Touched pages:** [[topics/v2-viewer]] [[log]]

Refined the v2 fade trail. The iter-based `freshness` is now the
*target*; a new per-node `displayed` slews toward it at a wall-clock
cap of `FRESHNESS_RATE_PER_SEC = 2.0/s`, so any single 0↔1 transition
takes ≥ 0.5 s. Under fast playback or scrub-bursts the displayed value
never reaches either extreme — it orbits the mean, producing a
continuous soft pulse instead of hard flashes. At normal cadence (per-
iter change within budget) behavior is visually unchanged.

Rendered brightness now blends two signals over the above-base
headroom: 80% `displayed` (age-delta), 20% `pNorm` (max edge pressure
touching node, per-frame auto-scaled the same way the arrow widths
are). Max pressure on a long-static node tops out at 20% over base;
max pressure plus a fresh change hits full bright. Idle nodes fall to
base. Strength still drives node size — color is the activity layer
on top.

Updated: [[v2-viewer]]'s "Node fade trail" section rewritten to cover
the three-stage target/displayed/render pipeline and the mix.

## [2026-05-15 | workspace | v2 viewer — per-node fade trail keyed to replay iters]

**Touched pages:** [[topics/v2-viewer]] [[log]]

Added a brightness pulse-and-fade to v2's node rendering. A node snaps
to full brightness whenever its owner or flow-membership signature
changes, then decays by `1/20` per replay frame-index advance —
fully dim ~20 iters after its last config change. Floor is
`MIN_BRIGHTNESS = 0.25` of the owner's base color. Pressure changes
deliberately don't pulse (continuous → every frame would flash).

Keyed to iters, not wall-clock: pause holds the glow, scrubbing back
doesn't silently fade, fast-forward burns through trails at the
forward-stepping rate. Frame-index delta handles all three.

User-facing toggle (`✦`/`✧` on the transport bar, default on,
persisted in `localStorage` under `flux-v2-fade-enabled`) flips a
`fadeEnabled` flag on `Scene`; when off every node renders full
brightness regardless of freshness. Knobs: `FADE_PER_ITER` and
`MIN_BRIGHTNESS` in `src_v2/render/scene.ts`.

Updated: [[v2-viewer]] gained a "Node fade trail" section and a
fade-button entry in the transport-bar list.

## [2026-05-15 | workspace | wiki curation — entry-page rewrite + ranking reconciliation]

**Touched pages:** [[entities/flux]] [[index]]
[[decisions/webgpu-evolution]] [[topics/neuroevolution]]
[[topics/v2-edge-voting-policy]] [[topics/v2-edge-loop-emergence]]
[[questions/open]] [[log]]

Driver: `entities/flux.md` was the route-map entry point per `index.md`
but described only v1. With all the v2 work (PPO, edge-pressure, Lightning
solver family, trainer-displayer at `/index-v2.html`, overnight research),
new readers were being routed to a page that read as if v2 didn't exist.

**Added:** none new.

**Updated:**
- [[entities/flux]] rewritten as a v1+v2 project overview. Top sections:
  "What flux is now," "v2 frontier (read this)," "v1 surface (still
  deployed, not the frontier)." Implementation map covers
  `python/flux_v2/`, `src_v2/`, `python/scripts/`, and the v1 surface.
- [[index]] route map regrouped: Topics split into Context / v2 reference
  / v2 research (freshest first). Decisions split into v2 / cross-track /
  v1 / earlier-experiments / retired. The stale `lightning_attn 12-12 tie
  with sum` line is replaced by the matched-pair ranking. Index now points
  v2-curious readers straight at
  [[topics/v2-rules-one-pager|v2-rules-one-pager]] +
  [[topics/v2-overnight-research|v2-overnight-research]].
- [[decisions/webgpu-evolution]] status flipped to `superseded` with a
  "Status (2026-05)" preamble. Still kept live as the parity reference and
  the home of the deployed `evolved` seat.
- [[topics/neuroevolution]] Tier 5 split: v1 MLX path marked saturated;
  added Tier 6 (v2 PPO on persistent-edge sim) as the active training path
  with backlinks to [[../decisions/ppo-gnn|ppo-gnn]],
  [[../decisions/v2-edge-pressure-state]], and the overnight research.
- [[topics/v2-edge-voting-policy]] status flipped to `proposed`. Page now
  opens with what landed (`edge_features.py`, `edge_flow.py`, `--model
  edge`) vs what hasn't (full per-edge logit policy beating `sum`).
- [[topics/v2-edge-loop-emergence]] gained a "Note on rankings" preamble
  flagging that its `sum > attn ≈ sum > max > loop > sum_pw` ranking is
  superseded by overnight matched-pair results; the mechanism story is
  still useful.
- [[questions/open]] rewritten. v1-bootstrap-era questions moved to
  "Closed / answered." New v2 questions: can a learned policy beat
  `wave_long`, is `wave_long`'s gate doing anything beyond `sum_long`,
  why does γ=0.94 help `sum` and hurt `max`, and is `regen-flow` better
  for emergence than transfer-flow.

**Retired:** none (`inbound-bonus`, `loop-bonus` already retired and now
linked from the index under "Retired").

**Questions opened:** all consolidated into [[questions/open]].

## [2026-05-14 | workspace | lightning_attn — 2-head attention solver with frontier-tilt mixing]

**Touched pages:** [[topics/v2-edge-loop-emergence]] [[log]]

User intuition: "the pressure should be a reservoir rather than an
ultimatum." Loop mode banks pressure in the back but never releases;
sum mode releases at every step but stores nothing. Attention answers
this by running both heads simultaneously with a per-cell mixing
weight.

Heads:
- ATTACK head: `attack_score[k] = max(0, pot[nb[c,k]] - pot[c])`
  (max-mode gradient).
- LOOP head:  `loop_score[k] = 1 iff k ∈ {0, 2, 4} and slots k, k+1
  both friendly` (the even-k curl from `lightning_loop`).

Per-cell α from BFS frontier distance: α=0 at frontier (pure attack
release), α=1 deep interior (pure triskelion storage), α intermediate
(blend — loops "tilt" forward as the frontier gets closer). Combined
score = (1-α)·attack + α·loop; friendly slots activate when ≥ 0.5×max.

Initial head-to-head (R=6 P=6 4000 ticks):
- All-attn self-play (6 games)   — 6/6 decisive, no deadlock, mean 1485
  ticks. The mixing breaks the storage trap.
- Sum vs attn (24 games)         — **12-12 tie**. First-pass even with
  the previous champion.
- max/sum/attn 3-way (24 games)  — sum 11 (45.8%), attn 8 (33.3%),
  max 5 (20.8%). Attn decisively beats max but doesn't catch sum.

Full solver ordering after this branch:
`sum` > `attn` ≈ `sum` > `max` > `loop` > `sum_pw`.

Interpretation: hand-designed attention pulling even with `sum` on the
first attempt — with no tuning of `deep_threshold` or `relay_thresh` —
says the architectural shape is right. The next step is *learned* Q/K,
not more knob-tuning: a PPO head that produces `(attack_score,
loop_score)` per slot plus a per-cell α, trained end-to-end on the
existing v2 reward stack.

**Added:** `lightning_attn` mode in `python/flux_v2/solver_lightning.py`;
solver registration; three new `.flxr` replays.
**Updated:** [[topics/v2-edge-loop-emergence]] (three new run tables,
"reservoir-with-release" architecture section, solver-table update).
**Retired:** none.
**Questions opened:** does a learned attention head close the gap to
`sum` and beat it? Does the visual triskelion-tilt actually match the
"loops lean forward toward the frontier" prediction in the replay?
Does the BFS-based α generalize to settings where the frontier moves
fast (capture cascades)?

## [2026-05-14 | workspace | lightning_loop — structural curl rule that closes directed 3-loops]

**Touched pages:** [[topics/v2-edge-loop-emergence]] [[log]]

Follow-up to the sum / sum_pw modes. User pointed out (via screenshot) that
sum_pw replays show bidirectional feeding and Y-confluences, not the
directed 3-cycles the hypothesis predicted. Diagnosis: the diffusion
change was necessary but not sufficient — the strict-uphill action rule
(`pot[d] > pot[c]`) is still tree-only by transitivity regardless of how
smooth the field is.

Fix: structural rule that ignores the potential field entirely. On a hex
grid, neighbors at slots k and k+1 (mod 6) are mutually adjacent — they
form a triangle with c. Each such triangle has a fixed slot-parity (from
all three corners, the slot pair is either both-even-k or both-odd-k).
Restricting the relay rule to k ∈ {0, 2, 4} fills the even-parity
triangles with directed 3-cycles and guarantees the back-edges (slot
k+3, always odd) are never set on the destination — so the v2 reducer's
"no friendly bidirectional flow" invariant never triggers.

Sanity-checked on R=3 all-friendly board: 18/18 interior cells set
exactly slots {0, 2, 4} (triskelion pattern), and the a→b→x→a 3-cycle
closes cleanly with all back-edges off.

Head-to-head (R=6 P=6 4000 ticks):

- All-`loop` self-play (6 games) — 4 decisive, 2 stalemates, mean 2361
  ticks. Loops resolve but slower than `sum`.
- `lightning` vs `lightning_loop` (24 games) — 17-7. Max-mode wins
  decisively.
- `lightning_sum` vs `lightning_loop` (24 games) — 18-3, 3 stalemates.
  Sum keeps its top spot.

Headline: hypothesis confirmed at the structural level (loops form and
the triskelion pattern is unambiguous in
`solver_v2_lightning_loop_*.flxr`), but the strategic cost is real —
3 outflow slots per interior cell go to circulation, away from attack
focus. Worth a frontier-aware hybrid (max-mode where ≤3 friendly
neighbors, loop rule where ≥4) as the next experiment.

**Added:** `lightning_loop` mode in
`python/flux_v2/solver_lightning.py`; solver registration in
`python/scripts/run_v2_solver.py`; three new `.flxr` replays in
`public/v2/replays/`.
**Updated:** [[topics/v2-edge-loop-emergence]] (three new run tables,
geometry explanation, solver registration table, replay list).
**Retired:** none.
**Questions opened:** frontier-aware hybrid effectiveness; whether the
triskelion visual matches expectation; whether a 4-cycle / 6-cycle
variant would have a different slot-cost / pressure-storage tradeoff.

## [2026-05-14 | workspace | lightning sum / sum_pw modes — diffusion that admits loops]

**Touched pages:** [[topics/v2-edge-loop-emergence]] [[topics/v2-algorithmic-solvers]] [[index]] [[log]]

Hypothesis (user-raised): BFS and lightning never make a→b→c→a cycles, and
cycles are a stronger pressure generator than single-cell outflow. Cause is
literally in the operator — original lightning uses `max(intrinsic, γ·max_nbr)`,
which is tree-only by construction (single steepest parent). Fixed by adding
two new modes to `compute_potential`:

- `sum`: `pot[c] = intrinsic + γ·Σ_d (1/deg(d))·pot[d]` (uniform Bellman)
- `sum_pw`: weights neighbor contributions by current `edge_pressure[d→c]`
  (rich-get-richer), uniform fallback when no flow yet.

Both are fixed-point iterations on a discounted Markov chain — the
closed-form "future residual" of pressure circulating, no rollout needed.
Wired through `lightning_solver_actions(mode=...)` and registered
`lightning_sum` / `lightning_sum_pw` seat names in
`scripts/run_v2_solver.py`.

Four head-to-head runs (R=6 P=6 4000 ticks, in
`public/v2/replays/solver_v2_*`):

- **All-`sum` self-play (8 games)** — 8/8 decisive at mean 1501 ticks. Sum
  breaks symmetry fast enough to resolve.
- **All-`sum_pw` self-play (8 games)** — **8/8 stalemates at the 4000-tick
  cap.** Loops do emerge — and they're so defensive nobody can crack
  anyone else's interior. Replay: `solver_v2_lightning_sum_pw_*.flxr`.
- **3-way mix (24 games)** — `sum` 15 wins, `lightning` 9, `sum_pw` 0.
  Sum-mode is the new best solver.
- **`max` vs `sum_pw` 3v3 (24 games)** — `lightning` 23, `sum_pw` 0,
  1 stalemate. Pressure-weighted gets steamrolled head-to-head.

Headline: hypothesis confirmed mechanically (loops form), but the
strategic implication inverted — cycle pressure is *defensive
infrastructure*, not offensive throughput. The genuine win was the
non-edge-weighted `sum` mode, which beats original `max` by ~6/24 games
despite using the same action rule. Worth considering `lightning_sum`
as the new default baseline.

Replays land in the v2 displayer index automatically.

**Added:** `python/flux_v2/solver_lightning.py` modes,
`python/scripts/run_v2_solver.py` solver registrations, four `.flxr`
replays in `public/v2/replays/`, [[topics/v2-edge-loop-emergence]].
**Updated:** [[topics/v2-algorithmic-solvers]] cross-link section,
[[index]] entry.
**Retired:** none.
**Questions opened:** hybrid (`max` frontier + `sum_pw` interior) untested;
whether longer-tick runs (12000+) eventually break `sum_pw` deadlocks;
whether `lightning_sum` should replace `lightning` as the default baseline.
## [2026-05-14 | workspace | v2 viewer recent-runs strip + feature wiki consolidation]

**Touched pages:** [[topics/v2-viewer]] [[decisions/v2-trainer-displayer]] [[index]] [[log]]

The top bar's 3-iteration drip-feed is replaced with a clickable
newest-first strip of all indexed runs (cap 50 in `append_index`), so
older replays are reachable on reload without waiting for the round-robin
to walk to them. The currently-playing chip is outlined; hover shows
filename + relative `saved_at` + radius/seats/kind. Selection unpauses
and sets `forceLoad` so it overrides the auto-cycle gate. New player API
method `loadReplay(file)` and `recentEntries()` now returns the full
index slice instead of the top 8. Topbar `setRecent` signature gained
`(entries, currentFile)`; `setOnSelect(handler)` wires the click path.

**Added:** [[topics/v2-viewer]] — feature catalog for the v2 viewer
(recent-runs list, transport bar, auto-cadence, mixed-radius rebuild,
FLXR v2 binary format, layout). Collected from
[[decisions/v2-trainer-displayer]] and the operational notes in
[[topics/v2-training-runs]] so anyone asking "what does the v2 viewer
do?" has one page to read.

**Updated:** [[decisions/v2-trainer-displayer]] trimmed to the decision
itself (separate UI track + why) and now points at [[topics/v2-viewer]]
for the feature catalog. [[index]] links the new topic.

**Retired:** none.

**Questions opened:** none.

## [2026-05-14 | workspace | v2 displayer gets media-style transport controls]

**Touched pages:** [[v2-trainer-displayer]] [[log]]

The v2 trainer-displayer is a player, so it gets player UI. Added a
bottom-fixed bar with prev / step-back / play-pause / step-forward / next
buttons, a frame scrubber, frame counter, and a 0.25–4× speed cycle, plus
keyboard bindings (Space toggles, ←/→ jog by frame, Shift+←/→ swap
replay). Layered the transport API onto the existing cadence-estimating
player rather than replacing it: scrubbing auto-pauses; auto-cycle to the
next replay is gated on `!paused` so paused state survives the index
poll; user-driven prev/next sets a `forceLoad` flag that bypasses the
gate. New file `src_v2/render/playback.ts`. Speed cycle is a runtime
multiplier on top of the configured `PLAYBACK_SPEED` (does not replace
the auto-cadence logic).

## [2026-05-14 | workspace | v2 edge-voting implementation slice]

**Touched pages:** [[topics/v2-edge-voting-policy]] [[topics/v2-training-runs]] [[log]]
**Added:** first staged edge-voting implementation: shared NumPy edge feature/category builder, edge-channel trainer metrics, non-learning edge-flow heuristic with tiny arena tests, `EdgeAwareActorCritic` behind `train_v2.py --model edge`, and `scripts/pretrain_v2_edge_aux.py` for auxiliary category/channel pretraining.
**Updated:** [[topics/v2-edge-voting-policy]] now records implementation status and the remaining frontier; [[topics/v2-training-runs]] documents the `--model edge` checkpoint split and auxiliary pretrain smoke command.
**Retired:** none.
**Questions opened:** none.

## [2026-05-14 | workspace | v2 edge-channel metrics wired]

**Touched pages:** [[topics/v2-training-runs]] [[log]]
**Added:** trainer-side edge-channel structural metrics for active mine-to-enemy, mine-to-neutral, friendly-relay, and friendly-sink pressure; stored pressure behind frontier; three-AI-tick enemy-pressure release bursts; and capture follow-through pressure.
**Updated:** [[topics/v2-training-runs]] now names the edge-channel wandb metrics, adds monitoring cadence and stop conditions for fresh runs, and records that stale open slots are intentionally unavailable until policy intent/open-score channels exist.
**Retired:** none.
**Questions opened:** none.

## [2026-05-14 | workspace | v2 edge-voting spec review guardrails]

**Touched pages:** [[topics/v2-edge-voting-policy]] [[log]]
**Added:** design guardrails for the edge-voting policy: observer votes advise but only source-owned edges actuate, aggregation must be visibility-normalized, deterministic local-flow labels are soft teachers rather than laws, and the existing v2 `Set`/`Clear`/`No-op` action surface remains the first implementation target.
**Updated:** [[topics/v2-edge-voting-policy]] now includes early success criteria for pure edge-feature tests, channel metrics, heuristic baselines, category/channel prediction, and PPO timing improvement without always-open collapse.
**Retired:** none.
**Questions opened:** none.

## [2026-05-13 | workspace | v2 edge-voting policy spec]

**Touched pages:** [[topics/v2-edge-voting-policy]] [[index]]
**Added:** [[topics/v2-edge-voting-policy]] spec for shifting v2 perception from node-centric aggregate actions toward edge-centric local-flow votes. It defines derived edge types, visible-edge vote aggregation, multi-signal channels, pulse-preserving hold/release constraints, training paths, metrics, and a first implementation slice.
**Updated:** [[index]] links the new spec from the topic route map.
**Retired:** none.
**Questions opened:** whether observers should include owned cells only, all visible cells, or all cells in the local patch; whether final action selection stays one cell action per AI tick or moves to direct edge gates.

## [2026-05-13 | workspace | v2 rules one-pager]

**Touched pages:** [[topics/v2-rules-one-pager]] [[index]]
**Added:** [[topics/v2-rules-one-pager]] as a compact rules reference for v2 pressure-state, tick/capture/action semantics, invariants, reward intuition, and the current policy-vision limit. Added `wiki/media/flux-v2-node-edge-vision.svg` plus app-friendly rendered PNG diagram showing directed slots, edge pressure, node aggregate features, 3-hop GCN context, and the 13-action head.
**Updated:** [[index]] links the new one-pager from the topic route map.
**Retired:** none.
**Questions opened:** none.

## [2026-05-13 | workspace | v2 small-board single-tick run launched]

**Touched pages:** [[v2-training-runs]] [[v2-trainer-displayer]] [[replay-rendering]] [[index]] [[log]]
**Added:** live run entry for `v2-r5-tick1-001`.
**Updated:**
- Killed `v2-transit-strict-001` intentionally at iter 61 while healthy to
  pivot from large-board training to tiny arenas.
- Launched `v2-r5-tick1-001`: radius=5, 12 seats, 18 dead cells, random
  placement, `ai_period_ticks=1`, `record_stride=1`, `gamma=0.998`,
  `gae_lambda=0.99`, and rescaled per-decision reward terms.
- `/index-v2.html` now plays v2 replays at 10 game ticks/sec, shows the
  active board/stride, rebuilds on full board signature changes, and
  interrupts an old replay when a new radius/seat-count stream appears.
**Retired:** none.
**Questions opened:** whether single-tick tiny arenas learn transferable
edge tactics faster than radius-9 arenas.

## [2026-05-13 | workspace | v2 single-tick AI cadence made safe]

**Touched pages:** [[v2-training-runs]] [[log]]
**Added:** none new.
**Updated:**
- `train_v2.py` now uses an explicit AI diagnostic-accumulation window helper
  so `--ai-period-ticks 1` resets waste/transit credit every physics tick
  instead of accumulating across the rollout.
- [[v2-training-runs]] documents the single-tick experiment caveat: 5x more
  PPO decision steps, shorter real-time discount horizon at unchanged
  `gamma`, and per-AI-tick rewards paying 5x as often unless rescaled.
**Retired:** none.
**Questions opened:** whether the first single-tick run should preserve
real-time horizon by raising gamma to about 0.998 and reducing per-tick
reward coefficients.

## [2026-05-13 | workspace | Codex skill for v2 training runs]

**Touched pages:** [[v2-training-runs]] [[log]]
**Added:** none in-repo. Codex skill `flux-v2-training` created at
`~/.codex/skills/flux-v2-training/SKILL.md`.
**Updated:**
- [[v2-training-runs]] now points to the Codex skill from the `Launching a run`
  runbook section.
- The skill routes future agents to the wiki, standard log path, wandb naming,
  replay displayer, launch verification, and run archival steps.
**Retired:** none.
**Questions opened:** none.

## [2026-05-13 | workspace | v2-transit-strict-001 launched]

**Touched pages:** [[v2-training-runs]] [[log]]
**Added:** live run entry for `v2-transit-strict-001`.
**Updated:**
- Launched strict transit run from fresh policy with `--transit-coef 0.001`
  and the known-good v2 reward block.
- Confirmed first four iterations are in-scale: transit ~216-238 per rollout,
  entropy stable at ~2.558, KL calm, and replays landing in the v2 displayer.
- Trainer is teeing into `/tmp/flux-train-v2.log`; direct `nohup` launches
  exited immediately in this Codex app session, so the persistent PTY is the
  live process while keeping the standard log/replay/wandb surfaces.
**Retired:** none.
**Questions opened:** none.

## [2026-05-13 | workspace | v2 strict transit credit wired]

**Touched pages:** [[v2-training-runs]] [[v2-three-term-reward]] [[index]] [[log]]
**Added:** none new.
**Updated:**
- Implemented strict slime-mold transit credit: sources receive an optional
  positive reward when pressure lands on a friendly MAX-strength relay with
  active outflows.
- `tick_batched` now emits `transit_credit_per_cell` alongside
  `waste_per_cell`; `train_v2.py` adds `--transit-coef` and logs
  `reward_transit_iter`.
- Pure diagnostic `transit_credit_per_cell_for_tick` plus tests keep MLX and
  pure reducer accounting aligned.
**Retired:** none.
**Questions opened:** whether a later lax transit mode should also reward
friendly below-MAX relays.

## [2026-05-13 | workspace | strict transit coefficient corrected]

**Touched pages:** [[v2-training-runs]] [[v2-three-term-reward]] [[index]] [[log]]
**Added:** none new.
**Updated:**
- First strict transit launch at `--transit-coef 0.1` produced
  `trn≈24k` and immediate KL/entropy trouble, so it was killed after two
  iterations.
- Restarted the live trainer at `--transit-coef 0.001`; early transit reward
  is ~230 per rollout, in scale with kill/waste instead of dominating them.
**Retired:** none.
**Questions opened:** none.
## [2026-05-14 | workspace | v2-board-connectivity decision + carve fix]

**Touched pages:** [[v2-board-connectivity]] [[v2-algorithmic-solvers]] [[index]] [[log]]

**Added:** [[v2-board-connectivity]] as a first-class decision page. The
rule: every non-DEAD cell must reach every other non-DEAD cell; every seat
must reach every other seat; max seat-pair distance ≤ 4·R. Lots of dead is
fine — isolated live pockets are not. Two enforcement layers documented:
`random_seat_and_dead` (sampler greedy guard) and `run_v2_solver.py` retry
or carve mode.

**Updated:** [[v2-algorithmic-solvers]] gains a "Board precondition" section
backlinking the new decision. [[index]] lists the new decision.

**Fixed:** `carve_seat_connectors` previously bridged only seat-bearing
components, leaving live islands that contained no seats isolated — exactly
the "isolated possibly-active nodes" case the invariant is supposed to
exclude. Now bridges every live component to the main island (most seats,
size as tiebreaker). Solver docstrings (`solver.py`, `solver_lightning.py`)
declare the precondition explicitly.

**Tests added:** `python/tests/test_v2_solver_connectivity.py` — retry mode
across 20 seeds × 3 configs, carve mode across 20 seeds × 2 configs,
helper-function unit tests, and a BFS/lightning smoke test.

## [2026-05-13 | workspace | v2-bigger-hidden64 run kicked off]

**Touched pages:** [[v2-training-runs]] [[log]]

`v2-rebalanced` produced the best plateau yet (R=−532 vs baseline −2725,
dominance 0.85, alive 3.25/12, games actually decisive) — the reward
rebalance was a real unlock. Killed at iter ~40, write-up in
[[v2-training-runs]]. Hypothesis for the third consecutive plateau:
representational capacity. The 32-dim hidden GCN can't discriminate
productive vs wasteful outflows precisely enough.

Started `v2-bigger-hidden64`: HIDDEN 32 → 64, VALUE_HIDDEN 16 → 32. All
other knobs unchanged. If this also plateaus, the bottleneck is the
single-scalar-per-seat value head, not capacity — next move would be
per-cell value heads.

## [2026-05-13 | workspace | v2-rebalanced run kicked off]

**Touched pages:** [[v2-training-runs]] [[log]]

`v2-deeper-3hop` ran ~120 iters and hit a second plateau — better than the
first (entropy collapsed 2.55 → 1.87, action distribution flipped to
SET/CLEAR 31/66, alive_seats_end dropped 12 → 10.5), but waste still
flat at ~−2300 and dominance frozen at 0.29. Diagnosis in
[[v2-training-runs]]: reward magnitudes are imbalanced — waste (~−2300)
overwhelms power (~+91), so the policy converged on "play defensively,
minimize my contribution to waste" instead of "win by attacking." The
2-hop → 3-hop GCN bump was a real unlock; reward shape is now the
bottleneck.

Started `v2-rebalanced` (wandb name `v2-rebalanced`): `power_coef 0.05 →
0.20`, `waste_coef 0.05 → 0.015`, `win_bonus 50 → 200`. Keeps 3-layer
GCN and `entropy_coef=0.003` from previous run. Now an aggressive seat's
expected payoff is meaningfully better than a pacifist's, so the policy
should learn to actually engage.

## [2026-05-13 | workspace | v2-deeper-3hop run kicked off]

**Touched pages:** [[v2-training-runs]] [[log]]

`v2-overnight` plateaued by iter ~30: value head locked in at ev=0.98, but
the policy never committed (entropy stuck at 2.31 ≈ 90% of uniform) and
waste term flat at ~−2700. Diagnosis written into [[v2-training-runs]].

Killed it and started `v2-deeper-3hop` (wandb name `v2-deeper-3hop`) with
two structural changes: (a) 3-layer GCN — third message-passing layer
extends receptive field to 3 hops, addressing the credit-assignment limit
from chain terminator → upstream sources; (b) `entropy_coef 0.01 → 0.003`
so PPO can actually commit to a distribution. Reward shape and board left
untouched for direct comparison.

## [2026-05-13 | workspace | v2-overnight run kicked off]

**Touched pages:** [[v2-training-runs]] [[index]] [[log]]

Spun up the first uncapped v2 PPO run with the full feature stack:

- 9-channel policy input (pressure_in_friendly / pressure_in_enemy /
  pressure_out added on top of the 6-channel baseline; checkpoint-incompatible
  with prior v2 runs).
- 40 dead cells per game, connectivity-guaranteed via BFS-on-removal in
  `random_seat_and_dead`.
- 10000 game ticks (2000 AI ticks) — gives the waste term real exposure.
- `record_stride=1` so the displayer plays tick-by-tick.

wandb: `jasonyandell-forge42/flux-v2/6xjpm2ld` (v2-overnight). Iter cadence
~15 s after JIT warmup. CronCreate job `a67ffc75` polls every 10 min.

Baseline (iter 1): R=−2725, ev≈0, action entropy ≈ 2.55 (uniform), waste
dominates the loss. By iter 18 ev had locked in at 0.97 and entropy started
its gentle decline — value head solid, policy still flat.

[[v2-training-runs]] is the new home for run-level lessons going forward.

## [2026-05-13 | workspace | v2: implementation slice shipped]

**Touched pages:** [[v2-edge-pressure-state]] [[v2-set-clear-actions]] [[v2-three-term-reward]] [[v2-trainer-displayer]] [[index]] [[log]]

Implementation slice for the v2 PRD landed end-to-end:

- **Pure reducer + tests** (`python/flux_v2/state.py`, `step.py`, plus
  `tests/test_v2_step.py` — 11 cases covering loop persistence, capture
  strength, multi-hop, friendly bidirectional resolution, waste accounting,
  dead cells, stale targets).
- **MLX batched step** (`python/flux_v2/mlx_step.py`) with parity test
  against the pure reducer over 25 ticks of randomized boards.
- **PPO trainer fork** (`python/scripts/train_v2.py`) — three-term reward
  (power Δ, waste, time + win bonus), 13-action space, randomized seats +
  10 dead cells per game, full wandb panel including the three reward terms
  broken out separately.
- **FLXR v2 replay format** — version=2 header, 6-byte flow records with
  quantized edge pressure. Python writer in `python/flux_v2/replay.py`,
  TS reader in `src_v2/replay/format.ts`.
- **Trainer-displayer UI** — `index-v2.html` + `src_v2/`. Plays back
  `public/v2/replays/*.flxr`, polls index every 3s, auto-reloads.
- **Overnight training run** started against radius=9, num_players=12,
  G=4, max_ticks=5000, wandb project `flux-v2`.

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
