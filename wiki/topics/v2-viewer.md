---
title: v2 viewer (trainer-displayer)
kind: topic
first_seen: 2026-05-14
last_updated: 2026-05-16
status: active
---

## What this page is

The single feature catalog for the v2 trainer-displayer at
`/index-v2.html`. The *why* of a separate UI entrypoint lives in
[[v2-trainer-displayer]]; this page collects what the viewer **does**
for someone watching v2 PPO replays land.

The viewer plays back `.flxr` v2 replays from `public/v2/replays/`. It
never runs the simulation in-browser — the trainer is the source of
truth, the viewer is the lens. The list below stays accountable to
`src_v2/` as it ships.

## Why anyone reaches for it

- Live training monitor — the trainer drops a `.flxr` plus an
  `index.json` entry each `--record-stride` iters; the viewer polls and
  auto-swaps to the newest replay on a 3 s cadence (`POLL_INTERVAL_MS`).
  New writes also append to `events.jsonl`, which drives the hamburger
  arrival badge.
- Post-hoc inspection — by default the index keeps the last **50** runs
  (`append_index(cap=50)` in [`python/flux_v2/replay.py`](../../python/flux_v2/replay.py)),
  so you can scroll back through whatever's still on disk.
- Agent handoff — a run is addressable by URL:
  `/index-v2.html?replay=<file.flxr>`. The viewer loads that exact file,
  keeps the URL in sync when the user selects another run, and exposes copy
  link controls in the current-run header and playlist rows.
- Transport handoff — explicit replay changes preserve the current play/pause
  state and speed multiplier. If the viewer is paused at `0×` or playing in
  reverse, loading another replay keeps that same temporal mode.
- Mixed-experiment streams — radius/seat changes are normal between
  runs; the viewer rebuilds geometry on board-signature changes and
  doesn't need a reload.

## Replay browser

The top-left hamburger opens a searchable replay drawer. The drawer is inert
while closed, so hidden rows cannot catch clicks or confuse browser
automation.

- **Order:** index is sorted `saved_at` descending on every write
  (`python/flux_v2/replay.py::append_index`), so the freshest run is first.
- **Search:** filename, solver name, ruleset, model, `rN`, `pN`, and
  `ea0p05`-style alpha tokens are searchable.
- **Quick filters:** All, Solver, Train, Fluid, New, Current.
- **Copy link:** every row has a `link` button that copies
  `/index-v2.html?replay=<file>`. Selecting a row loads that replay and
  closes the drawer.
- **Arrival badge:** `events.jsonl` is tailed every 3 s. The badge counts
  replay events newer than the last time the drawer was closed
  (`flux-v2-playlist-last-closed` in `localStorage`).

## Current-run header

The compact top-right run header shows the current filename plus useful
metadata when present: `ruleset`, `edge_alpha`, `seed`, `winner`, and
`ticks`. Its `link` button copies the exact current replay URL. Solver
runs now write richer metadata and index fields from
[`python/scripts/run_v2_solver.py`](../../python/scripts/run_v2_solver.py):
seed, board/run shape, seat solvers, edge alpha, winner/leader, final ticks,
alive count, dominance, and final cell counts.

## R40 smooth-replay probes

On 2026-05-16, the viewer/feed path was smoke-tested with large,
smooth-recorded solver games: `radius=40`, `num_players=12`,
`num_dead_cells=720`, `edge_alpha=0.05`, `max_ticks=9000`,
`record_stride=1`, and carved connectivity. The useful replay artifacts are:

- `solver_v2_lightning_sum_throttled+lightning_wave_long_ea0p05_20260516T210154.flxr`
  — frontier duel; `lightning_sum_throttled` seat 8 won at tick 6794.
- `solver_v2_lightning_chase+lightning_sum_wave+lightning_vortex_ea0p05_20260516T210154.flxr`
  — morphology mix; `lightning_sum_wave` seat 11 won at tick 4921.

## Transport bar (bottom)

```
[⏮]  [⏪]  [⏯]  [⏩]  [⏭]   ━━━●━━━━━   12 / 240   1×   ✦
 prev  step  play  step  next   scrubber    counter   speed  fade
```

- **Prev / Next replay** walk the index round-robin
  (`Shift+←` / `Shift+→`). Skipped (unplayable) entries are filtered
  out, same set the auto-cycle uses.
- **Step back / forward one frame** (`←` / `→`) auto-pause and jog ±1.
- **Play / Pause** (`Space`) freezes the current frame. While paused,
  the index poller still discovers new replays but does not preempt;
  auto-cycle fires only at end-of-replay AND `!paused`. Manual prev/next
  still works (`forceLoad` bypasses the gate).
- **Scrubber** is a `<input type="range">` over the frame index
  (0..N−1). Dragging auto-pauses; programmatic updates are suppressed
  while the user has the slider grabbed so playback doesn't fight the
  drag.
- **Speed cycle** picks `-8 / -4 / -2 / -1 / -0.5 / -0.25 / -0.1 /
  -0.05 / 0 / 0.05 / 0.1 / 0.25 / 0.5 / 1 / 2 / 4 / 8×`, applied as a
  *runtime multiplier* on top of `PLAYBACK_SPEED` and the cadence-aware
  `framesPerSec` the player computes per replay. 1× means "whatever the
  player picked"; negative values play the current replay backwards, and
  `0×` pauses on the current frame. The multiplier never replaces the
  auto-cadence logic.
- **Fade trail** (`✦` on / `✧` off) toggles the per-node brightness
  pulse-and-fade effect. On by default; preference persists in
  `localStorage` under `flux-v2-fade-enabled`. See
  [[#node fade trail]] below.

The bar sits at z-index 9, 0.55 opacity, fading to 1.0 on hover, so it
stays out of the way during passive viewing.

## Canvas gestures

The canvas owns Mac-friendly inspection gestures:

- **Trackpad pinch** (Chromium/Electron reports this as `ctrl+wheel`) zooms
  the orthographic camera around the cursor. The camera clamps to `0.35..8×`.
- **Two-finger trackpad scroll** pans the map. The camera clamps to the board
  envelope plus a small margin so a zoomed-in view cannot drift completely
  away from the action.
- **Shift-scroll** adjusts playback speed. Small scrolls feather through the
  fine `0.05 / 0.1 / 0.25×` stops around pause; faster scrolls advance through
  the ladder faster. Crossing direction stops at `0×` and pauses. A separate
  follow-up shift-scroll crosses from `0×` into reverse or forward playback.
  Reverse playback walks the current replay backward and stops at frame 0.

Wheel handling is attached only to the canvas, so the replay drawer can keep
using normal scroll behavior. The canvas mental model is spatial by default:
unmodified scroll pans; the Shift modifier opts into time control.

## Status row

The top bar shows the live player
status — `polling` / `loading <file>` / `playing <file> (N frames over
~Xs, fps)` / `skipped <file> (0 frames)` — alongside iteration, fitness,
model tag (`[gnn]` or `[edge]`), and board signature
(`r<radius> · p<num_players> · n<num_nodes> · <stride>t/frame · <s>s`).

## Auto-cadence playback

Playback follows game time, not "fit this replay into N seconds". With
`dt_per_tick_ms=100` and `tick_stride=1`, the UI shows every recorded
physics tick at 10 ticks/sec. Wider strides still render correctly but
visibly jump by that many game ticks per frame.

When the index has timestamps, the player estimates the trainer's drop
cadence from the median delta between neighboring `saved_at` values and
speeds playback up so the current replay finishes slightly before the
next one is expected — never below the configured baseline tick rate.
This is what keeps a backlog from accumulating during live monitoring.

## Node fade trail

Three-stage brightness model on each node:

1. **Target freshness** (`freshness[i] ∈ [0, 1]`) — snaps to `1`
   whenever the node's **owner** or **flow-membership signature**
   changes, then decays by `FADE_PER_ITER = 1/20` per replay-frame
   advance. Pressure changes don't trigger snaps (continuous → every
   frame would flash). Iter-keyed, not wall-clock: pause holds the
   glow, scrubbing back doesn't silently fade, fast-forward burns
   through trails at the forward-stepping rate. Forward jump → decay
   by `delta · FADE_PER_ITER`; non-positive delta → no decay.

2. **Displayed value** (`displayed[i] ∈ [0, 1]`) — wall-clock-eased
   toward the target at `FRESHNESS_RATE_PER_SEC = 2.0/s`, so any single
   `0 ↔ 1` transition takes ≥ 0.5 s of real time. Under rapid
   bouncing (fast playback, scrub burst) the displayed value never
   reaches either extreme — it orbits the mean, producing a continuous
   soft pulse instead of hard flashes.

3. **Render brightness** —
   `MIN_BRIGHTNESS + (1 − MIN_BRIGHTNESS) · (0.8 · displayed + 0.2 · pNorm)`.
   `pNorm` is max-pressure-touching-node, per-frame auto-scaled to the
   frame's heaviest flow (same convention the arrow widths use, so an
   idle frame doesn't crank everything up). Above-base headroom splits
   80% age-delta / 20% pressure: max pressure alone on a long-static
   node sits at 20% over base; max pressure *and* a fresh change hits
   full bright.

Toggle: the `✦` button on the transport bar flips a `fadeEnabled` flag
on `Scene`; when off, every node renders at full brightness regardless
of `displayed` or `pNorm`. The state persists in `localStorage` under
`flux-v2-fade-enabled` (default on). Implementation:
[`src_v2/render/scene.ts`](../../src_v2/render/scene.ts) for
`FADE_PER_ITER` / `FRESHNESS_RATE_PER_SEC` / `MIN_BRIGHTNESS` knobs and
the brightness mix;
[`src_v2/render/playback.ts`](../../src_v2/render/playback.ts) and
[`src_v2/main.ts`](../../src_v2/main.ts) for the toggle wiring.

## Mixed-experiment safety

The displayer rebuilds geometry from each replay's
`(radius, num_players, num_nodes)` signature, so radius-5 and radius-9
runs can share the same replay index. When a *newly-arrived* replay has
a different board shape, the player interrupts the current replay
instead of waiting for it to finish; a static newest entry that's been
sitting in the index doesn't preempt every poll.

## FLXR v3 binary format

V3 replays use a JSON header plus gzip-compressed dense frames. The header
carries radius, seat count, node count, tick stride, `max_strength`,
`max_edge`, and metadata. Per frame: owners, quantized strengths, outflow
bitset, and pressure bytes for active outflows. Geometry is derived from
`(radius, num_players)` by `buildBoard` on the client. Writer side:
[`python/flux_v2/replay.py`](../../python/flux_v2/replay.py). Parser side:
[`src_v2/replay/format.ts`](../../src_v2/replay/format.ts).

The browser reader streams FLXR v3 responses: it reads the fixed header first,
starts gzip decompression on the remaining body, attaches the replay after the
first two frames, and keeps appending frames as they arrive. Large R40 replays
therefore show a first frame before the full file has downloaded, inflated,
and parsed. If browser streaming/decompression APIs are unavailable, it falls
back to the older whole-file parse.

## Layout

```
index-v2.html               Vite entrypoint (root); served at /index-v2.html
src_v2/board.ts             hex grid builder (mirrors python/flux_v2/graph.py)
src_v2/replay/format.ts     FLXR v3 parser (gzip dense frames)
src_v2/replay/events.ts     events.jsonl tailer for arrival badges
src_v2/replay/player.ts     index poller + frame advancer + transport API
src_v2/render/scene.ts      three.js orthographic scene, instanced nodes
src_v2/render/topbar.ts     iter row + status
src_v2/render/playlist.ts   searchable replay drawer
src_v2/render/playback.ts   bottom transport bar
src_v2/render/runHeader.ts  current-run summary + copy-link button
src_v2/main.ts              entry: wires it all up
```

## What's deliberately not there

Holding the displayer to "trainer-displayer" — no live sim, no
parameter knobs — keeps the surface small enough to evolve alongside
the v2 ruleset without breaking the v1 entrypoint. The decisions:

- No three.js debug GUI (`lil-gui`).
- No live-sim toggle, no tunables panel.
- No selection ring, no drag-to-edit affordances.
- No restart / evolve buttons.

What's kept from v1: same colors, same hex layout, same
arrow-thickness-by-pressure flow rendering, same auto-poll-and-swap
replay flow.

## Operational pointers

- Replays land at `public/v2/replays/*.flxr` + `index.json` +
  `events.jsonl`; the Vite
  dev server serves them at `/v2/replays/`. `.gitignore` already
  excludes them, so long runs don't pollute git.
- Each replay at `record-stride=25` is ~1.5 MB and the index keeps 50,
  so steady-state disk use is small. Tick-by-tick (`record-stride=1`)
  was the v2 default early on; runs that need play-by-play resolution
  still pass `--record-stride 1`.
- To preserve a notable replay across the 50-cap rotation, force-add it
  to git: `git add -f public/v2/replays/<file>.flxr`.

Related: [[v2-trainer-displayer]] (the *why* of a separate UI track),
[[v2-training-runs]] (how to launch a run that feeds this viewer),
[[v2-edge-pressure-state]], [[v2-rules-one-pager]],
[[decisions/replay-rendering]].
