---
title: v2 champion lab — autonomous campaign board
kind: topic
first_seen: 2026-06-13
last_updated: 2026-06-13T05:16:36+00:00
status: active
---

## What this is

The live board for the autonomous "greatest champion" campaign
([[v2-beat-the-solver-plan]]). A persistent queue + scheduler
(`python/scripts/champion_lab.py`) keeps <=2 evolution runs and a few eval
jobs alive at once, evaluates every finished checkpoint across scales
(R=7,12,20) vs the fixed baseline `lightning_sum_throttled`, and ranks them below. A
10-minute /loop pokes `champion_lab.py tick`, refreshes this page, proposes
new experiments when the queue runs low, and commits + pushes.

**Score** = 0.5·mean-win-rate + 0.5·worst-scale-win-rate (so a champion that
wins at R=7 but loses at R=20 can't top a transfer-robust one). Baseline
self-play = 0.50. A score clearly above 0.50 whose worst-scale win rate is
also above 0.50 is a genuine, transfer-robust improvement over the champion —
the campaign's target. Promotion to a named solver still goes through Todd
(`eval_solvers.py`), never the lab's internal score.

Raw state lives under `python/lab/` (gitignored, durable on disk); this page
is the committed, human-readable projection.

## Live leaderboard

_Updated 2026-06-13T05:16:36+00:00 · baseline `lightning_sum_throttled` = 0.50 · 2 running, 14 pending._

No evaluated candidates yet.


**Running:** `ms_ring0` (gen    9), `ring0_r20` (gen    6)

**Queued:** `ms_ring0_fluid`, `ring0_p12`, `ms_ring1`, `ring1_r12_noanchor`, `ring0_r12`, `ms_ring0_vs_evolver0`, `ms_ring0_wide`, `ms_ring0_dead40`, `ms_ring0_p12`, `ring0_r20_p12`, `ms_ring1_anchor`, `ms_ring1_p12`, `ring1_r20_noanchor`, `ms_ring1_fluid`

Related: [[v2-beat-the-solver-plan]], [[v2-grand-research-plan]], [[v2-todd-measurement-lab]].
