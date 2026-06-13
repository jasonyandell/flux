---
title: Todd measurement lab
kind: topic
first_seen: 2026-05-16
last_updated: 2026-05-16
status: active
---

## Frame

Todd is the trimmed-down, M5-scale measurement lab for flux v2. It sits next
to [[v2-vectorized|Pete]]: Pete generates candidates, games, corpora, and
sample replays; Todd measures them with disciplined local protocols and sends
the useful summaries to wandb.

Todd's job is not to become a cloud-scale eval platform. Its job is to make
local evidence trustworthy enough that the wiki can promote or reject a
candidate without re-litigating seat bias, stale rankings, or noisy one-off
replays.

## Scope

Todd owns:

- matched-pair evaluation and seat-order controls;
- Wilson intervals, coherent-pair sign tests, and stalemate accounting;
- local run manifests, result JSONL, and wiki-ready scoreboards;
- wandb run logging for dashboards and artifact indexing;
- champion-promotion recommendations.

Todd does not own:

- board/corpus generation beyond invoking Pete artifacts;
- solver implementation;
- long PPO training;
- raw replay storage at scale;
- declaring truth from unpaired raw win rates.

## M5 Constraints

Todd assumes one local M5 workstation:

- default to R=20-30, 6-12 players, and bounded wallclock cells;
- use `--workers` around 8-10 on M5 Max unless interactive work suffers;
- keep normal experiments below the overnight-research 30-minute spirit;
- use R=100 only for throughput/stress checks, not routine ranking;
- write compact local artifacts first, then log selected summaries to wandb.

Good arena defaults:

- decisive hierarchy checks: R=20, 5-15% dead, >=100 matched pairs;
- throttle checks: R=25, P=12, 40% dead, same matched-pair discipline;
- smoke checks: smaller pair counts are allowed only to test plumbing.

## Data Model

Todd v0 should be deliberately boring and append-only.

```txt
Candidate     name, kind, source command/hash/config, optional Pete corpus id
Arena         radius, players, dead cells, max ticks, ai period, seed list
MatchPair     candidate A/B, board seed, A-even/B-odd and B-even/A-odd halves
GameResult    winner seat/candidate, ticks, stalemate, cells, dominance, alive seats
Diagnostics   waste, active-slot histogram, stale slots, target-spell completion,
              follow-through after capture, defense saves
RunManifest   git rev, command, timestamps, host, worker count, wandb run id,
              artifact paths
```

Local artifact root:

```txt
python/artifacts/todd/
  runs/<run_id>/manifest.json
  runs/<run_id>/results.jsonl
  runs/<run_id>/summary.json
  runs/<run_id>/scoreboard.md
  runs/<run_id>/interesting_replays.json
```

## Wandb Contract

Todd and wandb are the presentation layer besties, not the source-of-truth
pair. The local JSONL/summary files remain authoritative; wandb indexes and
visualizes them.

Todd wandb runs should use:

- project: `flux-v2`;
- job type: `todd_eval`;
- run names like `todd-vectorized-hierarchy-YYYYMMDD`;
- config containing candidate names, arena config, seed range, git rev, worker
  count, and source Pete manifest when present.

Todd logs:

- raw and coherent win counts;
- Wilson intervals;
- sign-test p-values;
- stalemate rate;
- mean ticks and wallclock;
- diagnostic aggregates;
- compact tables for scoreboards;
- manifest/summary/scoreboard as wandb artifacts.

Todd does not upload every replay frame. It logs replay references or selected
small artifacts for interesting failures and champion examples.

## Protocol

Todd's core protocol is the corrected [[v2-overnight-research]] method:

1. choose a board seed;
2. run A-even/B-odd;
3. run B-even/A-odd on the same board;
4. compare the same-board pair as the evidence unit;
5. report coherent pairs, sign test, Wilson intervals, and stalemates.

First required suite:

- `wave_long` vs `sum`;
- `sum` vs `bfs`;
- `sum` vs `max`;
- `sum` vs `attn`;
- `lightning_sum_throttled` vs `lightning_sum`;
- `lightning_sum_throttled` vs `bfs`.

This refreshes the provisional [[v2-vectorized]] hierarchy and checks the
[[v2-temporal-strategy]] throttle result under Todd's reproducible harness.

## Command Surface

Todd should be a thin command surface over the existing v2 solver/eval path,
not a second simulator.

```bash
cd /Users/jason/code/flux/python

uv run python scripts/todd.py eval-pair lightning_sum_throttled lightning_sum \
  --pairs 100 --radius 25 --num-players 12 --num-dead-cells 200 \
  --max-ticks 12000 --workers 10 --wandb --run-name todd-throttle-r25

uv run python scripts/todd.py suite vectorized-hierarchy \
  --pairs 100 --radius 20 --num-players 6 --num-dead-cells 40 \
  --workers 10 --wandb

uv run python scripts/todd.py measure-manifest \
  python/artifacts/pete/runs/<run_id>/manifest.json \
  --suite champion-regret --wandb
```

Todd may reuse `python/scripts/eval_solvers.py`,
`python/scripts/run_v2_solver.py`, and Pete's vectorized solver path. Its value
is manifests, suites, diagnostics, wandb logging, and scoreboards.

## Stop Conditions

Stop or mark a run invalid when:

- pair halves do not share the same board seed/config;
- a decisive suite exceeds about 20% stalemates;
- coherent pairs are too sparse to interpret;
- diagnostics are obviously off-scale;
- wallclock exceeds the declared local budget;
- solver semantic drift is detected against the run manifest.

Do not stop because raw win rate is near 50%. That is where matched-pair
evidence matters.

## First Milestone

Todd M0 is a wrapper around the current solver eval path that can run the
six-pair starter suite, write local manifests/results/summaries, optionally log
to wandb, and produce a wiki-ready scoreboard. If M0 can confirm or revise the
current `lightning_sum_throttled` champion claim in one reproducible local pass,
Todd has earned the next layer: Pete corpus measurement and preference-example
validation.

Related: [[v2-vectorized]], [[v2-pete-factory]],
[[v2-grand-research-plan]], [[v2-overnight-research]],
[[v2-temporal-strategy]], [[v2-ml-gameplay-opportunities]].
