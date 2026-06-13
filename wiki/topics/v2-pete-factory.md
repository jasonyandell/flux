---
title: Pete factory
kind: topic
first_seen: 2026-05-16
last_updated: 2026-05-16
status: active
---

## Frame

Pete is already the canonical name for the vectorized v2 generator / trainer /
solver path ([[v2-vectorized]]). The Pete factory is the local M5 artifact
factory around that path: it makes deterministic boards, solver games, replay
samples, teacher shards, and raw divergence data for [[v2-todd-measurement-lab]]
to measure.

The boundary is simple: Pete makes material; Todd measures it.

## Scope

Pete owns:

- connected v2 board generation through the vectorized path;
- local solver/candidate game generation;
- raw game records;
- selected FLXR v3 sample replays;
- teacher shards for solver distillation;
- raw divergence and weak preference candidates;
- resumable local job manifests.

Pete does not own:

- wandb logging;
- Wilson intervals or sign-test interpretation;
- champion promotion;
- long PPO training;
- leaderboard updates.

## Artifact Layout

Bulk artifacts stay out of `public/`. Only curated visual samples go through
the replay viewer path.

```txt
python/artifacts/pete/
  runs/<run_id>/manifest.json
  runs/<run_id>/jobs.jsonl
  runs/<run_id>/raw_games.jsonl
  runs/<run_id>/replay_refs.jsonl
  corpora/teacher/<corpus_id>/shard-00000.npz
  corpora/preferences/<corpus_id>/pairs-00000.npz
  corpora/candidates/<corpus_id>/candidates.jsonl

public/v2/replays/*.flxr
public/v2/replays/index.json
public/v2/replays/events.jsonl
```

## Candidate Generation

Candidate records are config, not quality claims.

Initial candidate family:

- `lightning_sum_throttled`;
- `lightning_sum_long`;
- `lightning_wave_long`;
- `lightning_wave_keep_attack_long`;
- `lightning_sum`;
- `bfs`;
- `max`;
- `attn`.

Each candidate record should include solver name, solver parameters, board
distribution, seed range, `connect-mode`, `EDGE_ALPHA`, git rev, and source
command.

## Teacher Shards

Teacher shards support solver distillation and edge-aware auxiliary learning
without pretending the teacher is final truth.

Each decision-tick sample should include:

- compact state reference or state tensor;
- acting seat;
- legal mask;
- chosen cell action;
- desired edge mask when available;
- solver name and parameters;
- board seed and tick;
- edge categories/channels from `python/flux_v2/edge_features.py`.

Teacher labels remain soft. They teach the physics vocabulary described in
[[v2-edge-voting-policy]]; Todd later measures whether a learned policy beats
the teacher.

## Preference Candidates

Pete may mine same-board divergences between candidate solvers, but those
records are weak until Todd validates them.

Preference candidates should store small windows around disagreements rather
than full games:

- board seed and arena config;
- candidate A/B;
- tick window;
- action/edge-intent differences;
- local state slice or state reference;
- raw outcome pointer;
- `todd_status: pending`.

Todd promotes only statistically supported disagreement records into durable
preference data.

## M5 Budget

Pete should stay cheap and local:

- default to R=20/R=30, 6-12 players, 6000 ticks, stride 25;
- cap workers around 8-10 on M5 Max;
- run FLXR only for sample replays, not bulk corpora;
- write compact `.npz` shards for training data;
- avoid partial MLX ports unless the whole hot path moves there.

Pete's current advantage is the Numba-JIT path documented in
[[v2-vectorized]]: warm-start Bellman, batched per-AI-tick solver, JIT board
setup, and compact FLXR v3 output.

## Queue Lifecycle

Pete jobs should be resumable and inspectable.

```txt
queued -> warming_jit -> running -> materialized -> handed_to_todd -> archived
                                      failed -> retryable/terminal
```

Each `jobs.jsonl` row needs `job_id`, `run_id`, seed range, board config,
candidate config, artifact paths, status, timestamps, exit code, and error
summary. Resume skips materialized jobs whose artifact checksums still match.

## Todd Handoff

Pete writes one manifest per run:

```json
{
  "schema": "pete.manifest.v0",
  "pete_run_id": "pete-20260516-001",
  "artifact_root": "python/artifacts/pete/runs/pete-20260516-001",
  "raw_games": "raw_games.jsonl",
  "replay_refs": "replay_refs.jsonl",
  "corpora": [],
  "needs_todd_measurement": true,
  "todd_status": "pending",
  "todd_result": null
}
```

Todd consumes the manifest, runs or verifies the matched-pair scoring, logs
wandb, and may write back a Todd result pointer.

## First Milestone

Pete M0 is a generator wrapper, not a dashboard:

1. create a run manifest and queue format;
2. generate the four provisional rerank batches from [[v2-vectorized]]:
   `wave_long` vs `sum`, `sum` vs `bfs`, `sum` vs `max`, `sum` vs `attn`;
3. emit raw paired-game records and one or two sample FLXR replays per matchup;
4. emit a small teacher shard from the current champion candidate plus one
   challenger;
5. hand the manifest to Todd for measurement and wandb logging.

Related: [[v2-vectorized]], [[v2-todd-measurement-lab]],
[[v2-grand-research-plan]], [[v2-ml-gameplay-opportunities]],
[[v2-edge-voting-policy]], [[v2-algorithmic-solvers]].
