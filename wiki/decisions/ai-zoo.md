---
title: AI Zoo
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Decision

Replace the single `dumb` heuristic with a registry of six hand-written AI controllers under `src/ai/`. Each is a pure function `(state, player, seed?) => Action[]`.

- **aggressive** — attack the weakest non-friendly neighbor (formerly `dumb`).
- **random** — pick a uniform-random non-friendly neighbor; seeded by mulberry32 from `src/ai/rng.ts`.
- **defensive** — only attack from cells with `strength ≥ 50`; otherwise hold.
- **greedy-neutral** — prefer neutral targets over enemy; weakest first within each category.
- **opportunist** — only attack if the target is capturable within 10 ticks given current rates.
- **cluster** — prefer targets adjacent to many friendly cells (consolidates the front).

Shared helpers in `src/ai/utils.ts`: `setup(state, player, strengthThreshold)` returns open sources (owned, strong, no outgoing flow), friendly-cancel actions, and an adjacency list. `pickWeakestNonFriendly` is reused by `aggressive` and `defensive`.

`src/ai/index.ts` exports `AIs: Record<AIName, AIFn>` and `AI_NAMES: AIName[]`. Both `main.ts` and `sim/run.ts` look AIs up by name.

## Tournament finding

`npm run sim -- tournament 3` produces a clean win matrix on the default board with [[attack-bonus]] = 0.5:

- **defensive and opportunist lose every pairing they enter** (other than mirror-vs-self draws). Both are tuned too cautiously for the starting strengths: defensive's threshold of 50 takes ~20 sim seconds to reach from start; opportunist's "capturable in 10 ticks" excludes neutrals with starting strength 10.
- **aggressive, random, greedy-neutral, and cluster all stalemate against each other** (0-0-3 across mirror matchups). The four share a target-selection attractor — "attack a weak non-friendly neighbor" — that produces near-identical play at the 1000-cell scale.

The redundant active heuristics make the failure mode legible: not that any individual AI is wrong, but that the *family* of weakest-local-neighbor strategies converges to the same play. That observation is the motivation for [[../topics/neuroevolution|neuroevolution]] — evolving controllers rather than writing more by hand.

## Wiring

Sim subcommands:

```sh
npm run sim                              # default: aggressive vs aggressive, 10 runs
npm run sim -- 50                        # default AIs, 50 runs
npm run sim -- random aggressive 50      # specific pair, 50 runs
npm run sim -- tournament 3              # round-robin, 3 runs per pair
```

Browser is [[multi-player-free-for-all|spectator mode]]: lil-gui exposes a `players` dropdown (2/4/6/8/12) and `respawn` reshuffles AI assignments.
