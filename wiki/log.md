---
title: flux Wiki Log
kind: log
first_seen: bootstrap
last_updated: bootstrap
status: active
---

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
