---
title: One flow per edge
kind: decision
first_seen: 2026-05-10
last_updated: 2026-05-10
status: active
---

## Decision

At most one `Flow` may exist per undirected edge across all players. `applyAction` enforces this in three branches:

- No flow on the edge → add the requested flow.
- Existing flow matches the request exactly (same `src`, `dst`, `player`) → remove it (toggle-off).
- Existing flow differs in direction or player → replace it with the requested flow (the "reverse" semantic).

Source-ownership and adjacency validation are unchanged.

## Why

- Per-edge state becomes a single optional flow, which simplifies UI: an edge has at most one arrow.
- A click opposite to an existing flow is interpreted as a flip, which is the obvious UX. Without this rule the action silently fails or stacks an antiparallel flow that cancels out arithmetically but clutters the display.
- The `step` function is unchanged. Force accumulation already handled the multi-flow case; the rule simply narrows the input space.

## Consequences

- Two players cannot push along the same edge in opposite directions at the same time; the later action wins. This is a real change to play, not a cosmetic one.
- Existing AI (`src/ai/dumb.ts`) is unaffected — it never issues conflicting flows on the same edge.

## Rejected

- **Allow stacking, but display only the net arrow**: keeps state messy and makes UI reasoning harder. The arithmetic equivalence is not worth the input ambiguity.
