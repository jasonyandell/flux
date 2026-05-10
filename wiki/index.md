---
title: flux Wiki Index
kind: index
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Route map

Start at [[entities/flux|flux]] for the whole project.

Core entities:

- [[entities/flux|flux]] — the game.

Topics:

- [[topics/galcon-like|galcon-like]] — the genre and its lineage.

Decisions:

- [[decisions/continuous-flow-model|continuous-flow-model]] — why strength is a continuous scalar with rate-based flows.
- [[decisions/pure-step-function|pure-step-function]] — game logic is a pure `step(state, dt) → state`, shared by browser and headless sim.
- [[decisions/one-flow-per-edge|one-flow-per-edge]] — at most one flow per undirected edge; reverse means flip.

Questions:

- [[questions/open|open]] — unresolved questions.
