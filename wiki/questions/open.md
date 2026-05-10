---
title: Open questions
kind: question
first_seen: bootstrap
last_updated: bootstrap
status: active
---

## Open

### Dumb AI stalemates against itself

`npm run sim` of dumb-vs-dumb produces 100% draws at the tick limit. Still 100% on the new ~1000-cell hex board, and still 100% after [[attack-bonus]] (which accelerates mutual fire) — the symmetric stalemate is structural (front-line oscillation), not bottlenecked by combat speed or graph size. Trace on the bootstrap graph shows the cause; both sides capture their adjacent neutrals quickly, then the front line oscillates because:

- The AI only attacks the *weakest* non-friendly neighbor; the contested center node (strength 10) is always less weak than a recently-recaptured front-line node (strength near 0), so attacks bypass the choke point.
- Front-line nodes drop below the `strength > 5` send threshold during attacks, so they stop attacking back; flows are recreated on the next AI tick but make no net progress.

The scaffold is correct — the heuristic is genuinely too weak to break the symmetric stalemate. Useful as a baseline for any future AI.

Possible directions:

- AI that targets paths to the enemy base rather than weakest local neighbor (BFS-based prioritization).
- Stochastic AI seed parameter so headless sims explore the action space.
- Graph topology that lacks a single central choke (multiple parallel routes).
- Tuning `REGEN_PER_SEC` / `TRANSFER_PER_SEC` to make sustained attacks viable.
