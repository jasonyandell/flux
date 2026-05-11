---
title: Todo
kind: todo
first_seen: workspace
last_updated: workspace
status: active
---

Active threads, scratchpad for now. Moving to GitHub Issues or [beads](https://github.com/...) once volume justifies it.

History lives in [[log|log.md]]. Theory-shaped questions live in [[questions/open|questions/open.md]].

## Open — AI / evolution

- **Python pipeline.** Discussed architecture: MLX evolution loop on Apple Silicon, league-style mixing of past champions (AlphaZero-inspired), filesystem bridge via JSON `champions/` directory consumed by the browser. Three forks pending decision:
  - League sampling strategy: pure-random vs latest-N-only vs Elo-weighted bracketed.
  - Server: Vite-static (existing dev server serves `python/champions/`) vs separate Python `http.server`.
  - Browser automation level: manual "record next tournament" button vs scripted loops for unattended runs.
  - 2–3 days of work once forks resolve.

- **rtNEAT / proper NEAT.** Current evolution is [[topics/neuroevolution|tier 1]] (fixed topology, weights only). Tier 2 (structural mutations, innovation numbers, speciation) and tier 3 (rtNEAT continuous replacement) are open. No urgency — evolution already shows wins.

- **AI that paths to enemy bases (not just weakest local).** From [[questions/open]] possible-directions list. BFS-based target prioritization would break the four-AI weakest-local-neighbor attractor without needing neural.

## Open — capture / video

- **Tournament-from-pool video pipeline (browser-only version).** Load multiple champion JSON files, assign to seats randomly, auto-record N consecutive games, save webms. ~1 day. Doesn't depend on Python — uses files from the existing save-champion button. Closes the "I want recordings of evolved AI wars" thread.

## Open — game model / bugs

- **Graph topology variants.** From [[questions/open]]: multiple parallel routes as a structural alternative to a single central choke. Untested — could change AI dynamics meaningfully but no concrete plan.

- **Further constant tuning.** `REGEN_PER_SEC` already bumped 1.0 → 1.1 for pacing. `TRANSFER_PER_SEC` and `ATTACK_BONUS` are untouched and probably fine; revisit only if game pacing shifts under heavier evolution.

## Process

- **Move to GH issues or [beads](beads://...) once we have >10 active items.** This file is a holding pen.
