---
title: Multi-player free-for-all (spectator mode)
kind: decision
first_seen: workspace
last_updated: workspace
status: active
---

## Decision

The default browser mode is N-player spectator: 2/4/6/8/12 AI-controlled seats placed evenly around the hex grid perimeter, each running a randomly-assigned controller from the [[ai-zoo]]. No human seat. Click input is removed; mouse-wheel zoom is retained for inspection.

## Details

- `makeInitialState(radius, distance, numPlayers)` in `src/game/graph.ts` collects perimeter cells (those with `max(|q|, |r|, |s|) === radius`), sorts them by polar angle, and assigns seats at evenly-spaced indices.
- `src/render/scene.ts` exports a 12-color palette; player IDs index into it modulo length.
- `src/render/gameui.ts:showBanner` takes an optional AI label and colors the banner per the winner's owner color.
- HUD lists each seat with a color swatch and AI name; built via `innerHTML` for the inline color box.
- **Speed multiplier**: `SPEED = 5` in `main.ts` scales both `stepAcc` and `aiAcc` by the same factor, so the game plays 5× faster in real time without changing balance constants in `state.ts`.

## Why

- Hand-written AIs against themselves stalemate at 2 players (see [[ai-zoo]]). At 6–12 players with mixed AIs the dynamics are visibly different per session — different colors dominate different runs, alliances form by accident.
- Spectator framing matches the project's direction toward [[../topics/neuroevolution|neuroevolution]], where the player is the observer, not a participant.
- Click input was a holdover from the original 2-player vs-AI model and was noise once human play stopped being the goal.

## Re-enabling human play

The dropdown / click-to-flow input that existed before this commit (see git history around the "ai zoo + multi-player free-for-all" commit) is the reference. Add a `humanSeat: -1 | 0..N-1` tunable, gate the pointer handlers on `humanSeat === 0`, and route clicks to `applyAction` with `player = humanSeat`. Removed for cleanliness, not because it was wrong.
