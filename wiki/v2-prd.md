---
title: Flux v2 — Pressure-Game PRD
kind: prd
first_seen: workspace
last_updated: workspace
status: draft
---

## What this is

A successor to current flux (v1). Same shape — territorial RTS on a hex
graph, neural-net opponents — but the simulation is rewritten to remove
the failure modes v1 hits: spazzy policies, no emergent loops, friendly
bidirectional flow, fanout-as-special-rule, every-tick re-coordination tax.

Mechanics, action encoding, capture behavior, and reward shape are pinned.
Future knobs (growth-rule nonlinearity) and visual specifics remain open.

**Delivery shape:** v2 ships as a *separate codebase track* — new sim, new
trainer, new web UI. The UI is a **trainer-displayer**, not a simulator —
it plays back replays produced by the Python trainer; no in-browser game
logic. Same colors and layout as the current v1 page; just fresh, separate,
and stripped of the three.js debug dropdown. Top bar shows iter / gen and
the last ~3 incoming playbacks as they arrive, like v1 today. See
§ Project structure.

## What didn't work in v1

Every cell re-decides its outflow every 5 ticks. A loop of N cells needs N
cells × hundreds of consecutive correct decisions for the loop to actually
function. Multi-hop transport needs every intermediate cell to *also* be
deciding to send onward every tick. The combinatorics are punishing for
PPO: it finds good single-step policies but never builds emergent structure.
We patched symptoms (passthrough, fanout, override rules, waste penalty)
but the underlying issue is that **the simulation has no persistent edge
state**.

## Design pillars

1. **Pure core.** `step(state, actions) → state'`. No in-place mutation. The
   reducer is pure; rendering, IO, training loop plug in.
2. **Edges are first-class state.** Each edge has its own pressure, persistent
   across ticks, mutated only by what flows through it.
3. **Locality.** No global pathfinding. Pressure on an edge depends on the
   two adjacent nodes plus the cell's outflow configuration. Multi-hop
   transport *emerges* from local rules.
4. **Persistence over re-assertion.** Setting an outflow once is enough.
   The simulation doesn't quiz the policy "are you still sure?" every tick.
5. **No special cases in physics.** No "if maxed, fanout" branch. The
   fill-then-overflow rule applies uniformly to every cell.

## State

### Per node
- `owner ∈ {NEUTRAL, DEAD, player_id}`
- `strength ∈ [0, MAX_STRENGTH]` — only shrinks from enemy damage; sending
  never reduces it.

### Per cell
- `outflow_intent[c, k] ∈ {0, 1}` for each of `K = 6` direct hex neighbors.
  **Multi-outflow:** a cell can have several outflows active simultaneously.

**Action range vs vision range.** Outflows can only target a cell's 6
*direct* hex neighbors (K=6). The policy's GNN can *see* further (planned
r=2 or r=3 via message-passing depth) — so the network plans around cells
it can't directly touch. Multi-hop transport emerges via the persistent
edge state, not via long-range actions.

### Per directed edge (cell c, slot k)
- `edge_pressure[c, k] ∈ [0, MAX_EDGE]` — what's flowing through that edge
  this tick. Computed each tick from the cell's overflow; the *previous*
  tick's pressure is what the downstream cell reads. **One-tick lag** is the
  propagation mechanism.

**Directed half-edges.** A pair (X, O) has *two* independent edges:
`outflow_intent[X, slot→O]` and `outflow_intent[O, slot→X]`. Each cell
owns its own 6 outgoing bits. The pair can be in any of 4 states (idle,
X→O, O→X, X↔O). For enemies all 4 are valid (mutual attack). For
friendlies the bidirectional state is forbidden — see mutation invariants.

### Constants
- `MAX_STRENGTH = 100` — node ceiling.
- `MAX_EDGE = 100` — per-edge pressure ceiling.
- `CAPTURE_STRENGTH = 50` — strength a freshly captured cell starts with.
  Set roughly half `MAX_STRENGTH` so the previous owner's residual
  pressure does damage but doesn't insta-recapture (whip-back fix). Tunable.

## Per-tick algorithm (the only rule)

For each cell c (read end-of-last-tick state):

```
pressure_in_friendly = regen(strength_c)
                     + Σ edge_pressure[d, slot_d→c] for friendly neighbors d
pressure_in_enemy    = Σ edge_pressure[d, slot_d→c] for enemy neighbors d

# 1. Fill first.
grew         = min(pressure_in_friendly, MAX − strength_c)
new_strength = clamp(strength_c + grew − pressure_in_enemy, 0, MAX)
overflow     = pressure_in_friendly − grew      # only excess that didn't fit

# 2. Spill (only if any outflows are set AND overflow > 0).
num_active = count(outflow_intent[c, *])
if num_active > 0 and overflow > 0:
    per_edge = min(overflow / num_active, MAX_EDGE)     # cap; excess = waste
    for each active slot k: edge_pressure_next[c, k] = per_edge
    waste += (overflow / num_active − per_edge) * num_active
else:
    for all k: edge_pressure_next[c, k] = 0             # overflow w/o outflows = waste
    if overflow > 0: waste += overflow
```

That's it. One branch — "do I have anywhere to spill to?" Everything else
falls out:

- **Loops** keep their structure persistently; no re-decision tax.
- **Multi-hop supply** works because each cell's overflow propagates one
  hop per tick through the configured outflows.
- **Maxed-cell fanout** isn't special — it's just what the rule does when
  a cell hits MAX and has multiple outflows.
- **Bidirectional friendly flow** is impossible by construction (see
  mutation invariants below).
- **Closed loops** correctly leak: per-edge cap binds, Σ regen per tick
  becomes waste once edges saturate.

## Actions: mutating `outflow_intent` (Set/Clear, 13 actions)

Each AI tick, per owned cell, the policy emits **one** action from a
**`2K + 1 = 13`** action space (K = 6 direct neighbors):

- Action `0..5`: **set** slot k → 1 (idempotent if already 1)
- Action `6..11`: **clear** slot k → 0 (idempotent if already 0)
- Action `12`: **no-op**

**Why Set/Clear over toggle.** Idempotent, state-independent semantics.
Each action token has the same meaning regardless of current outflow
state, so the network doesn't need the current outflow vector as input
to predict its effect. With K=6 the output layer is small either way
(13 vs 7), so toggle's compactness advantage doesn't pay for its
state-dependence cost.

Output layer: 13 logits per cell.

## Mutation invariants (resolved at AI tick)

1. **No friendly bidirectional flow.** If my action turns on a slot
   pointing at friendly d, and d's outflow includes a slot pointing at
   me, d's slot toward me is forcibly cleared.
2. **Simultaneous bidirectional.** If two friends mutate their respective
   slots toward each other this same AI tick, higher cell-index wins;
   lower-index's mutation is undone.
3. **Capture clears (origin).** When *my* cell's owner changes, all its
   `outflow_intent` bits reset to 0. The new owner inherits a blank slate.
4. **Stale targets stay on.** If my outflow points at a friend who got
   captured by an enemy, the slot **stays set**. The pressure now arrives
   as damage (enemy receiver). Pure scalar semantics — the receiver
   decides what inbound pressure means based on current ownership. The
   raised `CAPTURE_STRENGTH = 50` is what makes this livable: a freshly
   captured cell has enough HP to survive residual whip-back from the
   previous owner's still-set inflows.

## Reward shape

Three terms, deliberately minimal. PPO does better with a small set of
clean signals than a dense stack of overlapping ones (v1's lesson).

```
step_reward[player] =
    + power_coef * Δ(Σ strength_owned[player])    # build territory at strength
    - waste_coef * waste_per_player[player]        # don't generate dead-end pressure
    - time_coef                                    # impatience → speed
terminal_reward[winner] += win_bonus               # +score for finishing
```

**Power** is the stock measure: sum of strength across cells you own. Use
the per-tick delta to avoid huge magnitudes. Higher strength implicitly
rewards more regen (regen scales with strength), so this single term
captures "more territory and more developed" together.

**Waste** is the algorithmically attributed dead-end pressure the
fill-then-overflow rule already emits per cell per tick:
- Overflow with no outflows → all of it is waste.
- Per-edge cap binds → excess past `MAX_EDGE` is waste.
Both are summed by cell-owner — exact attribution.

**Speed** is a small per-tick time penalty plus a terminal win bonus.

**Overkill is NOT counted as waste** (an attacker pumping 100 pressure
into a 5-strength cell). The defender's strength isn't known at commit
time, so penalizing overkill teaches the policy to be too cautious. We
may revisit this if overkill turns out to dominate observed waste.

**Engagement / activity coefs from v1 are gone.** "Fraction of cells with
active outflows" doesn't translate cleanly — under v2 a stable loop has
every cell active permanently and shouldn't be rewarded extra for that.
Persistence makes activity measures meaningless.

## Why this fixes v1's failure modes

- **Loops.** N slots set once, persists. The "1000-tick coordination cost"
  evaporates.
- **Multi-hop supply.** Each cell's overflow chains through configured
  outflows automatically — no per-tick re-deciding by intermediate cells.
- **Bidirectional friendly flow.** Impossible by construction.
- **Maxed cells.** No special rule. They overflow naturally because they
  can't grow further. If outflows are set, the overflow goes out. Same
  rule everyone follows.
- **Spazzy policies.** The policy is choosing *structure*, not re-affirming
  it every tick. Effective decision horizon shrinks dramatically.

## Project structure (delivery shape)

v2 is a *new track* — does not replace or share code with v1. Sketch:

```
python/flux_v2/             # new sim, new step function, new state types
python/scripts/train_v2.py  # new PPO trainer, writes checkpoints + replays
src_v2/                     # new web UI (trainer-displayer, not a simulator)
public/v2/                  # v2's replays + checkpoint statics
wiki/decisions/v2-*.md      # design notes as we hit them
```

**The v2 UI is a trainer-displayer.** It reads `.flxr` replays the trainer
writes; no in-browser game logic. Visually it mirrors the current v1 page
— same colors, same layout — but stripped down:

- **No** three.js debug dropdown, no live-sim toggle, no tunables panel.
- **Top bar** shows iter / gen and the last ~3 incoming playbacks as they
  arrive (same drip-feed UX as v1's replay rotation).
- **Auto-reload** on new replay write, identical to v1's flow.

v1 page is left alone. Routing TBD — separate Vite entrypoint, `/v2/`
route, or `index-v2.html`. Pick simplest at implementation time.

## Open questions

1. **Carry decay.** Edge pressures persist and are re-written each tick
   from the source's overflow. No decay needed — a cell that stops
   overflowing writes pressure 0 next tick, edges naturally go quiet.
   (Resolved-by-design; listed for completeness.)

2. **Growth rule knob (future).** Today: `grew = min(pressure_in_friendly,
   MAX − strength)`. Linear, no scaling on incoming flow. Future knob:
   `grew = f(strength, friendly_in, enemy_in, num_outflows)` — keeps the
   door open for nonlinear growth dynamics without changing the v2 launch.

3. **Visual specifics.** Edges with active outflow + non-zero pressure
   render with thickness proportional to pressure; carry/overflow visible
   at the receiving end as growing strength. Color palette and layout
   match v1. Details settled at implementation.

4. **Overkill-as-waste (deferred).** Currently overkill (enemy pumping
   pressure past what was needed to capture) is *not* counted as waste —
   the attacker can't know the defender's strength when committing.
   Revisit if overkill turns out to dominate observed waste in training.

## Not in scope

- MLX implementation details
- Migration / coexistence with v1 (v2 is a new codebase track)
- Browser rendering specifics
- PPO hyperparameter retuning

## Next step

Design phase done. First implementation slice: pure reducer in
`python/flux_v2/` with a small unit test suite (loops persist, captures
respect CAPTURE_STRENGTH, waste accounting matches algorithm spec).
Trainer (`train_v2.py`) and UI (`src_v2/`) come after the reducer is
locked.
