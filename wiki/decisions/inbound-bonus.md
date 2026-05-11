---
title: Inbound bonus
kind: decision
first_seen: workspace
last_updated: workspace
status: retired
---

**Retired 2026-05-10.** Walked the math after shipping — the wall-breaker / "choo-choo" dynamic was already produced by the three existing mechanics (ATTACK_BONUS, REGEN_PER_SEC, friendly inbound flow delivering +k to destination). Adding +0.05/sec per inbound flow was invisible next to the +3/sec inbound transfer it sat beside, and for hubs the destination is already MAX-capped so the bonus did literally nothing. Reverted in 0xxx_ commit; this page kept for backlinks.

## Decision

Owned cells receive a regen bonus proportional to the count of **active inbound friendly flows** arriving at them. Specifically:

```ts
regen = REGEN_PER_SEC + INBOUND_BONUS * inbound_friendly_flow_count
```

With `INBOUND_BONUS = 0.05` and distance-2 connectivity (up to ~18 reachable neighbors), a fully-supplied cell tops out around `1.1 + 0.9 = 2.0/sec` regen — ~82% faster than an isolated cell.

A flow is "active inbound friendly" iff:

- `src.owner === flow.player` (the source still owns its end)
- `src.strength >= MIN_STRENGTH_TO_SEND` (source is actually delivering this tick)
- `dst.owner === flow.player` (destination is friendly to the sender)

Inactive flows (e.g., starved sources) don't count.

## Why

The user observed that the game should reward *connectivity density*, not specific shapes. "More links, not specifically in a straight line." A friendly cycle, chain, and clique on N cells all generate the same regen if you only count nodes — but a clique has many more active flows than a chain, and that should matter.

Tying the bonus to **active inbound flows** rather than passive neighborhood membership:

- Rewards intentional supply-line setup (the AI / player has to actually route flow)
- Naturally distinguishes a 5-clique from a 5-line (clique has ~5× the inbound flows)
- Doesn't trigger for sleeping territory — passive cells don't get free juice

This is what the retired `LOOP_BONUS` was reaching for. That mechanic modified flow delivery (destination receives more than source sends). This is cleaner: regen is augmented; flows stay symmetric. Loops naturally benefit because every cycle participant has one inbound flow.

## Rejected alternatives

- **Friendly-neighbor count (passive).** Cell gets bonus per friendly neighbor regardless of active flow. Simpler but rewards mere co-location, not actual supply. Loses the "AI has to set this up" property.
- **Connected-component size.** Bonus = component_size − 1 (spanning-tree length). Topologically pure — every connected structure pays equally per node. Requires connected-components computation. Doesn't differentiate dense from sparse.
- **Loop bonus on flow delivery.** Was tried and retired. Broke the "friendly flow is a wash" property and felt like a kludge for an effect we wanted to come from real mechanics. [[attack-bonus]] replaced it; this page restores a complementary mechanic without re-introducing the wash break.

## Parity

The CPU `step` (`src/game/step.ts`) and the WGSL `step` (`src/gpu/shaders/step.wgsl`) both implement this. Params buffer in `src/gpu/step.ts` carries `inboundBonus` at offset 40. Parity test (`runParity` in the lil-gui evolution folder) confirms agreement.

## Implications

- Effective regen now depends on the dynamic flow state. The "constant regen" property is gone.
- AI controllers that learn to set up dense supply networks should outperform sprayers. Existing `evolved` champions weren't trained with this rule and may not exploit it; new generations will.
- Magnitudes were chosen to be felt but not break attack-bonus economics. Tunable via `INBOUND_BONUS` in `state.ts`.
