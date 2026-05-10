---
title: Attack bonus
kind: decision
first_seen: 2026-05-10
last_updated: 2026-05-10
status: active
---

## Decision

Non-friendly destinations receive a multiplier `1 + ATTACK_BONUS` (default `0.5`) on incoming transfer. Friendly destinations and the source loss are unchanged.

```ts
const k = TRANSFER_PER_SEC * dt;
forces[flow.src][flow.player] -= k;
const enemy = state.nodes[flow.dst].owner !== flow.player;
forces[flow.dst][flow.player] += enemy ? k * (1 + ATTACK_BONUS) : k;
```

Since the resolve step subtracts the attacker's force from the defender's strength, multiplying it amplifies damage dealt without changing the attacker's cost.

## Why

Discrete-ship galcon-likes get combat asymmetry for free: a ship in transit is committed, and a defender absorbs the full ship cost. The flux continuous model had no such asymmetry — `+k` out of the attacker, `−k` from the defender, net zero on the system. Two equally-strong neighbours firing at each other drained at the same rate as one of them firing alone, and neither could break the tie.

With `ATTACK_BONUS = 0.5`:

- One-sided attack: attacker loses `+REGEN − k = 1·dt − 3·dt = −2·dt`/tick; passive defender loses `+REGEN − k·1.5 = 1·dt − 4.5·dt = −3.5·dt`/tick. Defender drains 1.75× faster than attacker.
- Mutual fire: both nodes lose `1·dt − 3·dt − 4.5·dt = −6.5·dt`/tick. Symmetric, but accelerated — front-line stalemates resolve faster.
- Friendly destinations: unchanged. A closed friendly loop is a wash again (regen-only growth), as in the original model.

## Consequences

- The `step` signature is unchanged. Pure-core invariants hold.
- Capture mechanics unchanged in form — the resolve step still flips ownership when strength crosses zero, with surplus becoming the new owner's strength. The surplus is just larger now because the incoming force was larger.
- Neutral targets count as "enemy" under `dst.owner !== flow.player`, so neutral capture is also accelerated. Intentional; symmetric with combat.
- Dumb AI mirror-match outcomes may change — see [[questions/open]].

## Rejected

- **Loop bonus** ([[loop-bonus]], retired): rewards circulation in friendly territory. Works, but pushes the meta toward defensive turtle-grinding. Combat asymmetry encodes the same "you need flows" lesson but in the dimension that produces conflict, not in the dimension that avoids it.
- **Multiplier on source loss for enemy targets** (attacker pays less than defender absorbs): also accelerates resolution but is harder to read at a glance — players see strength leave at a different rate than they expect. Multiplying the destination side keeps the source rate stable.
- **Bonus scales with strength differential** (strong-attacks-weak hits harder): nonlinear, requires more reasoning at the UI level. Out of scope.
