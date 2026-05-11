/**
 * JS reference parity scenario. Runs the identical 100-tick deterministic
 * scenario as python/tests/test_parity.py and prints the same hashes.
 *
 * Run from repo root:
 *   npx tsx python/tests/dump_reference.ts
 */
import { createHash } from 'node:crypto';
import { makeInitialState } from '../../src/game/graph';
import { applyAction, step } from '../../src/game/step';
import { mulberry32 } from '../../src/ai/rng';
import {
  randomGenome,
  buildNeighborTable,
  nnInferCell,
  NEIGHBOR_STRIDE,
} from '../../src/gpu/genome';
import type { GameState, Action } from '../../src/game/state';

function aiThinkLocal(
  state: GameState,
  player: number,
  genome: Float32Array,
  nb: Int32Array,
): Action[] {
  const actions: Action[] = [];
  const desiredBySrc = new Map<number, number>();
  for (let cell = 0; cell < state.nodes.length; cell++) {
    if (state.nodes[cell].owner !== player) continue;
    const a = nnInferCell(genome, state, cell, player, nb);
    if (a < 0 || a >= NEIGHBOR_STRIDE) continue;
    const dst = nb[cell * NEIGHBOR_STRIDE + a];
    if (dst < 0) continue;
    desiredBySrc.set(cell, dst);
  }
  for (const f of state.flows) {
    if (f.player !== player) continue;
    const desired = desiredBySrc.get(f.src);
    if (desired === undefined || desired !== f.dst) {
      actions.push({ kind: 'toggleFlow', src: f.src, dst: f.dst, player });
    } else {
      desiredBySrc.delete(f.src);
    }
  }
  for (const [src, dst] of desiredBySrc) {
    actions.push({ kind: 'toggleFlow', src, dst, player });
  }
  return actions;
}

// Python uses repr(float(x)) for strength formatting; that's the same as JS
// `String(Number)` (shortest-round-trip representation) per IEEE 754 spec.
// Both runtimes use the same shortest-round-trip algorithm for doubles.
function reprFloat(x: number): string {
  // Match Python's repr(float) — shortest decimal that round-trips. JS's
  // default Number.prototype.toString is the same algorithm (ECMA-262 §7.1.12.1
  // and Python use the same Grisu/Ryū family). One subtlety: Python prints
  // "1.0" for whole numbers while JS prints "1". Normalize whole numbers.
  if (Number.isInteger(x) && Number.isFinite(x)) {
    // Python repr: integers-valued floats look like "10.0" or "-0.0".
    if (Object.is(x, -0)) return '-0.0';
    return `${x}.0`;
  }
  return String(x);
}

function stateHash(state: GameState): string {
  const h = createHash('sha256');
  h.update('NODES');
  for (const n of state.nodes) {
    const owner = n.owner === null ? -1 : n.owner;
    h.update(`${n.id}|${owner}|`);
    h.update(reprFloat(n.strength));
    h.update(';');
  }
  h.update('FLOWS');
  for (const f of state.flows) {
    h.update(`${f.src}-${f.dst}-${f.player};`);
  }
  h.update('TICK');
  h.update(String(state.tick));
  return h.digest('hex').slice(0, 16);
}

function run(): void {
  let state: GameState = makeInitialState(9, 2, 4);
  const rng = mulberry32(42);
  // Larger std so the NN produces diverse, non-trivial argmax actions.
  const genome = randomGenome(rng, 2.0);
  const nb = buildNeighborTable(state);

  console.log(`tick 0   hash ${stateHash(state)}`);

  // Seed actions to exercise apply_action toggle paths before NN takes over.
  const seedActions: Action[] = [
    { kind: 'toggleFlow', src: 10, dst: 11, player: 0 },
    { kind: 'toggleFlow', src: 10, dst: 21, player: 0 },
  ];
  for (const a of seedActions) {
    state = applyAction(state, a);
  }

  for (let t = 1; t <= 100; t++) {
    for (let p = 0; p < state.numPlayers; p++) {
      const actions = aiThinkLocal(state, p, genome, nb);
      for (const a of actions) {
        state = applyAction(state, a);
      }
    }
    if (t === 50) {
      state = applyAction(state, { kind: 'toggleFlow', src: 10, dst: 11, player: 0 });
    }
    state = step(state, 0.1);
    if (t % 10 === 0) {
      console.log(`tick ${String(t).padStart(3, ' ')} hash ${stateHash(state)}`);
    }
  }
}

run();
