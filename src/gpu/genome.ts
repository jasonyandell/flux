import type { GameState, NodeId, Player } from '../game/state';
import { MAX_STRENGTH } from '../game/state';

export const NN_IN_DIM = 91;
export const NN_HIDDEN = 32;
export const NN_OUT_DIM = 19;
export const NEIGHBOR_STRIDE = 18;

export const WEIGHTS_PER_GENOME =
  NN_IN_DIM * NN_HIDDEN + NN_HIDDEN + NN_HIDDEN * NN_OUT_DIM + NN_OUT_DIM;
// = 91*32 + 32 + 32*19 + 19 = 2912 + 32 + 608 + 19 = 3571

export type Genome = Float32Array;

// Build per-cell neighbor index table sorted by (q, r) of relative offset for
// determinism. Length cells * NEIGHBOR_STRIDE. -1 = empty slot.
export function buildNeighborTable(state: GameState): Int32Array {
  const adj: number[][] = state.nodes.map(() => []);
  for (const e of state.edges) {
    adj[e.a].push(e.b);
    adj[e.b].push(e.a);
  }
  const out = new Int32Array(state.nodes.length * NEIGHBOR_STRIDE).fill(-1);
  for (let i = 0; i < state.nodes.length; i++) {
    const list = adj[i].slice().sort((a, b) => {
      // Stable order: by position.
      const pa = state.nodes[a].pos, pb = state.nodes[b].pos;
      if (pa.x !== pb.x) return pa.x - pb.x;
      return pa.y - pb.y;
    });
    for (let j = 0; j < Math.min(list.length, NEIGHBOR_STRIDE); j++) {
      out[i * NEIGHBOR_STRIDE + j] = list[j];
    }
  }
  return out;
}

// Box-Muller gaussian (mean=0, std=1) — pure
export function gaussian(rand: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randomGenome(rand: () => number, std = 0.5): Genome {
  const g = new Float32Array(WEIGHTS_PER_GENOME);
  for (let i = 0; i < g.length; i++) g[i] = gaussian(rand) * std;
  return g;
}

export function mutate(parent: Genome, rand: () => number, sigma = 0.05): Genome {
  const child = new Float32Array(parent.length);
  for (let i = 0; i < parent.length; i++) {
    child[i] = parent[i] + gaussian(rand) * sigma;
  }
  return child;
}

// JS forward pass for a single cell. Must match WGSL nn_forward for identical
// weights. Returns chosen action index in [0..NN_OUT_DIM-1]. 18 = do nothing.
export function nnInferCell(
  genome: Genome,
  state: GameState,
  cellId: NodeId,
  player: Player,
  neighbors: Int32Array,
): number {
  const W1off = 0;
  const b1off = NN_IN_DIM * NN_HIDDEN;
  const W2off = b1off + NN_HIDDEN;
  const b2off = W2off + NN_HIDDEN * NN_OUT_DIM;

  const h = new Float32Array(NN_HIDDEN);
  for (let j = 0; j < NN_HIDDEN; j++) h[j] = genome[b1off + j];

  const f0 = state.nodes[cellId].strength / MAX_STRENGTH;
  for (let j = 0; j < NN_HIDDEN; j++) h[j] += f0 * genome[W1off + j];

  for (let n = 0; n < NEIGHBOR_STRIDE; n++) {
    const nbId = neighbors[cellId * NEIGHBOR_STRIDE + n];
    let sFeat = 0, isFriendA = 0, isFriendB = 0, isEnemy = 0, isNeutral = 0;
    if (nbId >= 0) {
      sFeat = state.nodes[nbId].strength / MAX_STRENGTH;
      const o = state.nodes[nbId].owner;
      if (o === player) { isFriendA = 1; isFriendB = 1; }
      else if (o === null) { isNeutral = 1; }
      else { isEnemy = 1; }
    } else {
      isNeutral = 1;
    }
    const featBase = 1 + n * 5;
    for (let j = 0; j < NN_HIDDEN; j++) {
      h[j] +=
          sFeat * genome[W1off + (featBase + 0) * NN_HIDDEN + j]
        + isFriendA * genome[W1off + (featBase + 1) * NN_HIDDEN + j]
        + isFriendB * genome[W1off + (featBase + 2) * NN_HIDDEN + j]
        + isEnemy * genome[W1off + (featBase + 3) * NN_HIDDEN + j]
        + isNeutral * genome[W1off + (featBase + 4) * NN_HIDDEN + j];
    }
  }

  for (let j = 0; j < NN_HIDDEN; j++) {
    const x = h[j];
    const e1 = Math.exp(x), e2 = Math.exp(-x);
    h[j] = (e1 - e2) / (e1 + e2);
  }

  const out = new Float32Array(NN_OUT_DIM);
  for (let k = 0; k < NN_OUT_DIM; k++) out[k] = genome[b2off + k];
  for (let j = 0; j < NN_HIDDEN; j++) {
    const hj = h[j];
    for (let k = 0; k < NN_OUT_DIM; k++) {
      out[k] += hj * genome[W2off + j * NN_OUT_DIM + k];
    }
  }

  let bestK = 0, bestV = out[0];
  for (let k = 1; k < NN_OUT_DIM; k++) {
    if (out[k] > bestV) { bestV = out[k]; bestK = k; }
  }
  return bestK;
}
