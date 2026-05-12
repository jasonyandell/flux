// Regen-flow ruleset.
//
// Key differences from the original transfer-flow step:
//   • Outflow has no health cost. Instead, a cell with outflows forfeits its
//     own regen — that regen is converted into outgoing flux, split evenly
//     across the cell's outflows.
//   • Per-cell regen rate scales linearly with strength: regen(s) =
//     REGEN_BASE_PER_SEC * (1 + REGEN_SLOPE * (s - 1)).
//   • Damage is symmetric: friendly destinations gain `flux`, enemy
//     destinations lose `flux`. No attacker bonus on damage itself.
//   • On capture (strength ≤ 0 from net enemy contribution), ownership flips
//     to the largest contributor and the cell becomes strength = 1.
//     Any leftover damage past 0 is wasted.
//   • Strength is capped at MAX_STRENGTH. Overage past the cap is currently
//     discarded; overage-propagation through outflows is a v1 follow-up.
//
// Same GameState / Flow shape as the transfer ruleset, so the renderer and
// replay format need no schema changes — only metadata.ruleset === "regen-flow".

import {
  type Action,
  type Edge,
  type Flow,
  type GameState,
  type Owner,
  MAX_STRENGTH,
} from './state';

export const REGEN_BASE_PER_SEC = 0.5;
export const REGEN_SLOPE = 2.0;          // regen(s=MAX) = base * (1 + slope*(MAX-1))
export const CAPTURE_STRENGTH = 1.0;     // deterministic post-capture strength

export function regenRate(strength: number): number {
  return REGEN_BASE_PER_SEC * (1 + REGEN_SLOPE * (strength - 1));
}

type Force = number[]; // strength delta per player at one node
const zeroForce = (n: number): Force => Array(n).fill(0);

const adjacencyCache = new WeakMap<readonly Edge[], Set<number>>();

function edgeKey(a: number, b: number): number {
  return a < b ? a * 1e7 + b : b * 1e7 + a;
}

function adjacencySet(edges: readonly Edge[]): Set<number> {
  let s = adjacencyCache.get(edges);
  if (s) return s;
  s = new Set();
  for (const e of edges) s.add(edgeKey(e.a, e.b));
  adjacencyCache.set(edges, s);
  return s;
}

export function applyAction(state: GameState, action: Action): GameState {
  if (action.kind !== 'toggleFlow') return state;
  const src = state.nodes[action.src];
  if (!src || src.owner !== action.player) return state;
  if (!adjacencySet(state.edges).has(edgeKey(action.src, action.dst))) return state;
  const onEdge = state.flows.findIndex(f =>
    (f.src === action.src && f.dst === action.dst) ||
    (f.src === action.dst && f.dst === action.src),
  );
  if (onEdge < 0) {
    return { ...state, flows: [...state.flows, { src: action.src, dst: action.dst, player: action.player }] };
  }
  const existing = state.flows[onEdge];
  const exact = existing.src === action.src && existing.dst === action.dst && existing.player === action.player;
  const flows = state.flows.filter((_, j) => j !== onEdge);
  if (exact) return { ...state, flows };
  return { ...state, flows: [...flows, { src: action.src, dst: action.dst, player: action.player }] };
}

export function stepRegen(state: GameState, dt: number): GameState {
  const N = state.numPlayers;
  const forces: Force[] = state.nodes.map(() => zeroForce(N));

  // Count friendly outflows per cell.
  const outCount: number[] = new Array(state.nodes.length).fill(0);
  for (const flow of state.flows) {
    const src = state.nodes[flow.src];
    if (src.owner !== flow.player) continue;
    outCount[flow.src]++;
  }

  // Idle cells regen normally; sending cells forfeit.
  for (const node of state.nodes) {
    if (node.owner === null) continue;
    if (outCount[node.id] === 0) {
      forces[node.id][node.owner] += regenRate(node.strength) * dt;
    }
  }

  // Each outflow delivers regen(src) / K to its destination.
  const aliveFlows: Flow[] = [];
  for (const flow of state.flows) {
    const src = state.nodes[flow.src];
    if (src.owner !== flow.player) continue;
    aliveFlows.push(flow);
    const flux = (regenRate(src.strength) * dt) / outCount[flow.src];
    const dst = state.nodes[flow.dst];
    if (dst.owner === flow.player) {
      forces[flow.dst][flow.player] += flux;          // support
    } else {
      forces[flow.dst][flow.player] -= flux;          // attack (symmetric, no bonus)
    }
  }

  // Resolve per-cell state changes. force[p] is signed:
  //   positive  → owner-side support contribution (only their own slot is filled)
  //   negative  → attack contribution from player p (magnitude = -force[p])
  // Sum across all players to get net strength delta; capture if it drops to ≤ 0.
  const nodes = state.nodes.map(node => {
    const force = forces[node.id];
    let owner = node.owner;
    let strength = node.strength;

    for (let p = 0; p < N; p++) strength += force[p];

    if (strength < 0) {
      let bestP: Owner = null, bestDamage = 0;
      for (let p = 0; p < N; p++) {
        if (p !== owner && -force[p] > bestDamage) { bestDamage = -force[p]; bestP = p; }
      }
      if (bestP !== null) {
        owner = bestP;
        strength = CAPTURE_STRENGTH;
      } else {
        strength = 0;
      }
    }

    if (strength > MAX_STRENGTH) strength = MAX_STRENGTH;
    if (strength < 0) strength = 0;

    return { ...node, owner, strength };
  });

  return { ...state, nodes, flows: aliveFlows, tick: state.tick + 1 };
}
