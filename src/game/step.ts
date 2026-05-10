import {
  type Action,
  type Edge,
  type Flow,
  type GameState,
  type Owner,
  MAX_STRENGTH,
  MIN_STRENGTH_TO_SEND,
  REGEN_PER_SEC,
  TRANSFER_PER_SEC,
} from './state';

type Force = number[]; // strength contribution per player at one node

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

export function step(state: GameState, dt: number): GameState {
  const N = state.numPlayers;
  const forces: Force[] = state.nodes.map(() => zeroForce(N));

  for (const node of state.nodes) {
    if (node.owner !== null) forces[node.id][node.owner] += REGEN_PER_SEC * dt;
  }

  const aliveFlows: Flow[] = [];
  for (const flow of state.flows) {
    const src = state.nodes[flow.src];
    if (src.owner !== flow.player) continue;
    aliveFlows.push(flow);
    if (src.strength < MIN_STRENGTH_TO_SEND) continue;
    const k = TRANSFER_PER_SEC * dt;
    forces[flow.src][flow.player] -= k;
    forces[flow.dst][flow.player] += k;
  }

  const nodes = state.nodes.map(node => {
    const force = forces[node.id];
    let owner = node.owner;
    let strength = node.strength;

    if (owner !== null) {
      strength += force[owner];
      for (let p = 0; p < N; p++) if (p !== owner) strength -= force[p];
    } else {
      for (let p = 0; p < N; p++) if (force[p] > 0) strength -= force[p];
    }

    if (strength < 0) {
      let bestP: Owner = null, best = 0;
      for (let p = 0; p < N; p++) {
        if (p !== owner && force[p] > best) { best = force[p]; bestP = p; }
      }
      if (bestP !== null) { owner = bestP; strength = -strength; }
      else strength = 0;
    }

    if (strength > MAX_STRENGTH) strength = MAX_STRENGTH;
    if (strength < 0) strength = 0;

    return { ...node, owner, strength };
  });

  return { ...state, nodes, flows: aliveFlows, tick: state.tick + 1 };
}
