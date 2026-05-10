import type { Edge, GameNode, GameState } from './state';

const SQRT3 = Math.sqrt(3);

export function axialToPixel(q: number, r: number): { x: number; y: number } {
  return { x: SQRT3 * q + (SQRT3 / 2) * r, y: (3 / 2) * r };
}

export function hexDistance(q1: number, r1: number, q2: number, r2: number): number {
  const dq = q1 - q2, dr = r1 - r2;
  return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));
}

export function makeInitialState(radius = 18, distance = 2): GameState {
  const cells: { q: number; r: number }[] = [];
  for (let q = -radius; q <= radius; q++) {
    const rMin = Math.max(-radius, -q - radius);
    const rMax = Math.min(radius, -q + radius);
    for (let r = rMin; r <= rMax; r++) cells.push({ q, r });
  }

  const idByKey = new Map<string, number>();
  cells.forEach((c, i) => idByKey.set(`${c.q},${c.r}`, i));

  const nodes: GameNode[] = cells.map(({ q, r }, i) => {
    const { x, y } = axialToPixel(q, r);
    return { id: i, pos: { x, y }, owner: null, strength: 10 };
  });

  const p0 = idByKey.get(`${-radius},0`)!;
  const p1 = idByKey.get(`${radius},0`)!;
  nodes[p0].owner = 0; nodes[p0].strength = 30;
  nodes[p1].owner = 1; nodes[p1].strength = 30;

  const edges: Edge[] = [];
  for (let i = 0; i < cells.length; i++) {
    const { q, r } = cells[i];
    for (let dq = -distance; dq <= distance; dq++) {
      for (let dr = -distance; dr <= distance; dr++) {
        if (dq === 0 && dr === 0) continue;
        if (Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr)) > distance) continue;
        const j = idByKey.get(`${q + dq},${r + dr}`);
        if (j === undefined || j <= i) continue;
        edges.push({ a: i, b: j });
      }
    }
  }

  return { nodes, edges, flows: [], tick: 0, numPlayers: 2 };
}
