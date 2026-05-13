/**
 * v2 board geometry — mirrors python/flux_v2/graph.py.
 *
 * Builds the hex grid, node positions, axial coords, and direct-neighbor table
 * (K=6). The replay file does not store geometry; the browser rebuilds it
 * deterministically from (radius, numPlayers).
 */
const SQRT3 = Math.sqrt(3.0);

export const K = 6;

// Hex axial directions, indexed by slot k (slot k and slot (k+3)%6 are opposites).
export const HEX_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [+1, 0], [+1, -1], [0, -1],
  [-1, 0], [-1, +1], [0, +1],
];

export type Vec2 = { x: number; y: number };

export type Board = {
  N: number;
  pos: Float32Array;        // (N, 2) row-major
  coord: Int32Array;        // (N, 2) row-major axial (q, r)
  neighbors: Int32Array;    // (N, 6) row-major slot k → cell id (-1 = off-grid)
  radius: number;
  numPlayers: number;
};

export function axialToPixel(q: number, r: number): Vec2 {
  return { x: SQRT3 * q + (SQRT3 / 2) * r, y: (3 / 2) * r };
}

export function buildBoard(radius: number, numPlayers: number): Board {
  const cells: Array<readonly [number, number]> = [];
  for (let q = -radius; q <= radius; q++) {
    const rMin = Math.max(-radius, -q - radius);
    const rMax = Math.min(radius, -q + radius);
    for (let r = rMin; r <= rMax; r++) cells.push([q, r] as const);
  }
  const N = cells.length;
  const id = new Map<string, number>();
  for (let i = 0; i < N; i++) id.set(`${cells[i][0]},${cells[i][1]}`, i);

  const pos = new Float32Array(N * 2);
  const coord = new Int32Array(N * 2);
  for (let i = 0; i < N; i++) {
    const [q, r] = cells[i];
    const p = axialToPixel(q, r);
    pos[i * 2] = p.x; pos[i * 2 + 1] = p.y;
    coord[i * 2] = q; coord[i * 2 + 1] = r;
  }

  const neighbors = new Int32Array(N * K).fill(-1);
  for (let i = 0; i < N; i++) {
    const [q, r] = cells[i];
    for (let k = 0; k < K; k++) {
      const [dq, dr] = HEX_DIRECTIONS[k];
      const j = id.get(`${q + dq},${r + dr}`);
      if (j !== undefined) neighbors[i * K + k] = j;
    }
  }

  return { N, pos, coord, neighbors, radius, numPlayers };
}
