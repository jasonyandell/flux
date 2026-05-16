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
  pos: Float32Array;        // (N, 2) row-major flat hex layout
  pos3d: Float32Array;      // (N, 3) row-major sphere projection
  coord: Int32Array;        // (N, 2) row-major axial (q, r)
  neighbors: Int32Array;    // (N, 6) row-major slot k → cell id (-1 = off-grid)
  radius: number;
  numPlayers: number;
};

export const SPHERE_RADIUS = 10.0;

export function axialToPixel(q: number, r: number): Vec2 {
  return { x: SQRT3 * q + (SQRT3 / 2) * r, y: (3 / 2) * r };
}

// Inverse Lambert azimuthal equal-area projection from the 2D hex disc onto a
// sphere. Center cell maps to the north pole; edge cells crowd toward (but
// don't collapse onto) the south pole. Equal-area keeps cell spacing roughly
// uniform on the sphere instead of ballooning the equator.
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
  let maxDist2d = 0;
  for (let i = 0; i < N; i++) {
    const [q, r] = cells[i];
    const p = axialToPixel(q, r);
    pos[i * 2] = p.x; pos[i * 2 + 1] = p.y;
    coord[i * 2] = q; coord[i * 2 + 1] = r;
    const d = Math.hypot(p.x, p.y);
    if (d > maxDist2d) maxDist2d = d;
  }

  // Lambert azimuthal: full disc has ρ ∈ [0, 2]; ρ=2 hits the south pole.
  // Pull back to 1.94 so the outermost ring stays visible as a tight band
  // rather than collapsing to a single point.
  const SCALE_DISC = 1.94 / Math.max(maxDist2d, 1e-6);
  const pos3d = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const x2 = pos[i * 2];
    const y2 = pos[i * 2 + 1];
    const rho = Math.hypot(x2, y2) * SCALE_DISC;
    const z = 1 - (rho * rho) / 2;
    const sinC = Math.sqrt(Math.max(0, 1 - z * z));
    const denom = Math.hypot(x2, y2) || 1;
    const ux = x2 / denom, uy = y2 / denom;
    pos3d[i * 3]     = ux * sinC * SPHERE_RADIUS;
    pos3d[i * 3 + 1] = uy * sinC * SPHERE_RADIUS;
    pos3d[i * 3 + 2] = z  * SPHERE_RADIUS;
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

  return { N, pos, pos3d, coord, neighbors, radius, numPlayers };
}

// Build a Board from sphere geometry shipped in the replay metadata. The
// sim writes (pos3d unit-sphere, neighbors) so the viewer renders against
// the exact graph the sim played on. `radius` here carries the icosphere
// subdivision level (used only for cell-spacing heuristics in the renderer).
export function buildSphereBoard(args: {
  pos3dUnit: Float32Array;     // length N*3, unit sphere
  neighbors: Int32Array;       // length N*K
  radius: number;              // subdivision level (cosmetic)
  numPlayers: number;
}): Board {
  const { pos3dUnit, neighbors, radius, numPlayers } = args;
  const N = pos3dUnit.length / 3;
  if (N !== Math.floor(N)) throw new Error('sphere board: pos3d length not divisible by 3');
  const pos3d = new Float32Array(N * 3);
  // Scale to renderer's sphere radius. The sim ships unit-sphere vertices.
  for (let i = 0; i < N * 3; i++) pos3d[i] = pos3dUnit[i] * SPHERE_RADIUS;
  // 2D pos field: azimuth/colatitude projection — never used by the sphere
  // renderer but keeps the Board type happy for any 2D-aware code path.
  const pos = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    const x = pos3dUnit[i * 3];
    const y = pos3dUnit[i * 3 + 1];
    const z = pos3dUnit[i * 3 + 2];
    pos[i * 2] = Math.atan2(y, x);
    pos[i * 2 + 1] = Math.acos(Math.max(-1, Math.min(1, z)));
  }
  const coord = new Int32Array(N * 2); // unused on sphere
  return { N, pos, pos3d, coord, neighbors, radius, numPlayers };
}
