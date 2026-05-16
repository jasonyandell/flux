/**
 * v2 scene renderer — three.js orthographic, instanced node mesh,
 * pressure-thickened flow arrows.
 *
 * Mirrors the v1 renderer's palette + layout but without the in-page
 * editing affordances (selection ring, drag line).
 */
import * as THREE from 'three';
import type { Board } from '../board';

export const PLAYER_COLORS_HEX = [
  0x4a90e2, 0xe24a4a, 0x4ae28a, 0xe2c44a,
  0xa44ae2, 0xe2884a, 0x4ae2e2, 0xe24a88,
  0x88e24a, 0xe2e24a, 0x4a88e2, 0xa4e24a,
];
const NEUTRAL = new THREE.Color(0x666666);
const DEAD = new THREE.Color(0x3a2a30);
const NODE_BASE_RADIUS = 0.36;
const VIEW_PADDING = 1.05;

const tmpMatrix = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpScale = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const tmpColor = new THREE.Color();
const ownerColors = PLAYER_COLORS_HEX.map(c => new THREE.Color(c));

export type Scene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  nodeInstanced: THREE.InstancedMesh;
  edgeLines: THREE.LineSegments;
  flowMesh: THREE.Mesh;
  domElement: HTMLCanvasElement;
  viewSize: number;
  nodeCount: number;
  worldHalfWidth: number;
  worldHalfHeight: number;
  freshness: Float32Array;     // per-node target, 1 on change, decays 1/iter advance
  displayed: Float32Array;     // per-node rendered value, time-smoothed toward target
  prevOwners: Int32Array;      // last owner per node (-1 neutral, -2 dead, ≥0 player)
  prevFlowSigs: Float64Array;  // last flow-membership signature per node
  lastFrameIdx: number;        // last rendered replay frame index, -1 = none yet
  fadeEnabled: boolean;        // when false, all nodes render at full brightness
};

export function setFadeEnabled(s: Scene, enabled: boolean): void {
  s.fadeEnabled = enabled;
}

// Pulse-on-change + fade-over-iters. Target decay is keyed off replay frame
// advance — pause/scrub holds the glow, fast-forward fades fast.
// FADE_PER_ITER = 1/N means ~N iters from full bright to fully faded.
const FADE_PER_ITER = 1 / 20;
const MIN_BRIGHTNESS = 0.25;
// Wall-clock slew rate on the *displayed* value tracking the target. Caps how
// fast brightness can jump per second, so rapid iter changes (fast forward,
// scrub back-and-forth) become a soft continuous pulse instead of hard flashes.
// 2.0/s ⇒ one full extreme-to-extreme transition takes 0.5s.
const FRESHNESS_RATE_PER_SEC = 2.0;

export function createScene(canvas: HTMLCanvasElement, board: Board): Scene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0a0a0a);

  const scene = new THREE.Scene();

  let xMax = 1, yMax = 1;
  for (let i = 0; i < board.N; i++) {
    const x = Math.abs(board.pos[i * 2]);
    const y = Math.abs(board.pos[i * 2 + 1]);
    if (x > xMax) xMax = x;
    if (y > yMax) yMax = y;
  }
  const maxAbs = Math.max(xMax, yMax);
  const viewSize = maxAbs * VIEW_PADDING;

  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.OrthographicCamera(
    -aspect * viewSize, aspect * viewSize,
    viewSize, -viewSize,
    0.1, 100,
  );
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  // Edge mesh — direct adjacency only (K=6).
  const edgePoints: number[] = [];
  for (let c = 0; c < board.N; c++) {
    for (let k = 0; k < 6; k++) {
      const d = board.neighbors[c * 6 + k];
      if (d < 0 || d < c) continue;             // dedupe
      const ax = board.pos[c * 2], ay = board.pos[c * 2 + 1];
      const bx = board.pos[d * 2], by = board.pos[d * 2 + 1];
      edgePoints.push(ax, ay, 0, bx, by, 0);
    }
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePoints, 3));
  const edgeLines = new THREE.LineSegments(
    edgeGeom, new THREE.LineBasicMaterial({ color: 0x2a3548 }),
  );
  scene.add(edgeLines);

  const nodeGeom = new THREE.CircleGeometry(NODE_BASE_RADIUS, 16);
  const nodeMat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const nodeInstanced = new THREE.InstancedMesh(nodeGeom, nodeMat, board.N);
  nodeInstanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(board.N * 3), 3);
  for (let i = 0; i < board.N; i++) {
    tmpPos.set(board.pos[i * 2], board.pos[i * 2 + 1], 0.1);
    tmpScale.setScalar(1);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    nodeInstanced.setMatrixAt(i, tmpMatrix);
    nodeInstanced.setColorAt(i, NEUTRAL);
  }
  nodeInstanced.instanceMatrix.needsUpdate = true;
  if (nodeInstanced.instanceColor) nodeInstanced.instanceColor.needsUpdate = true;
  scene.add(nodeInstanced);

  // Flows are real triangles (shaft as a quad, head as a triangle) so we get
  // visible thickness — WebGL ignores LineBasicMaterial.linewidth on every
  // major browser, so lines always rendered as 1px which is too faint here.
  const flowGeom = new THREE.BufferGeometry();
  flowGeom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
  flowGeom.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
  const flowMat = new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    transparent: true, depthWrite: false,
  });
  const flowMesh = new THREE.Mesh(flowGeom, flowMat);
  flowMesh.frustumCulled = false;
  scene.add(flowMesh);

  // Start everything fully bright; the board opens with a glow that fades as
  // iters advance and nodes settle without further config changes.
  const freshness = new Float32Array(board.N);
  const displayed = new Float32Array(board.N);
  const prevOwners = new Int32Array(board.N);
  const prevFlowSigs = new Float64Array(board.N);
  for (let i = 0; i < board.N; i++) {
    freshness[i] = 1;
    displayed[i] = 1;
    prevOwners[i] = -1;
  }

  return {
    renderer, scene, camera,
    nodeInstanced, edgeLines, flowMesh,
    domElement: canvas,
    viewSize,
    nodeCount: board.N,
    worldHalfWidth: xMax,
    worldHalfHeight: yMax,
    freshness,
    displayed,
    prevOwners,
    prevFlowSigs,
    lastFrameIdx: -1,
    fadeEnabled: true,
  };
}

export type FrameRender = {
  owners: Int8Array;
  strengths: Uint8Array;
  flows: { src: number; dst: number; player: number; pressure: number }[];
};

export function updateScene(s: Scene, board: Board, frame: FrameRender, frameIdx: number = 0, dt: number = 0): void {
  // Per-node flow signature: sum of flow keys (src/dst/player, ignoring
  // pressure since it's continuous) for every flow touching the node.
  // Permutation-invariant; different multisets → different sums in practice.
  // Simultaneously: per-node max-pressure-touching-node, and the frame's max
  // pressure so we can auto-scale (same convention the arrow widths use).
  const flowSig = new Float64Array(board.N);
  const nodePressure = new Float32Array(board.N);
  let frameMaxPressureSig = 0;
  for (let i = 0; i < frame.flows.length; i++) {
    const f = frame.flows[i];
    const k = (f.src + 1) * 7919 + (f.dst + 1) * 191 + (f.player + 1);
    flowSig[f.src] += k;
    flowSig[f.dst] += k;
    if (f.pressure > nodePressure[f.src]) nodePressure[f.src] = f.pressure;
    if (f.pressure > nodePressure[f.dst]) nodePressure[f.dst] = f.pressure;
    if (f.pressure > frameMaxPressureSig) frameMaxPressureSig = f.pressure;
  }
  const pressureScale = 1 / Math.max(frameMaxPressureSig, 1.0);

  // Decay step keyed to iter advance: 0 if paused or scrubbed back, N if the
  // player jumped forward N frames since last render.
  const delta = s.lastFrameIdx < 0 ? 0 : frameIdx - s.lastFrameIdx;
  const decay = delta > 0 ? delta * FADE_PER_ITER : 0;
  s.lastFrameIdx = frameIdx;

  // Time-based slew cap on the displayed value. dt comes from the frame loop;
  // 0 (no time passed) means render whatever's currently displayed unchanged.
  const maxSlew = FRESHNESS_RATE_PER_SEC * dt;

  for (let i = 0; i < board.N; i++) {
    const owner = frame.owners[i];
    if (owner !== s.prevOwners[i] || flowSig[i] !== s.prevFlowSigs[i]) {
      s.freshness[i] = 1;
    } else if (decay > 0 && s.freshness[i] > 0) {
      s.freshness[i] = Math.max(0, s.freshness[i] - decay);
    }
    s.prevOwners[i] = owner;
    s.prevFlowSigs[i] = flowSig[i];

    // Ease the displayed value toward the iter-based target at a wall-clock
    // rate cap. A single 0→1 snap (config change) plays out over ~0.5s, and
    // rapid toggle bursts hover mid-range instead of flashing.
    const target = s.freshness[i];
    const diff = target - s.displayed[i];
    if (maxSlew <= 0 || Math.abs(diff) <= maxSlew) s.displayed[i] = target;
    else s.displayed[i] += diff < 0 ? -maxSlew : maxSlew;

    // `strengths` is the writer's uint8 quantization of [0, max_strength];
    // we only need the 0..1 ratio for sizing (max_strength is now carried
    // in the FLXR v3 header, not a compile-time constant).
    const ratio = frame.strengths[i] / 255;
    const scale = 0.45 + ratio * 1.0;
    tmpPos.set(board.pos[i * 2], board.pos[i * 2 + 1], 0.1);
    tmpScale.setScalar(scale);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    s.nodeInstanced.setMatrixAt(i, tmpMatrix);

    const base = owner === -1 ? NEUTRAL
               : owner === -2 ? DEAD
               : ownerColors[owner % ownerColors.length];
    // Above-base headroom is split 80% age-delta (smoothed freshness) and
    // 20% per-frame-scaled max pressure touching this node. Max pressure on a
    // long-static node tops out at 20% of headroom over base; max pressure
    // *and* a fresh config change adds the remaining 80% for full brightness.
    const pNorm = nodePressure[i] * pressureScale;
    const mix = s.fadeEnabled ? 0.8 * s.displayed[i] + 0.2 * pNorm : 1;
    const brightness = MIN_BRIGHTNESS + (1 - MIN_BRIGHTNESS) * mix;
    tmpColor.copy(base).multiplyScalar(brightness);
    s.nodeInstanced.setColorAt(i, tmpColor);
  }
  s.nodeInstanced.instanceMatrix.needsUpdate = true;
  if (s.nodeInstanced.instanceColor) s.nodeInstanced.instanceColor.needsUpdate = true;

  // Per-frame auto-scale for thickness: normalize each flow's pressure to
  // the frame's max instead of MAX_EDGE. In practice pressure stays in the
  // 0–20 range (regen per tick is small vs MAX_EDGE=100), so a fixed-cap
  // mapping squishes everything into "near MIN" and arrows look uniform.
  // Auto-scale makes the heaviest carrier in each frame visually pop and
  // shows the spread between active and idle outflows.
  let frameMaxPressure = 0;
  for (let i = 0; i < frame.flows.length; i++) {
    const p = frame.flows[i].pressure;
    if (p > frameMaxPressure) frameMaxPressure = p;
  }
  // Floor the divisor so an all-idle frame doesn't divide by zero and all
  // arrows don't suddenly spike to max width when there's no real flow.
  const pressureNorm = Math.max(frameMaxPressure, 1.0);

  // Flow rendering: each flow is one quad (shaft) + one triangle (head).
  // Vertex layout per flow (7 verts, 3 triangles, 9 indices):
  //   0: shaft source left      4: head left wing
  //   1: shaft source right     5: head right wing
  //   2: shaft tail left        6: head tip
  //   3: shaft tail right
  // Triangles: (0,1,3), (0,3,2), (4,6,5).
  const nFlows = frame.flows.length;
  const VERTS_PER_FLOW = 7;
  const INDICES_PER_FLOW = 9;
  const positions = new Float32Array(nFlows * VERTS_PER_FLOW * 3);
  const colors = new Float32Array(nFlows * VERTS_PER_FLOW * 3);
  const indices = new Uint32Array(nFlows * INDICES_PER_FLOW);

  // Shaft + head thickness scale strongly with pressure so the eye can read
  // pressure at a glance: idle outflows (set but pressure=0) render as a hair
  // line, fully-saturated edges render as a fat band. Curve uses pressure^0.7
  // so the bottom of the range is visually spread out (mid-pressure is more
  // visible than a linear ramp would make it).
  const SHAFT_HALF_MIN = 0.015;
  const SHAFT_HALF_MAX = 0.32;
  const HEAD_HALF_MIN = 0.06;
  const HEAD_HALF_MAX = 0.46;
  const HEAD_LEN_MIN = 0.10;
  const HEAD_LEN_MAX = 0.34;
  const SHAFT_INSET = 0.45;
  const Z = 0.3;

  for (let i = 0; i < nFlows; i++) {
    const f = frame.flows[i];
    const ax = board.pos[f.src * 2], ay = board.pos[f.src * 2 + 1];
    const bx = board.pos[f.dst * 2], by = board.pos[f.dst * 2 + 1];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;            // perpendicular unit
    const powLin = Math.max(0, Math.min(1, f.pressure / pressureNorm));
    // Compress low end so even modest pressure is visibly thicker than idle.
    const pow = Math.pow(powLin, 0.7);

    // Where the arrow ends: closer to dst at high pressure, more retracted
    // at low pressure so an idle flow doesn't visually overrun.
    const shaftEndT = 0.55 + 0.30 * pow;
    const sx = ax + ux * SHAFT_INSET, sy = ay + uy * SHAFT_INSET;
    const tipX = ax + dx * shaftEndT, tipY = ay + dy * shaftEndT;

    // Geometry sizes.
    const shaftHalf = SHAFT_HALF_MIN + (SHAFT_HALF_MAX - SHAFT_HALF_MIN) * pow;
    const headHalf  = HEAD_HALF_MIN  + (HEAD_HALF_MAX  - HEAD_HALF_MIN)  * pow;
    const headLen   = HEAD_LEN_MIN   + (HEAD_LEN_MAX   - HEAD_LEN_MIN)   * pow;
    const tailX = tipX - ux * headLen;
    const tailY = tipY - uy * headLen;

    // Shaft corners (quad).
    const v0x = sx + px * shaftHalf, v0y = sy + py * shaftHalf;     // src left
    const v1x = sx - px * shaftHalf, v1y = sy - py * shaftHalf;     // src right
    const v2x = tailX + px * shaftHalf, v2y = tailY + py * shaftHalf;  // tail left
    const v3x = tailX - px * shaftHalf, v3y = tailY - py * shaftHalf;  // tail right

    // Head triangle: wider than shaft, tip at tip.
    const v4x = tailX + px * headHalf, v4y = tailY + py * headHalf; // head left
    const v5x = tailX - px * headHalf, v5y = tailY - py * headHalf; // head right
    const v6x = tipX,                  v6y = tipY;                  // tip

    const vBase = i * VERTS_PER_FLOW * 3;
    positions[vBase + 0]  = v0x; positions[vBase + 1]  = v0y; positions[vBase + 2]  = Z;
    positions[vBase + 3]  = v1x; positions[vBase + 4]  = v1y; positions[vBase + 5]  = Z;
    positions[vBase + 6]  = v2x; positions[vBase + 7]  = v2y; positions[vBase + 8]  = Z;
    positions[vBase + 9]  = v3x; positions[vBase + 10] = v3y; positions[vBase + 11] = Z;
    positions[vBase + 12] = v4x; positions[vBase + 13] = v4y; positions[vBase + 14] = Z;
    positions[vBase + 15] = v5x; positions[vBase + 16] = v5y; positions[vBase + 17] = Z;
    positions[vBase + 18] = v6x; positions[vBase + 19] = v6y; positions[vBase + 20] = Z;

    tmpColor.copy(ownerColors[f.player % ownerColors.length]);
    // Brightness scales with pressure too so idle outflows aren't just thin
    // but also dim — pressure correlates with "actually doing work."
    const sourceBright = 0.20 + 0.40 * pow;
    const tipBright    = 0.45 + 0.55 * pow;
    const headBright   = 0.65 + 0.35 * pow;

    // src corners use source brightness; tail corners use tip brightness;
    // head wings use head brightness; tip itself maxed out.
    const writeVert = (vi: number, b: number) => {
      const o = vBase + vi * 3;
      colors[o + 0] = tmpColor.r * b;
      colors[o + 1] = tmpColor.g * b;
      colors[o + 2] = tmpColor.b * b;
    };
    writeVert(0, sourceBright);
    writeVert(1, sourceBright);
    writeVert(2, tipBright);
    writeVert(3, tipBright);
    writeVert(4, headBright);
    writeVert(5, headBright);
    writeVert(6, headBright);

    const iBase = i * INDICES_PER_FLOW;
    const vi0 = i * VERTS_PER_FLOW;
    // Shaft (two triangles).
    indices[iBase + 0] = vi0 + 0; indices[iBase + 1] = vi0 + 1; indices[iBase + 2] = vi0 + 3;
    indices[iBase + 3] = vi0 + 0; indices[iBase + 4] = vi0 + 3; indices[iBase + 5] = vi0 + 2;
    // Head (one triangle).
    indices[iBase + 6] = vi0 + 4; indices[iBase + 7] = vi0 + 6; indices[iBase + 8] = vi0 + 5;
  }

  s.flowMesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  s.flowMesh.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  s.flowMesh.geometry.setIndex(new THREE.Uint32BufferAttribute(indices, 1));
  s.flowMesh.geometry.computeBoundingSphere();
}

export function rebuildSceneGeometry(s: Scene, board: Board): void {
  s.scene.remove(s.nodeInstanced);
  s.nodeInstanced.dispose();
  s.nodeInstanced.geometry.dispose();
  (s.nodeInstanced.material as THREE.Material).dispose();
  s.scene.remove(s.edgeLines);
  s.edgeLines.geometry.dispose();
  (s.edgeLines.material as THREE.Material).dispose();

  let xMax = 1, yMax = 1;
  for (let i = 0; i < board.N; i++) {
    const x = Math.abs(board.pos[i * 2]);
    const y = Math.abs(board.pos[i * 2 + 1]);
    if (x > xMax) xMax = x;
    if (y > yMax) yMax = y;
  }
  s.worldHalfWidth = xMax; s.worldHalfHeight = yMax;
  s.viewSize = Math.max(xMax, yMax) * VIEW_PADDING;
  const w = s.renderer.domElement.clientWidth || window.innerWidth;
  const h = s.renderer.domElement.clientHeight || window.innerHeight;
  const aspect = w / h;
  s.camera.left = -aspect * s.viewSize;
  s.camera.right = aspect * s.viewSize;
  s.camera.top = s.viewSize;
  s.camera.bottom = -s.viewSize;
  s.camera.position.set(0, 0, 10);
  s.camera.updateProjectionMatrix();

  const edgePoints: number[] = [];
  for (let c = 0; c < board.N; c++) {
    for (let k = 0; k < 6; k++) {
      const d = board.neighbors[c * 6 + k];
      if (d < 0 || d < c) continue;
      const ax = board.pos[c * 2], ay = board.pos[c * 2 + 1];
      const bx = board.pos[d * 2], by = board.pos[d * 2 + 1];
      edgePoints.push(ax, ay, 0, bx, by, 0);
    }
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePoints, 3));
  s.edgeLines = new THREE.LineSegments(
    edgeGeom, new THREE.LineBasicMaterial({ color: 0x2a3548 }),
  );
  s.scene.add(s.edgeLines);

  const nodeGeom = new THREE.CircleGeometry(NODE_BASE_RADIUS, 16);
  const nodeMat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const nodeInstanced = new THREE.InstancedMesh(nodeGeom, nodeMat, board.N);
  nodeInstanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(board.N * 3), 3);
  for (let i = 0; i < board.N; i++) {
    tmpPos.set(board.pos[i * 2], board.pos[i * 2 + 1], 0.1);
    tmpScale.setScalar(1);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    nodeInstanced.setMatrixAt(i, tmpMatrix);
    nodeInstanced.setColorAt(i, NEUTRAL);
  }
  nodeInstanced.instanceMatrix.needsUpdate = true;
  if (nodeInstanced.instanceColor) nodeInstanced.instanceColor.needsUpdate = true;
  s.scene.add(nodeInstanced);
  s.nodeInstanced = nodeInstanced;
  s.nodeCount = board.N;

  // Fresh replay: open with a glow, reset the "last frame" anchor.
  s.freshness = new Float32Array(board.N);
  s.displayed = new Float32Array(board.N);
  s.prevOwners = new Int32Array(board.N);
  s.prevFlowSigs = new Float64Array(board.N);
  for (let i = 0; i < board.N; i++) {
    s.freshness[i] = 1;
    s.displayed[i] = 1;
    s.prevOwners[i] = -1;
  }
  s.lastFrameIdx = -1;
}

export function resizeRenderer(s: Scene): void {
  const w = window.innerWidth, h = window.innerHeight;
  s.renderer.setSize(w, h, false);
  const aspect = w / h;
  s.camera.left = -aspect * s.viewSize;
  s.camera.right = aspect * s.viewSize;
  s.camera.top = s.viewSize;
  s.camera.bottom = -s.viewSize;
  s.camera.updateProjectionMatrix();
}

export function render(s: Scene): void {
  s.renderer.render(s.scene, s.camera);
}
