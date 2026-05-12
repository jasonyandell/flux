import * as THREE from 'three';
import type { GameState } from '../game/state';
import { MAX_STRENGTH } from '../game/state';

export const COLORS = [
  0x4a90e2, 0xe24a4a, 0x4ae28a, 0xe2c44a,
  0xa44ae2, 0xe2884a, 0x4ae2e2, 0xe24a88,
  0x88e24a, 0xe2e24a, 0x4a88e2, 0xa4e24a,
];
const NEUTRAL = new THREE.Color(0x666666);
const NODE_BASE_RADIUS = 0.36;
const VIEW_PADDING = 1.05;

export type Scene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  nodeInstanced: THREE.InstancedMesh;
  edgeLines: THREE.LineSegments;
  selectedHighlight: THREE.Mesh;
  flowLines: THREE.LineSegments;
  dragLine: THREE.Line;
  domElement: HTMLCanvasElement;
  viewSize: number;
  nodeCount: number;
  nodePositions: Float32Array;
  worldHalfWidth: number;
  worldHalfHeight: number;
};

export const NODE_PICK_RADIUS = NODE_BASE_RADIUS;

const tmpMatrix = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpScale = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const tmpColor = new THREE.Color();

const ownerColors = COLORS.map(c => new THREE.Color(c));

export function createScene(canvas: HTMLCanvasElement, state: GameState): Scene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0a0a0a);

  const scene = new THREE.Scene();

  let xMax = 1, yMax = 1;
  for (const n of state.nodes) {
    if (Math.abs(n.pos.x) > xMax) xMax = Math.abs(n.pos.x);
    if (Math.abs(n.pos.y) > yMax) yMax = Math.abs(n.pos.y);
  }
  const maxAbs = Math.max(xMax, yMax);
  const viewSize = maxAbs * VIEW_PADDING;
  const worldHalfWidth = xMax;
  const worldHalfHeight = yMax;

  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.OrthographicCamera(
    -aspect * viewSize, aspect * viewSize,
    viewSize, -viewSize,
    0.1, 100,
  );
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  const edgePoints: number[] = [];
  for (const e of state.edges) {
    const a = state.nodes[e.a].pos, b = state.nodes[e.b].pos;
    edgePoints.push(a.x, a.y, 0, b.x, b.y, 0);
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePoints, 3));
  const edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: 0x2a3548 }));
  scene.add(edgeLines);

  const nodeGeom = new THREE.CircleGeometry(NODE_BASE_RADIUS, 12);
  const nodeMat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const nodeInstanced = new THREE.InstancedMesh(nodeGeom, nodeMat, state.nodes.length);
  nodeInstanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(state.nodes.length * 3), 3);
  for (let i = 0; i < state.nodes.length; i++) {
    const n = state.nodes[i];
    tmpPos.set(n.pos.x, n.pos.y, 0.1);
    tmpScale.setScalar(1);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    nodeInstanced.setMatrixAt(i, tmpMatrix);
    nodeInstanced.setColorAt(i, NEUTRAL);
  }
  nodeInstanced.instanceMatrix.needsUpdate = true;
  if (nodeInstanced.instanceColor) nodeInstanced.instanceColor.needsUpdate = true;
  scene.add(nodeInstanced);

  const highlightGeom = new THREE.RingGeometry(NODE_BASE_RADIUS * 1.4, NODE_BASE_RADIUS * 1.6, 24);
  const highlightMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const selectedHighlight = new THREE.Mesh(highlightGeom, highlightMat);
  selectedHighlight.visible = false;
  selectedHighlight.position.z = 0.15;
  scene.add(selectedHighlight);

  const flowGeom = new THREE.BufferGeometry();
  flowGeom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
  flowGeom.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
  const flowLines = new THREE.LineSegments(flowGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(flowLines);

  const dragGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0.2),
    new THREE.Vector3(0, 0, 0.2),
  ]);
  const dragMat = new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.2, gapSize: 0.15 });
  const dragLine = new THREE.Line(dragGeom, dragMat);
  dragLine.computeLineDistances();
  dragLine.visible = false;
  scene.add(dragLine);

  const nodePositions = new Float32Array(state.nodes.length * 2);
  for (let i = 0; i < state.nodes.length; i++) {
    nodePositions[i * 2 + 0] = state.nodes[i].pos.x;
    nodePositions[i * 2 + 1] = state.nodes[i].pos.y;
  }

  return {
    renderer, scene, camera,
    nodeInstanced, edgeLines, selectedHighlight,
    flowLines, dragLine,
    domElement: canvas,
    viewSize,
    nodeCount: state.nodes.length,
    nodePositions,
    worldHalfWidth,
    worldHalfHeight,
  };
}

/** Swap out the geometry (instanced mesh + edge lines + camera bounds) to
 *  match a new state with a different shape (e.g. replay loaded with a
 *  different board radius). Disposes old geometry to avoid GPU leaks. */
export function rebuildSceneGeometry(s: Scene, state: GameState): void {
  // Remove + dispose the old node mesh and edge lines. Also dispose the
  // per-instance attribute buffers so GPU memory doesn't leak across boards.
  s.scene.remove(s.nodeInstanced);
  s.nodeInstanced.dispose();
  s.nodeInstanced.geometry.dispose();
  (s.nodeInstanced.material as THREE.Material).dispose();

  s.scene.remove(s.edgeLines);
  s.edgeLines.geometry.dispose();
  (s.edgeLines.material as THREE.Material).dispose();

  // Recompute camera bounds from the new state.
  let xMax = 1, yMax = 1;
  for (const n of state.nodes) {
    if (Math.abs(n.pos.x) > xMax) xMax = Math.abs(n.pos.x);
    if (Math.abs(n.pos.y) > yMax) yMax = Math.abs(n.pos.y);
  }
  s.worldHalfWidth = xMax;
  s.worldHalfHeight = yMax;
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

  // Rebuild edges.
  const edgePoints: number[] = [];
  for (const e of state.edges) {
    const a = state.nodes[e.a].pos, b = state.nodes[e.b].pos;
    edgePoints.push(a.x, a.y, 0, b.x, b.y, 0);
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePoints, 3));
  s.edgeLines = new THREE.LineSegments(
    edgeGeom, new THREE.LineBasicMaterial({ color: 0x2a3548 }),
  );
  s.scene.add(s.edgeLines);

  // Rebuild node InstancedMesh at the new count.
  const nodeGeom = new THREE.CircleGeometry(NODE_BASE_RADIUS, 12);
  const nodeMat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const nodeInstanced = new THREE.InstancedMesh(nodeGeom, nodeMat, state.nodes.length);
  nodeInstanced.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(state.nodes.length * 3), 3,
  );
  for (let i = 0; i < state.nodes.length; i++) {
    const n = state.nodes[i];
    tmpPos.set(n.pos.x, n.pos.y, 0.1);
    tmpScale.setScalar(1);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    nodeInstanced.setMatrixAt(i, tmpMatrix);
    nodeInstanced.setColorAt(i, NEUTRAL);
  }
  nodeInstanced.instanceMatrix.needsUpdate = true;
  if (nodeInstanced.instanceColor) nodeInstanced.instanceColor.needsUpdate = true;
  s.scene.add(nodeInstanced);
  s.nodeInstanced = nodeInstanced;
  s.nodeCount = state.nodes.length;

  // Refresh cached node positions.
  s.nodePositions = new Float32Array(state.nodes.length * 2);
  for (let i = 0; i < state.nodes.length; i++) {
    s.nodePositions[i * 2 + 0] = state.nodes[i].pos.x;
    s.nodePositions[i * 2 + 1] = state.nodes[i].pos.y;
  }
}

export function updateScene(s: Scene, state: GameState, selected: number | null): void {
  for (let i = 0; i < state.nodes.length; i++) {
    const node = state.nodes[i];
    const scale = 0.45 + (node.strength / MAX_STRENGTH) * 1.0;
    tmpPos.set(node.pos.x, node.pos.y, 0.1);
    tmpScale.setScalar(scale);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    s.nodeInstanced.setMatrixAt(i, tmpMatrix);
    if (node.owner === null) {
      s.nodeInstanced.setColorAt(i, NEUTRAL);
    } else {
      s.nodeInstanced.setColorAt(i, ownerColors[node.owner % ownerColors.length]);
    }
  }
  s.nodeInstanced.instanceMatrix.needsUpdate = true;
  if (s.nodeInstanced.instanceColor) s.nodeInstanced.instanceColor.needsUpdate = true;

  if (selected !== null) {
    const node = state.nodes[selected];
    s.selectedHighlight.position.x = node.pos.x;
    s.selectedHighlight.position.y = node.pos.y;
    const scale = 0.45 + (node.strength / MAX_STRENGTH) * 1.0;
    s.selectedHighlight.scale.setScalar(scale);
    s.selectedHighlight.visible = true;
  } else {
    s.selectedHighlight.visible = false;
  }

  // Each flow renders as 3 line segments (shaft + 2 arrowhead wings) so the
  // direction reads at a glance against the underlying edge mesh.
  // WebGL line widths are ignored across all browsers (always 1px), so we
  // fake thickness by stacking 1–5 perpendicularly-offset copies of each
  // flow — strong flows render as a thick band, weak ones stay as a thin
  // single line. Brightness + reach + arrowhead size also scale with power.
  const nFlows = state.flows.length;
  const VERTS_PER_FLOW = 6;
  const MAX_STACK = 5;
  const STACK_OFFSET = 0.04;           // perpendicular spacing between stacked lines
  const positions = new Float32Array(nFlows * MAX_STACK * VERTS_PER_FLOW * 3);
  const colors = new Float32Array(nFlows * MAX_STACK * VERTS_PER_FLOW * 3);
  const SOURCE_BRIGHTNESS_MIN = 0.05;
  const SOURCE_BRIGHTNESS_MAX = 0.4;
  const HEAD_LEN_BASE = 0.16;
  const HEAD_HALF_BASE = 0.10;
  const SHAFT_INSET = 0.45;
  const Z = 0.3;
  for (let i = 0; i < nFlows; i++) {
    const flow = state.flows[i];
    const aNode = state.nodes[flow.src];
    const a = aNode.pos, b = state.nodes[flow.dst].pos;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;  // perpendicular unit
    const pow = Math.max(0, Math.min(1, aNode.strength / MAX_STRENGTH));
    const shaftEndT = 0.5 + 0.3 * pow;
    const sx = a.x + ux * SHAFT_INSET, sy = a.y + uy * SHAFT_INSET;
    const tx = a.x + dx * shaftEndT, ty = a.y + dy * shaftEndT;
    const headLen = HEAD_LEN_BASE * (0.7 + 0.6 * pow);
    const headHalf = HEAD_HALF_BASE * (0.7 + 0.6 * pow);
    const backX = tx - ux * headLen, backY = ty - uy * headLen;
    const lx = backX + px * headHalf, ly = backY + py * headHalf;
    const rx = backX - px * headHalf, ry = backY - py * headHalf;
    tmpColor.copy(ownerColors[flow.player % ownerColors.length]);
    const tipBrightness = 0.55 + 0.45 * pow;
    const sourceBrightness = SOURCE_BRIGHTNESS_MIN + (SOURCE_BRIGHTNESS_MAX - SOURCE_BRIGHTNESS_MIN) * pow;
    // Power → stack count: 1 line at pow=0, up to 5 at pow=1.
    const stackCount = Math.max(1, Math.min(MAX_STACK, Math.round(1 + (MAX_STACK - 1) * pow)));
    const stackBase = i * MAX_STACK * VERTS_PER_FLOW * 3;
    for (let k = 0; k < MAX_STACK; k++) {
      const koff = stackBase + k * VERTS_PER_FLOW * 3;
      if (k < stackCount) {
        // Center the stack on the shaft: idx ∈ [-(n-1)/2, (n-1)/2].
        const idx = k - (stackCount - 1) / 2;
        const ox = px * idx * STACK_OFFSET;
        const oy = py * idx * STACK_OFFSET;
        positions[koff + 0]  = sx + ox; positions[koff + 1]  = sy + oy; positions[koff + 2]  = Z;
        positions[koff + 3]  = tx + ox; positions[koff + 4]  = ty + oy; positions[koff + 5]  = Z;
        positions[koff + 6]  = tx + ox; positions[koff + 7]  = ty + oy; positions[koff + 8]  = Z;
        positions[koff + 9]  = lx + ox; positions[koff + 10] = ly + oy; positions[koff + 11] = Z;
        positions[koff + 12] = tx + ox; positions[koff + 13] = ty + oy; positions[koff + 14] = Z;
        positions[koff + 15] = rx + ox; positions[koff + 16] = ry + oy; positions[koff + 17] = Z;
        colors[koff + 0]  = tmpColor.r * sourceBrightness;
        colors[koff + 1]  = tmpColor.g * sourceBrightness;
        colors[koff + 2]  = tmpColor.b * sourceBrightness;
        for (let v = 1; v < VERTS_PER_FLOW; v++) {
          colors[koff + v * 3 + 0] = tmpColor.r * tipBrightness;
          colors[koff + v * 3 + 1] = tmpColor.g * tipBrightness;
          colors[koff + v * 3 + 2] = tmpColor.b * tipBrightness;
        }
      } else {
        // Degenerate stack — collapse to a zero-length point at the tip so
        // nothing visible renders. Colors zeroed for safety.
        for (let v = 0; v < VERTS_PER_FLOW; v++) {
          positions[koff + v * 3 + 0] = tx;
          positions[koff + v * 3 + 1] = ty;
          positions[koff + v * 3 + 2] = Z;
          colors[koff + v * 3 + 0] = 0;
          colors[koff + v * 3 + 1] = 0;
          colors[koff + v * 3 + 2] = 0;
        }
      }
    }
  }
  s.flowLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  s.flowLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  s.flowLines.geometry.computeBoundingSphere();
}

export function setDragLine(s: Scene, a: { x: number; y: number } | null, b: { x: number; y: number } | null): void {
  if (a === null || b === null) {
    s.dragLine.visible = false;
    return;
  }
  const positions = s.dragLine.geometry.getAttribute('position') as THREE.BufferAttribute;
  positions.setXYZ(0, a.x, a.y, 0.2);
  positions.setXYZ(1, b.x, b.y, 0.2);
  positions.needsUpdate = true;
  s.dragLine.geometry.computeBoundingSphere();
  s.dragLine.computeLineDistances();
  s.dragLine.visible = true;
}

function updateFrustum(s: Scene): void {
  const w = s.renderer.domElement.clientWidth || window.innerWidth;
  const h = s.renderer.domElement.clientHeight || window.innerHeight;
  const aspect = w / h;
  s.camera.left = -aspect * s.viewSize;
  s.camera.right = aspect * s.viewSize;
  s.camera.top = s.viewSize;
  s.camera.bottom = -s.viewSize;
  s.camera.updateProjectionMatrix();
}

export function clampCamera(s: Scene): void {
  const w = s.renderer.domElement.clientWidth || window.innerWidth;
  const h = s.renderer.domElement.clientHeight || window.innerHeight;
  const aspect = w / h;
  const halfW = aspect * s.viewSize;
  const halfH = s.viewSize;
  const xLimit = Math.max(0, s.worldHalfWidth - halfW);
  const yLimit = Math.max(0, s.worldHalfHeight - halfH);
  if (s.camera.position.x > xLimit) s.camera.position.x = xLimit;
  else if (s.camera.position.x < -xLimit) s.camera.position.x = -xLimit;
  if (s.camera.position.y > yLimit) s.camera.position.y = yLimit;
  else if (s.camera.position.y < -yLimit) s.camera.position.y = -yLimit;
  s.camera.updateMatrixWorld();
}

export function resizeRenderer(s: Scene): void {
  const w = window.innerWidth, h = window.innerHeight;
  s.renderer.setSize(w, h, false);
  updateFrustum(s);
  clampCamera(s);
}

export function setViewSize(s: Scene, viewSize: number): void {
  if (viewSize === s.viewSize) return;
  s.viewSize = viewSize;
  updateFrustum(s);
  clampCamera(s);
}

export function panBy(s: Scene, dx: number, dy: number): void {
  s.camera.position.x += dx;
  s.camera.position.y += dy;
  clampCamera(s);
}

export function render(s: Scene): void {
  s.renderer.render(s.scene, s.camera);
}
