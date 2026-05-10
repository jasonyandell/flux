import * as THREE from 'three';
import type { GameState } from '../game/state';
import { MAX_STRENGTH } from '../game/state';

const COLORS = [0x4a90e2, 0xe24a4a, 0x4ae28a, 0xe2c44a];
const NEUTRAL = new THREE.Color(0x666666);
const NODE_BASE_RADIUS = 0.45;
const VIEW_PADDING = 1.05;

export type Scene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  nodeInstanced: THREE.InstancedMesh;
  selectedHighlight: THREE.Mesh;
  flowLines: THREE.LineSegments;
  dragLine: THREE.Line;
  domElement: HTMLCanvasElement;
  viewSize: number;
  nodeCount: number;
  nodePositions: Float32Array;
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

  let maxAbs = 1;
  for (const n of state.nodes) {
    if (Math.abs(n.pos.x) > maxAbs) maxAbs = Math.abs(n.pos.x);
    if (Math.abs(n.pos.y) > maxAbs) maxAbs = Math.abs(n.pos.y);
  }
  const viewSize = maxAbs * VIEW_PADDING;

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
  scene.add(new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: 0x1a1a1a })));

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
    nodeInstanced, selectedHighlight,
    flowLines, dragLine,
    domElement: canvas,
    viewSize,
    nodeCount: state.nodes.length,
    nodePositions,
  };
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

  const nFlows = state.flows.length;
  const positions = new Float32Array(nFlows * 6);
  const colors = new Float32Array(nFlows * 6);
  for (let i = 0; i < nFlows; i++) {
    const flow = state.flows[i];
    const a = state.nodes[flow.src].pos, b = state.nodes[flow.dst].pos;
    const mx = a.x + (b.x - a.x) * 0.5;
    const my = a.y + (b.y - a.y) * 0.5;
    positions[i * 6 + 0] = a.x; positions[i * 6 + 1] = a.y; positions[i * 6 + 2] = 0.05;
    positions[i * 6 + 3] = mx;  positions[i * 6 + 4] = my;  positions[i * 6 + 5] = 0.05;
    tmpColor.copy(ownerColors[flow.player % ownerColors.length]);
    for (let k = 0; k < 2; k++) {
      colors[i * 6 + k * 3 + 0] = tmpColor.r;
      colors[i * 6 + k * 3 + 1] = tmpColor.g;
      colors[i * 6 + k * 3 + 2] = tmpColor.b;
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
