import * as THREE from 'three';
import type { GameState } from '../game/state';
import { MAX_STRENGTH } from '../game/state';

const COLORS = [0x4a90e2, 0xe24a4a, 0x4ae28a, 0xe2c44a];
const NEUTRAL = 0x666666;
const NODE_BASE_RADIUS = 0.45;
const VIEW_SIZE = 8;

export type Scene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  nodeMeshes: THREE.Mesh[];
  nodeMatById: Map<number, THREE.MeshBasicMaterial>;
  flowGroup: THREE.Group;
  selectedHighlight: THREE.Mesh;
  domElement: HTMLCanvasElement;
};

export function createScene(canvas: HTMLCanvasElement, state: GameState): Scene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0a0a0a);

  const scene = new THREE.Scene();
  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.OrthographicCamera(
    -aspect * VIEW_SIZE, aspect * VIEW_SIZE,
    VIEW_SIZE, -VIEW_SIZE,
    0.1, 100,
  );
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  const edgeMat = new THREE.LineBasicMaterial({ color: 0x2a2a2a });
  for (const e of state.edges) {
    const a = state.nodes[e.a].pos, b = state.nodes[e.b].pos;
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(a.x, a.y, 0),
      new THREE.Vector3(b.x, b.y, 0),
    ]);
    scene.add(new THREE.Line(geom, edgeMat));
  }

  const nodeMatById = new Map<number, THREE.MeshBasicMaterial>();
  const nodeMeshes: THREE.Mesh[] = state.nodes.map(n => {
    const geom = new THREE.CircleGeometry(NODE_BASE_RADIUS, 32);
    const mat = new THREE.MeshBasicMaterial({ color: NEUTRAL });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(n.pos.x, n.pos.y, 0.1);
    mesh.userData.nodeId = n.id;
    nodeMatById.set(n.id, mat);
    scene.add(mesh);
    return mesh;
  });

  const highlightGeom = new THREE.RingGeometry(NODE_BASE_RADIUS * 1.4, NODE_BASE_RADIUS * 1.6, 32);
  const highlightMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const selectedHighlight = new THREE.Mesh(highlightGeom, highlightMat);
  selectedHighlight.visible = false;
  selectedHighlight.position.z = 0.05;
  scene.add(selectedHighlight);

  const flowGroup = new THREE.Group();
  scene.add(flowGroup);

  return {
    renderer, scene, camera,
    nodeMeshes, nodeMatById,
    flowGroup, selectedHighlight,
    domElement: canvas,
  };
}

export function updateScene(s: Scene, state: GameState, selected: number | null): void {
  for (const node of state.nodes) {
    const mat = s.nodeMatById.get(node.id)!;
    mat.color.setHex(node.owner === null ? NEUTRAL : COLORS[node.owner % COLORS.length]);
    const scale = 0.55 + (node.strength / MAX_STRENGTH) * 1.2;
    s.nodeMeshes[node.id].scale.setScalar(scale);
  }

  if (selected !== null) {
    const node = state.nodes[selected];
    s.selectedHighlight.position.x = node.pos.x;
    s.selectedHighlight.position.y = node.pos.y;
    const scale = 0.55 + (node.strength / MAX_STRENGTH) * 1.2;
    s.selectedHighlight.scale.setScalar(scale);
    s.selectedHighlight.visible = true;
  } else {
    s.selectedHighlight.visible = false;
  }

  while (s.flowGroup.children.length > 0) {
    const child = s.flowGroup.children[0];
    s.flowGroup.remove(child);
    if ('geometry' in child) (child as THREE.Mesh).geometry.dispose();
  }
  for (const flow of state.flows) {
    const a = state.nodes[flow.src].pos, b = state.nodes[flow.dst].pos;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const offset = 0.12;
    const px = -uy * offset, py = ux * offset;
    const start = new THREE.Vector3(a.x + ux * 0.6 + px, a.y + uy * 0.6 + py, 0.05);
    const end   = new THREE.Vector3(b.x - ux * 0.6 + px, b.y - uy * 0.6 + py, 0.05);
    const tip   = end.clone();
    const headBase = end.clone().sub(new THREE.Vector3(ux, uy, 0).multiplyScalar(0.25));
    const left  = headBase.clone().add(new THREE.Vector3(-uy, ux, 0).multiplyScalar(0.12));
    const right = headBase.clone().add(new THREE.Vector3( uy, -ux, 0).multiplyScalar(0.12));
    const geom = new THREE.BufferGeometry().setFromPoints([start, end, left, tip, right, end]);
    const color = COLORS[flow.player % COLORS.length];
    s.flowGroup.add(new THREE.Line(geom, new THREE.LineBasicMaterial({ color })));
  }
}

export function resizeRenderer(s: Scene): void {
  const w = window.innerWidth, h = window.innerHeight;
  s.renderer.setSize(w, h, false);
  const aspect = w / h;
  s.camera.left = -aspect * VIEW_SIZE;
  s.camera.right = aspect * VIEW_SIZE;
  s.camera.top = VIEW_SIZE;
  s.camera.bottom = -VIEW_SIZE;
  s.camera.updateProjectionMatrix();
}

export function render(s: Scene): void {
  s.renderer.render(s.scene, s.camera);
}
