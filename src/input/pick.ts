import * as THREE from 'three';
import type { Scene } from '../render/scene';
import { NODE_PICK_RADIUS } from '../render/scene';

const worldVec = new THREE.Vector3();

export function pickNode(scene: Scene, ev: PointerEvent): number | null {
  const w = eventToWorld(scene, ev);
  const positions = scene.nodePositions;
  const r2 = NODE_PICK_RADIUS * NODE_PICK_RADIUS;
  let best = -1, bestD2 = Infinity;
  for (let i = 0; i < scene.nodeCount; i++) {
    const dx = positions[i * 2] - w.x;
    const dy = positions[i * 2 + 1] - w.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2 && d2 <= r2) { bestD2 = d2; best = i; }
  }
  return best === -1 ? null : best;
}

export function eventToWorld(scene: Scene, ev: PointerEvent): { x: number; y: number } {
  const rect = scene.domElement.getBoundingClientRect();
  worldVec.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  worldVec.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  worldVec.z = 0;
  worldVec.unproject(scene.camera);
  return { x: worldVec.x, y: worldVec.y };
}
