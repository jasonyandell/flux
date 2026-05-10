import * as THREE from 'three';
import type { Scene } from '../render/scene';

const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();

export function pickNode(scene: Scene, ev: PointerEvent): number | null {
  const rect = scene.domElement.getBoundingClientRect();
  ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  ray.setFromCamera(ndc, scene.camera);
  const hits = ray.intersectObjects(scene.nodeMeshes, false);
  if (hits.length === 0) return null;
  return hits[0].object.userData.nodeId as number;
}
