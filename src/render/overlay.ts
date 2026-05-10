import * as THREE from 'three';
import type { GameState } from '../game/state';
import type { Scene } from './scene';

const NODE_OFFSET_PX = 26;
const DELTA_SMOOTH = 0.25;

export type Overlay = {
  root: HTMLDivElement;
  totals: HTMLSpanElement[];
  deltas: HTMLSpanElement[];
  smoothed: number[];
  prev: number[];
};

export function createOverlay(state: GameState): Overlay {
  const root = document.createElement('div');
  root.style.cssText =
    'position:fixed;inset:0;pointer-events:none;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#fff;opacity:0.7;';
  document.body.appendChild(root);

  const spanStyle = 'position:absolute;transform:translate(-50%,0);white-space:nowrap;line-height:1.1;';
  const totals: HTMLSpanElement[] = state.nodes.map(() => {
    const t = document.createElement('span');
    t.style.cssText = spanStyle;
    root.appendChild(t);
    return t;
  });
  const deltas: HTMLSpanElement[] = state.nodes.map(() => {
    const d = document.createElement('span');
    d.style.cssText = spanStyle;
    root.appendChild(d);
    return d;
  });

  return {
    root,
    totals,
    deltas,
    smoothed: state.nodes.map(() => 0),
    prev: state.nodes.map(n => n.strength),
  };
}

const v = new THREE.Vector3();

export function updateOverlay(o: Overlay, s: Scene, state: GameState, dt: number): void {
  const w = s.renderer.domElement.clientWidth;
  const h = s.renderer.domElement.clientHeight;

  for (const node of state.nodes) {
    const raw = node.strength - o.prev[node.id];
    const perSec = dt > 0 ? raw / dt : 0;
    o.smoothed[node.id] = o.smoothed[node.id] * (1 - DELTA_SMOOTH) + perSec * DELTA_SMOOTH;
    o.prev[node.id] = node.strength;

    v.set(node.pos.x, node.pos.y, 0).project(s.camera);
    const px = (v.x * 0.5 + 0.5) * w;
    const py = (1 - (v.y * 0.5 + 0.5)) * h;

    const total = o.totals[node.id];
    total.textContent = String(Math.round(node.strength));
    total.style.left = `${px}px`;
    total.style.top = `${py + NODE_OFFSET_PX}px`;

    const d = o.smoothed[node.id];
    const delta = o.deltas[node.id];
    if (Math.abs(d) < 0.05) {
      delta.textContent = '0';
    } else {
      const arrow = d > 0 ? '↑' : '↓';
      const sign = d > 0 ? '+' : '';
      delta.textContent = `${sign}${d.toFixed(1)} ${arrow}`;
    }
    delta.style.left = `${px}px`;
    delta.style.top = `${py + NODE_OFFSET_PX + 14}px`;
  }
}
