import GUI from 'lil-gui';
import { aiThink } from './ai/dumb';
import { makeInitialState } from './game/graph';
import type { GameState } from './game/state';
import { applyAction, step } from './game/step';
import { eventToWorld, pickNode } from './input/pick';
import { createGameUI, fadeHint, getWinner, hideBanner, setPaused, showBanner } from './render/gameui';
import { createOverlay, updateOverlay } from './render/overlay';
import { createScene, render, resizeRenderer, setDragLine, updateScene } from './render/scene';

const HUMAN = 0;
const AI = 1;

const TICK_HZ = 10;
const TICK_DT = 1 / TICK_HZ;

let state: GameState = makeInitialState();
let selected: number | null = null;

let winner: number | null = null;
let hintFaded = false;

const canvas = document.getElementById('app') as HTMLCanvasElement;
const scene = createScene(canvas, state);
const overlay = createOverlay(state);
const gameUI = createGameUI();
gameUI.onPlayAgain(() => {
  state = makeInitialState();
  selected = null;
  winner = null;
  hideBanner(gameUI);
});

window.addEventListener('resize', () => resizeRenderer(scene));
resizeRenderer(scene);

const DRAG_THRESHOLD_PX = 8;
let dragStartClient: { x: number; y: number } | null = null;
let dragActive = false;

canvas.addEventListener('pointerdown', (ev) => {
  if (winner !== null) return;
  const nodeId = pickNode(scene, ev);
  if (nodeId === null) { selected = null; setDragLine(scene, null, null); return; }
  if (state.nodes[nodeId].owner !== HUMAN) { selected = null; setDragLine(scene, null, null); return; }
  selected = nodeId;
  dragStartClient = { x: ev.clientX, y: ev.clientY };
  dragActive = false;
  canvas.setPointerCapture(ev.pointerId);
});

canvas.addEventListener('pointermove', (ev) => {
  if (selected === null || dragStartClient === null) return;
  if (!dragActive) {
    const dx = ev.clientX - dragStartClient.x, dy = ev.clientY - dragStartClient.y;
    if (dx * dx + dy * dy < DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) return;
    dragActive = true;
  }
  setDragLine(scene, state.nodes[selected].pos, eventToWorld(scene, ev));
});

canvas.addEventListener('pointerup', (ev) => {
  if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
  if (selected === null) return;
  const src = selected;
  const wasDrag = dragActive;
  selected = null;
  dragStartClient = null;
  dragActive = false;
  setDragLine(scene, null, null);
  if (!wasDrag) return;
  const dst = pickNode(scene, ev);
  if (dst === null || dst === src) return;
  const before = state.flows;
  state = applyAction(state, { kind: 'toggleFlow', src, dst, player: HUMAN });
  if (state.flows === before) return;
  if (!hintFaded) {
    hintFaded = true;
    fadeHint(gameUI);
  }
});

canvas.addEventListener('pointercancel', (ev) => {
  if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
  selected = null;
  dragStartClient = null;
  dragActive = false;
  setDragLine(scene, null, null);
});

const tunables = {
  paused: false,
  aiPeriodSec: 0.5,
  reset: () => { state = makeInitialState(); selected = null; winner = null; hideBanner(gameUI); },
};
const gui = new GUI({ title: 'flux' });
const pausedCtrl = gui.add(tunables, 'paused');
gui.add(tunables, 'aiPeriodSec', 0.05, 2, 0.05);
gui.add(tunables, 'reset').name('reset board');
gameUI.onPauseClick(() => { tunables.paused = false; pausedCtrl.updateDisplay(); });

let last = performance.now();
let stepAcc = 0;
let aiAcc = 0;

function frame(now: number) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;

  if (!tunables.paused && winner === null) {
    stepAcc += dt;
    aiAcc += dt;
    while (stepAcc >= TICK_DT) {
      state = step(state, TICK_DT);
      stepAcc -= TICK_DT;
    }
    if (aiAcc >= tunables.aiPeriodSec) {
      for (const a of aiThink(state, AI)) state = applyAction(state, a);
      aiAcc = 0;
    }
    const w = getWinner(state);
    if (w !== null) {
      winner = w;
      showBanner(gameUI, w);
    }
  }

  setPaused(gameUI, tunables.paused && winner === null);
  updateScene(scene, state, selected);
  updateOverlay(overlay, scene, state, dt);
  updateHud(state);
  render(scene);
  requestAnimationFrame(frame);
}

function updateHud(s: GameState) {
  const hud = document.getElementById('hud');
  if (!hud) return;
  let p0 = 0, p1 = 0, neutral = 0;
  for (const n of s.nodes) {
    if (n.owner === null) neutral++;
    else if (n.owner === 0) p0++;
    else if (n.owner === 1) p1++;
  }
  hud.textContent = `tick ${s.tick}   you ${p0}   ai ${p1}   neutral ${neutral}`;
}

requestAnimationFrame(frame);
