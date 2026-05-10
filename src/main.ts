import GUI from 'lil-gui';
import { aiThink } from './ai/dumb';
import { makeInitialState } from './game/graph';
import type { GameState } from './game/state';
import { applyAction, step } from './game/step';
import { pickNode } from './input/pick';
import { createGameUI, fadeHint, getWinner, hideBanner, setPaused, showBanner } from './render/gameui';
import { createOverlay, updateOverlay } from './render/overlay';
import { createScene, render, resizeRenderer, updateScene } from './render/scene';

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

canvas.addEventListener('pointerdown', (ev) => {
  if (winner !== null) return;
  const nodeId = pickNode(scene, ev);
  if (nodeId === null) { selected = null; return; }
  if (selected === null) {
    if (state.nodes[nodeId].owner === HUMAN) selected = nodeId;
    return;
  }
  if (selected === nodeId) { selected = null; return; }
  const before = state.flows;
  state = applyAction(state, { kind: 'toggleFlow', src: selected, dst: nodeId, player: HUMAN });
  if (!hintFaded && state.flows !== before) {
    hintFaded = true;
    fadeHint(gameUI);
  }
  selected = null;
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
