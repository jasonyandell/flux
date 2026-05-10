import GUI from 'lil-gui';
import { AI_NAMES, AIs, type AIName } from './ai';
import { makeInitialState } from './game/graph';
import type { GameState } from './game/state';
import { applyAction, step } from './game/step';
import { eventToWorld } from './input/pick';
import { createGameUI, fadeHint, getWinner, hideBanner, setPaused, showBanner, showStasisBanner } from './render/gameui';
import { createScene, panBy, render, resizeRenderer, setViewSize, updateScene } from './render/scene';
import { detectStasis } from './sim/stasis';

const PLAYER_COUNT_OPTIONS = [2, 4, 6, 8, 12];
const PLAYER_COLORS_CSS = [
  '#4a90e2', '#e24a4a', '#4ae28a', '#e2c44a',
  '#a44ae2', '#e2884a', '#4ae2e2', '#e24a88',
  '#88e24a', '#e2e24a', '#4a88e2', '#a4e24a',
];

const TICK_HZ = 10;
const TICK_DT = 1 / TICK_HZ;
const SPEED = 5;

const STASIS_SAMPLE_PERIOD_TICKS = 5;
const STASIS_WINDOW = 50;
const STASIS_EPSILON = 1.0;

const tunables = {
  paused: false,
  aiPeriodSec: 0.5,
  numPlayers: 6,
  reset: () => respawn(),
};

let state: GameState = makeInitialState(undefined, undefined, tunables.numPlayers);
let winner: number | null = null;
let stasis = false;
const stasisBuffer: number[][] = [];
let lastStasisSampleTick = 0;
let playerAIs: AIName[] = randomAssignment(tunables.numPlayers);

function randomAssignment(n: number): AIName[] {
  const out: AIName[] = [];
  for (let i = 0; i < n; i++) {
    out.push(AI_NAMES[Math.floor(Math.random() * AI_NAMES.length)]);
  }
  return out;
}

function respawn() {
  state = makeInitialState(undefined, undefined, tunables.numPlayers);
  winner = null;
  stasis = false;
  stasisBuffer.length = 0;
  lastStasisSampleTick = 0;
  playerAIs = randomAssignment(tunables.numPlayers);
  hideBanner(gameUI);
}

function sampleCounts(s: GameState): number[] {
  const counts = new Array(s.numPlayers).fill(0);
  for (const n of s.nodes) if (n.owner !== null) counts[n.owner]++;
  return counts;
}

const canvas = document.getElementById('app') as HTMLCanvasElement;
const scene = createScene(canvas, state);
const gameUI = createGameUI();
gameUI.onPlayAgain(() => respawn());
fadeHint(gameUI);

window.addEventListener('resize', () => resizeRenderer(scene));
resizeRenderer(scene);

const ZOOM_STEP = 1.1;
const MIN_VIEW = 1.5;
const MAX_VIEW = 36;

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
  const next = Math.max(MIN_VIEW, Math.min(MAX_VIEW, scene.viewSize / factor));
  if (next === scene.viewSize) return;
  const worldBefore = eventToWorld(scene, ev);
  setViewSize(scene, next);
  const worldAfter = eventToWorld(scene, ev);
  panBy(scene, worldBefore.x - worldAfter.x, worldBefore.y - worldAfter.y);
}, { passive: false });

const gui = new GUI({ title: 'flux — robot wars' });
gui.add(tunables, 'numPlayers', PLAYER_COUNT_OPTIONS).name('players').onChange(() => respawn());
const pausedCtrl = gui.add(tunables, 'paused');
gui.add(tunables, 'aiPeriodSec', 0.05, 2, 0.05).name('ai period (s)');
gui.add(tunables, 'reset').name('respawn (reshuffle)');
gameUI.onPauseClick(() => { tunables.paused = false; pausedCtrl.updateDisplay(); });

let last = performance.now();
let stepAcc = 0;
let aiAcc = 0;

function frame(now: number) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;

  if (!tunables.paused && winner === null && !stasis) {
    const scaled = dt * SPEED;
    stepAcc += scaled;
    aiAcc += scaled;
    while (stepAcc >= TICK_DT) {
      state = step(state, TICK_DT);
      stepAcc -= TICK_DT;
      if (state.tick - lastStasisSampleTick >= STASIS_SAMPLE_PERIOD_TICKS) {
        lastStasisSampleTick = state.tick;
        stasisBuffer.push(sampleCounts(state));
        if (stasisBuffer.length > STASIS_WINDOW) stasisBuffer.shift();
        if (detectStasis(stasisBuffer, STASIS_EPSILON, STASIS_WINDOW)) {
          stasis = true;
          showStasisBanner(gameUI);
        }
      }
    }
    if (aiAcc >= tunables.aiPeriodSec) {
      for (let p = 0; p < state.numPlayers; p++) {
        const fn = AIs[playerAIs[p]];
        for (const a of fn(state, p, state.tick)) state = applyAction(state, a);
      }
      aiAcc = 0;
    }
    const w = getWinner(state);
    if (w !== null) {
      winner = w;
      showBanner(gameUI, w, playerAIs[w]);
    }
  }

  setPaused(gameUI, tunables.paused && winner === null && !stasis);
  updateScene(scene, state, null);
  updateHud(state);
  render(scene);
  requestAnimationFrame(frame);
}

function updateHud(s: GameState) {
  const hud = document.getElementById('hud');
  if (!hud) return;
  const counts = new Array(s.numPlayers).fill(0);
  let neutral = 0;
  for (const n of s.nodes) {
    if (n.owner === null) neutral++;
    else counts[n.owner]++;
  }
  let html = `<div>tick ${s.tick}   neutral: ${neutral}</div>`;
  for (let p = 0; p < s.numPlayers; p++) {
    const color = PLAYER_COLORS_CSS[p % PLAYER_COLORS_CSS.length];
    html += `<div><span style="display:inline-block;width:10px;height:10px;background:${color};margin-right:6px;vertical-align:middle"></span>${playerAIs[p]}: ${counts[p]}</div>`;
  }
  hud.innerHTML = html;
}

requestAnimationFrame(frame);
