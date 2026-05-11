import GUI from 'lil-gui';
import { AI_NAMES, AIs, type AIName } from './ai';
import { makeInitialState } from './game/graph';
import type { GameState } from './game/state';
import { applyAction, step } from './game/step';
import { eventToWorld } from './input/pick';
import { createGameUI, createTopBar, fadeHint, getWinner, hideBanner, setPaused, showBanner, showStasisBanner } from './render/gameui';
import { createScene, panBy, render, resizeRenderer, setViewSize, updateScene } from './render/scene';
import { detectStasis } from './sim/stasis';
import { initGPU } from './gpu/runtime';
import { DEFAULT_EVOLUTION_CONFIG, makeInitialEvolutionState, runGeneration, saveEvolutionState, loadEvolutionState, clearEvolutionState, saveEvolveEnabled, loadEvolveEnabled, type EvolutionState } from './gpu/evolution';
import { runParityTest } from './gpu/parity';
import { createOverlay } from './demo/overlay';
import { createRunner } from './demo/runner';

const PLAYER_COUNT_OPTIONS = [2, 4, 6, 8, 12];
const DEMO_MODE = new URLSearchParams(window.location.search).get('demo') === '1';
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
  numPlayers: 12,
  reset: () => respawn(),
  evolve: false,
  generation: 0,
  bestFitness: 0,
  allTimeBest: 0,
  parityRun: () => runParity(),
  parityResult: '(not run)',
};

let state: GameState = makeInitialState(undefined, undefined, tunables.numPlayers);
let winner: number | null = null;
let stasis = false;
const stasisBuffer: number[][] = [];
let lastStasisSampleTick = 0;
let playerAIs: AIName[] = defaultAssignment(tunables.numPlayers);

function defaultAssignment(n: number): AIName[] {
  const out: AIName[] = Array(n).fill('evolved' as AIName);
  out[0] = 'aggressive';
  return out;
}

function respawn() {
  state = makeInitialState(undefined, undefined, tunables.numPlayers);
  winner = null;
  stasis = false;
  stasisBuffer.length = 0;
  lastStasisSampleTick = 0;
  playerAIs = defaultAssignment(tunables.numPlayers);
  hideBanner(gameUI);
  rebuildSeatControls();
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

const topBar = createTopBar();
topBar.setEvolveAvailable(false, 'checking GPU…');
topBar.onRestart(() => respawn());
topBar.onEvolveToggle((next) => {
  tunables.evolve = next;
  topBar.setEvolveOn(next);
  saveEvolveEnabled(next);
  if (next) startEvolution();
});

window.addEventListener('resize', () => resizeRenderer(scene));
resizeRenderer(scene);

const ZOOM_STEP = 1.1;
const MIN_VIEW = 1.5;
const MAX_VIEW = 36;

function zoomAndPanAt(
  before: { clientX: number; clientY: number },
  after: { clientX: number; clientY: number },
  factor: number,
): void {
  const worldBefore = eventToWorld(scene, before);
  if (factor !== 1) {
    const next = Math.max(MIN_VIEW, Math.min(MAX_VIEW, scene.viewSize / factor));
    if (next !== scene.viewSize) setViewSize(scene, next);
  }
  const worldAfter = eventToWorld(scene, after);
  panBy(scene, worldBefore.x - worldAfter.x, worldBefore.y - worldAfter.y);
}

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
  zoomAndPanAt(ev, ev, factor);
}, { passive: false });

type Pointer = { id: number; x: number; y: number };
const pointers: Pointer[] = [];

function centroidOf(pts: Pointer[]): { clientX: number; clientY: number } {
  let sx = 0, sy = 0;
  for (const p of pts) { sx += p.x; sy += p.y; }
  return { clientX: sx / pts.length, clientY: sy / pts.length };
}

function spreadOf(pts: Pointer[], c: { clientX: number; clientY: number }): number {
  if (pts.length < 2) return 0;
  let sum = 0;
  for (const p of pts) sum += Math.hypot(p.x - c.clientX, p.y - c.clientY);
  return sum / pts.length;
}

canvas.addEventListener('pointerdown', (ev) => {
  if (ev.pointerType === 'mouse' && ev.button !== 0) return;
  canvas.setPointerCapture(ev.pointerId);
  pointers.push({ id: ev.pointerId, x: ev.clientX, y: ev.clientY });
});

canvas.addEventListener('pointermove', (ev) => {
  const p = pointers.find(p => p.id === ev.pointerId);
  if (!p) return;
  const beforeCentroid = centroidOf(pointers);
  const beforeSpread = spreadOf(pointers, beforeCentroid);
  p.x = ev.clientX;
  p.y = ev.clientY;
  const afterCentroid = centroidOf(pointers);
  const afterSpread = spreadOf(pointers, afterCentroid);
  const factor = beforeSpread > 0 && afterSpread > 0 ? afterSpread / beforeSpread : 1;
  zoomAndPanAt(beforeCentroid, afterCentroid, factor);
});

function releasePointer(ev: PointerEvent) {
  const i = pointers.findIndex(p => p.id === ev.pointerId);
  if (i !== -1) pointers.splice(i, 1);
  if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
}
canvas.addEventListener('pointerup', releasePointer);
canvas.addEventListener('pointercancel', releasePointer);

const gui = new GUI({ title: 'flux — robot wars' });
gui.close();
gui.add(tunables, 'numPlayers', PLAYER_COUNT_OPTIONS).name('players').onChange(() => {
  respawn();
  rebuildSeatControls();
});
const pausedCtrl = gui.add(tunables, 'paused');
gui.add(tunables, 'aiPeriodSec', 0.05, 2, 0.05).name('ai period (s)');
gui.add(tunables, 'reset').name('respawn (reshuffle)');
gameUI.onPauseClick(() => { tunables.paused = false; pausedCtrl.updateDisplay(); });

const seatFolder = gui.addFolder('seats');
const seatProxy: Record<string, AIName> = {};
let seatControls: { destroy: () => void }[] = [];
function rebuildSeatControls() {
  for (const c of seatControls) c.destroy();
  seatControls = [];
  for (let p = 0; p < tunables.numPlayers; p++) {
    const key = `seat ${p}`;
    seatProxy[key] = playerAIs[p];
    const c = seatFolder.add(seatProxy, key, AI_NAMES).onChange((v: AIName) => {
      playerAIs[p] = v;
    });
    seatControls.push(c);
  }
}
rebuildSeatControls();

const evoFolder = gui.addFolder('evolution');
evoFolder.add(tunables, 'evolve').name('evolve (run in bg)').listen().onChange((v: boolean) => {
  saveEvolveEnabled(v);
  topBar.setEvolveOn(v);
  if (v) startEvolution();
});
const genCtrl = evoFolder.add(tunables, 'generation').name('generation').listen().disable();
const fitCtrl = evoFolder.add(tunables, 'bestFitness').name('best fitness').listen().disable();
const allTimeCtrl = evoFolder.add(tunables, 'allTimeBest').name('all-time best').listen().disable();
void allTimeCtrl;
evoFolder.add({ save: saveChampionFile }, 'save').name('save champion');
evoFolder.add({ load: loadChampionFile }, 'load').name('load champion');
evoFolder.add({ clear: clearSavedEvolution }, 'clear').name('clear save (fresh start)');
evoFolder.add(tunables, 'parityRun').name('run parity test');
const parityCtrl = evoFolder.add(tunables, 'parityResult').name('parity').listen().disable();
void genCtrl; void fitCtrl; void parityCtrl;

const captureFolder = gui.addFolder('capture');
const captureState = { recording: false };
captureFolder.add({ snap: snapshotPng }, 'snap').name('snapshot (png)');
captureFolder.add(captureState, 'recording').name('record (webm)').listen().onChange((v: boolean) => {
  if (v) startRecording(); else stopRecording();
});

let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

function snapshotPng() {
  scene.renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-${timestamp()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

function startRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') return;
  const stream = scene.renderer.domElement.captureStream(60);
  recordedChunks = [];
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find(t => MediaRecorder.isTypeSupported(t)) ?? 'video/webm';
  mediaRecorder = new MediaRecorder(stream, { mimeType });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-recording-${timestamp()}.webm`;
    a.click();
    URL.revokeObjectURL(url);
    mediaRecorder = null;
  };
  mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
}

function saveChampionFile() {
  const g = getChampion();
  if (!g) { console.warn('no champion yet — evolve a generation first'); return; }
  const payload = {
    weights: Array.from(g),
    generation: tunables.generation,
    bestFitness: tunables.bestFitness,
    savedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `flux-champion-gen${tunables.generation}-fit${tunables.bestFitness}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function loadChampionFile() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json,.json';
  input.onchange = async () => {
    const f = input.files?.[0];
    if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      const weights: number[] = Array.isArray(data) ? data : data.weights;
      if (!Array.isArray(weights)) throw new Error('no weights array');
      setChampion(new Float32Array(weights));
      console.log(`loaded champion: ${weights.length} weights, gen=${data.generation ?? '?'}, fit=${data.bestFitness ?? '?'}`);
    } catch (err) {
      console.error('load failed:', err);
    }
  };
  input.click();
}

let gpuCtx: Awaited<ReturnType<typeof initGPU>> = null;
let evoState: EvolutionState | null = null;
let evoRunning = false;

(async () => {
  gpuCtx = await initGPU();
  if (!gpuCtx) {
    console.warn('WebGPU unavailable — evolution disabled. The "evolved" AI will use a random genome.');
    tunables.parityResult = 'WebGPU unavailable';
    topBar.setEvolveAvailable(false, 'no WebGPU');
    return;
  }
  console.log('WebGPU initialized');
  topBar.setEvolveAvailable(true);

  const saved = loadEvolutionState();
  if (saved) {
    evoState = saved;
    setChampion(saved.champion);
    setRoster(saved.population);
    tunables.generation = saved.generation;
    tunables.bestFitness = Math.round(saved.bestFitness * 100) / 100;
    tunables.allTimeBest = Math.round(saved.allTimeBest * 100) / 100;
    console.log(`resumed evolution: gen=${saved.generation}, best=${saved.bestFitness.toFixed(2)}, pop=${saved.population.length}`);
  }
  if (loadEvolveEnabled()) {
    tunables.evolve = true;
    topBar.setEvolveOn(true);
    startEvolution();
  }
})();

function clearSavedEvolution() {
  clearEvolutionState();
  evoState = null;
  evoRunning = false;
  tunables.evolve = false;
  tunables.generation = 0;
  tunables.bestFitness = 0;
  tunables.allTimeBest = 0;
  setChampion(null);
  setRoster(null);
  topBar.setEvolveOn(false);
  topBar.setStats(0, 0);
  console.log('evolution save cleared');
}

function startEvolution() {
  if (!gpuCtx || evoRunning) return;
  if (!evoState) evoState = makeInitialEvolutionState(DEFAULT_EVOLUTION_CONFIG);
  evoRunning = true;
  loopEvolution();
}

async function loopEvolution() {
  while (evoRunning && tunables.evolve && gpuCtx && evoState) {
    try {
      evoState = await runGeneration(gpuCtx, DEFAULT_EVOLUTION_CONFIG, evoState, Math.random);
      tunables.generation = evoState.generation;
      tunables.bestFitness = Math.round(evoState.bestFitness * 100) / 100;
      tunables.allTimeBest = Math.round(evoState.allTimeBest * 100) / 100;
      setChampion(evoState.champion);
      setRoster(evoState.population);
      saveEvolutionState(evoState);
    } catch (err) {
      console.error('evolution error:', err);
      tunables.evolve = false;
      break;
    }
    await new Promise(r => setTimeout(r, 0));
  }
  evoRunning = false;
}

async function runParity() {
  tunables.parityResult = 'running...';
  if (!gpuCtx) { tunables.parityResult = 'WebGPU unavailable'; return; }
  try {
    const r = await runParityTest(gpuCtx, { ticks: 50, radius: 6 });
    tunables.parityResult = r.ok
      ? `OK (maxΔ=${r.maxStrengthDiff.toExponential(2)})`
      : `FAIL maxΔ=${r.maxStrengthDiff.toExponential(2)} ownerMis=${r.ownerMismatches} flowMis=${r.flowMismatches}`;
    console.log('parity:', r);
  } catch (err) {
    console.error('parity error:', err);
    tunables.parityResult = 'error (see console)';
  }
}

// Expose a few hooks for headless testing / debugging.
import { getChampion, setChampion, setRoster } from './gpu/evolved';
(window as unknown as { fluxTest?: object }).fluxTest = {
  runParity: async () => {
    if (!gpuCtx) {
      await new Promise(r => setTimeout(r, 200));
    }
    if (!gpuCtx) return { error: 'no gpu' };
    return await runParityTest(gpuCtx, { ticks: 50, radius: 6 });
  },
  evolveOneGeneration: async () => {
    if (!gpuCtx) return { error: 'no gpu' };
    if (!evoState) evoState = makeInitialEvolutionState(DEFAULT_EVOLUTION_CONFIG);
    evoState = await runGeneration(gpuCtx, DEFAULT_EVOLUTION_CONFIG, evoState, Math.random);
    return { generation: evoState.generation, bestFitness: evoState.bestFitness };
  },
  getChampion: () => {
    const g = getChampion();
    return g ? Array.from(g) : null;
  },
  setChampion: (arr: number[]) => {
    setChampion(new Float32Array(arr));
  },
  hasGPU: () => !!gpuCtx,
};

let last = performance.now();
let stepAcc = 0;
let aiAcc = 0;
let topBarGen = -1;
let topBarBest = -1;
let topBarEvolveOn = false;

function frame(now: number) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;

  if (runner?.isActive()) {
    runner.tick(dt);
    const snap = runner.currentSnapshot();
    if (snap) updateScene(scene, snap, null);
    render(scene);
    requestAnimationFrame(frame);
    return;
  }

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
  if (tunables.generation !== topBarGen || tunables.bestFitness !== topBarBest) {
    topBarGen = tunables.generation;
    topBarBest = tunables.bestFitness;
    topBar.setStats(topBarGen, topBarBest);
  }
  if (tunables.evolve !== topBarEvolveOn) {
    topBarEvolveOn = tunables.evolve;
    topBar.setEvolveOn(topBarEvolveOn);
  }
  render(scene);
  requestAnimationFrame(frame);
}

const hudEl = document.getElementById('hud') as HTMLDivElement;
let hudExpanded = false;
hudEl.onclick = () => { hudExpanded = !hudExpanded; };

function updateHud(s: GameState) {
  if (!hudEl) return;
  const counts = new Array(s.numPlayers).fill(0);
  let neutral = 0, alive = 0;
  for (const n of s.nodes) {
    if (n.owner === null) neutral++;
    else counts[n.owner]++;
  }
  for (const c of counts) if (c > 0) alive++;
  const chevron = hudExpanded ? '▾' : '▸';
  let html = `<div class="summary">${chevron} tick ${s.tick} · ${alive}/${s.numPlayers} alive</div>`;
  if (hudExpanded) {
    html += '<div class="details">';
    for (let p = 0; p < s.numPlayers; p++) {
      const color = PLAYER_COLORS_CSS[p % PLAYER_COLORS_CSS.length];
      const dead = counts[p] === 0;
      html += `<div class="row${dead ? ' dead' : ''}"><span class="sw" style="background:${color}"></span><span>${playerAIs[p]}</span><span style="margin-left:auto;padding-left:10px">${counts[p]}</span></div>`;
    }
    html += `<div class="row dead"><span class="sw" style="background:#666"></span><span>neutral</span><span style="margin-left:auto;padding-left:10px">${neutral}</span></div>`;
    html += '</div>';
  }
  hudEl.innerHTML = html;
}

// Demo mode wiring: hide chrome, run scripted scenes when `?demo=1`.
const overlay = DEMO_MODE ? createOverlay() : null;
const runner = (DEMO_MODE && overlay)
  ? createRunner({ scene, overlay })
  : null;

if (DEMO_MODE) {
  gui.hide?.();
  const topBarEl = document.getElementById('flux-topbar');
  if (topBarEl) topBarEl.style.display = 'none';
  if (hudEl) hudEl.style.display = 'none';
  if (gameUI.hint) gameUI.hint.style.display = 'none';
  const installBanner = document.getElementById('install-banner');
  if (installBanner) installBanner.style.display = 'none';
  if (runner) void runner.enter();
}

requestAnimationFrame(frame);
