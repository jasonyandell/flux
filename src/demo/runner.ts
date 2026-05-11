import type { GameState } from '../game/state';
import { applyAction, step } from '../game/step';
import { type SceneLabel, loadSceneChampion } from './champions';
import { setChampion } from '../gpu/evolved';
import { AIs, type AIName } from '../ai';
import { makeInitialState } from '../game/graph';
import type { Scene as RenderScene } from '../render/scene';
import { setViewSize, panBy, clampCamera } from '../render/scene';
import type { Overlay } from './overlay';

type Phase =
  | 'intro-pan'
  | 'intro-title'
  | 'intro-zoom-out'
  | 'scene-caption-in'
  | 'scene-hold'
  | 'scene-caption-out';

export type SceneSpec = {
  label: SceneLabel;
  caption: string;
  durationSec: number;
};

export const SCENES: SceneSpec[] = [
  { label: 'gen0',    caption: 'watch ai battle',            durationSec: 5 },
  { label: 'gen100',  caption: 'the blue one is code',       durationSec: 5 },
  { label: 'gen200',  caption: 'the others are neural nets', durationSec: 5 },
  { label: 'gen1000', caption: 'watch them get smarter',     durationSec: 5 },
  { label: 'gen20k',  caption: 'watch them win',             durationSec: 5 },
];

const TICK_HZ = 10;
const TICK_DT = 1 / TICK_HZ;
const AI_PERIOD_SEC = 0.5;
const AI_PERIOD_TICKS = Math.max(1, Math.round(AI_PERIOD_SEC / TICK_DT));
const PRESIM_TICK_BUDGET = 5000;  // full game; presimGame breaks early on winner
const PRESIM_BATCH = 50;
const SNAPSHOT_STRIDE = 5;        // store every 5th tick; ~1000 frames per scene cap
const NUM_PLAYERS = 12;
const INTRO_TICKS = 150;

const INTRO_PAN_SEC = 1.0;
const INTRO_TITLE_SEC = 1.5;
const INTRO_ZOOM_OUT_SEC = 0.7;
const CAPTION_FADE_SEC = 0.6;
const HOT_VIEW_SIZE = 8;

type ScenePresim = {
  spec: SceneSpec;
  ready: Promise<void>;
  snapshots: GameState[];
  expectedLength: number;
};

export type DemoRunner = {
  isActive: () => boolean;
  enter: () => Promise<void>;
  tick: (dt: number) => void;
  currentSnapshot: () => GameState | null;
  currentScene: () => SceneSpec | null;
};

export function createRunner(opts: {
  scene: RenderScene;
  overlay: Overlay;
}): DemoRunner {
  const { scene, overlay } = opts;
  const seats: AIName[] = defaultSeats(NUM_PLAYERS);

  let active = false;
  let phase: Phase = 'intro-pan';
  let phaseElapsed = 0;
  let phaseEntered = false;
  let titleHidden = false;
  let sceneIdx = 0;
  let sceneElapsed = 0;

  let initialViewSize = scene.viewSize;
  let hotTarget = { x: 0, y: 0 };
  let panStart = { x: 0, y: 0 };
  let zoomStart = scene.viewSize;

  let introSnapshots: GameState[] = [];
  let introLastSnap: GameState | null = null;
  const presims: ScenePresim[] = SCENES.map((spec) => ({
    spec,
    snapshots: [],
    ready: Promise.resolve(),
    expectedLength: PRESIM_TICK_BUDGET + 1,
  }));

  let currentFrame: GameState | null = null;

  async function enter(): Promise<void> {
    active = true;
    initialViewSize = scene.viewSize;

    // Pre-sim the intro game (gen0 champion = fresh random) for the hot-area
    // hint. Champion is module-global; we set it before each pre-sim and
    // chain them sequentially so they don't thrash.
    setChampion(null);
    introSnapshots = [];
    await presimGame(
      makeInitialState(undefined, undefined, NUM_PLAYERS),
      seats,
      INTRO_TICKS,
      introSnapshots,
    );
    introLastSnap = introSnapshots[introSnapshots.length - 1] ?? null;
    hotTarget = introLastSnap ? pickHotArea(introLastSnap) : { x: 0, y: 0 };

    // Kick off all five scene pre-sims. Sequential so they share the global
    // champion serially; later scenes' snapshots race against playback wall-clock.
    presims[0].ready = presimScene(presims[0]);
    let chain = presims[0].ready;
    for (let i = 1; i < presims.length; i++) {
      const j = i;
      presims[j].ready = chain.then(() => presimScene(presims[j]));
      chain = presims[j].ready;
    }
    await presims[0].ready;

    panStart = { x: scene.camera.position.x, y: scene.camera.position.y };
    zoomStart = scene.viewSize;
    currentFrame = introLastSnap;
    phase = 'intro-pan';
    phaseElapsed = 0;
    phaseEntered = false;
    titleHidden = false;
    sceneElapsed = 0;
    overlay.hideTitle();
    overlay.hideCaption();
  }

  function tick(dt: number): void {
    if (!active) return;
    phaseElapsed += dt;

    if (phase === 'intro-pan') {
      currentFrame = introLastSnap;
      const t = Math.min(1, phaseElapsed / INTRO_PAN_SEC);
      const e = easeInOut(t);
      const targetSize = lerp(zoomStart, HOT_VIEW_SIZE, e);
      if (Math.abs(targetSize - scene.viewSize) > 1e-3) setViewSize(scene, targetSize);
      const cam = scene.camera.position;
      const wantX = lerp(panStart.x, hotTarget.x, e);
      const wantY = lerp(panStart.y, hotTarget.y, e);
      panBy(scene, wantX - cam.x, wantY - cam.y);
      clampCamera(scene);
      if (t >= 1) advance('intro-title');
      return;
    }

    if (phase === 'intro-title') {
      currentFrame = introLastSnap;
      if (!phaseEntered) {
        overlay.showTitle('AI WARS');
        titleHidden = false;
        phaseEntered = true;
      }
      if (!titleHidden && phaseElapsed >= INTRO_TITLE_SEC - CAPTION_FADE_SEC) {
        overlay.hideTitle();
        titleHidden = true;
      }
      if (phaseElapsed >= INTRO_TITLE_SEC) {
        panStart = { x: scene.camera.position.x, y: scene.camera.position.y };
        zoomStart = scene.viewSize;
        advance('intro-zoom-out');
      }
      return;
    }

    if (phase === 'intro-zoom-out') {
      currentFrame = introLastSnap;
      const t = Math.min(1, phaseElapsed / INTRO_ZOOM_OUT_SEC);
      const e = easeInOut(t);
      const targetSize = lerp(zoomStart, initialViewSize, e);
      if (Math.abs(targetSize - scene.viewSize) > 1e-3) setViewSize(scene, targetSize);
      const cam = scene.camera.position;
      const wantX = lerp(panStart.x, 0, e);
      const wantY = lerp(panStart.y, 0, e);
      panBy(scene, wantX - cam.x, wantY - cam.y);
      clampCamera(scene);
      if (t >= 1) {
        sceneIdx = 0;
        kickScene(sceneIdx);
      }
      return;
    }

    // Scene phases: playback uses a single `sceneElapsed` counter that runs
    // across caption-in + hold + caption-out, so snapshot sampling is monotonic
    // across the whole 5s window.
    sceneElapsed += dt;
    const spec = SCENES[sceneIdx];
    const ps = presims[sceneIdx];
    const sample = sampleSnapshot(ps.snapshots, ps.expectedLength, sceneElapsed, spec.durationSec);
    if (sample !== null) currentFrame = sample;

    if (phase === 'scene-caption-in') {
      if (!phaseEntered) {
        overlay.showCaption(spec.caption);
        phaseEntered = true;
      }
      if (phaseElapsed >= CAPTION_FADE_SEC) advance('scene-hold');
      return;
    }

    if (phase === 'scene-hold') {
      const holdFor = spec.durationSec - 2 * CAPTION_FADE_SEC;
      if (phaseElapsed >= holdFor) {
        overlay.hideCaption();
        advance('scene-caption-out');
      }
      return;
    }

    if (phase === 'scene-caption-out') {
      if (phaseElapsed >= CAPTION_FADE_SEC) {
        sceneIdx = (sceneIdx + 1) % SCENES.length;
        kickScene(sceneIdx);
      }
      return;
    }
  }

  function advance(next: Phase): void {
    phase = next;
    phaseElapsed = 0;
    phaseEntered = false;
  }

  function kickScene(idx: number): void {
    sceneElapsed = 0;
    advance('scene-caption-in');
    // If the pre-sim hasn't finished yet, playback falls back to whatever's
    // already in snapshots (possibly empty → currentFrame stays on previous).
    void presims[idx].ready;
  }

  async function presimScene(p: ScenePresim): Promise<void> {
    const champion = await loadSceneChampion(p.spec.label);
    setChampion(champion);
    p.snapshots.length = 0;
    p.expectedLength = PRESIM_TICK_BUDGET + 1;
    await presimGame(
      makeInitialState(undefined, undefined, NUM_PLAYERS),
      seats,
      PRESIM_TICK_BUDGET,
      p.snapshots,
    );
    // Lock in the actual recorded length so playback's t→idx mapping uses
    // the true span (winner-early-exits shrink it; otherwise unchanged).
    p.expectedLength = p.snapshots.length;
  }

  return {
    isActive: () => active,
    enter,
    tick,
    currentSnapshot: () => currentFrame,
    currentScene: () => active ? SCENES[sceneIdx] : null,
  };
}

async function presimGame(
  initial: GameState,
  seats: AIName[],
  tickBudget: number,
  out: GameState[],
): Promise<void> {
  let s = initial;
  out.push(s);
  for (let t = 0; t < tickBudget; t++) {
    s = step(s, TICK_DT);
    if (t % AI_PERIOD_TICKS === 0) {
      for (let p = 0; p < s.numPlayers; p++) {
        const fn = AIs[seats[p]];
        for (const a of fn(s, p, s.tick)) s = applyAction(s, a);
      }
    }
    if (SNAPSHOT_STRIDE === 1 || t % SNAPSHOT_STRIDE === 0) out.push(s);
    if (winnerOf(s) !== null) break;
    if ((t + 1) % PRESIM_BATCH === 0) await new Promise((r) => setTimeout(r, 0));
  }
}

function sampleSnapshot(
  snapshots: GameState[],
  expectedLength: number,
  elapsed: number,
  duration: number,
): GameState | null {
  if (snapshots.length === 0) return null;
  const t = Math.max(0, Math.min(1, elapsed / duration));
  const target = Math.floor(t * expectedLength);
  const idx = Math.min(snapshots.length - 1, target);
  return snapshots[idx];
}

function winnerOf(s: GameState): number | null {
  let alive = -1;
  for (let p = 0; p < s.numPlayers; p++) {
    let has = false;
    for (const n of s.nodes) if (n.owner === p) { has = true; break; }
    if (has) {
      if (alive === -1) alive = p;
      else return null;
    }
  }
  return alive >= 0 ? alive : null;
}

function defaultSeats(n: number): AIName[] {
  const out: AIName[] = Array(n).fill('evolved' as AIName);
  out[0] = 'aggressive';
  return out;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

export function pickHotArea(state: GameState): { x: number; y: number } {
  let sx = 0, sy = 0, wSum = 0;
  let sxAll = 0, syAll = 0, wAll = 0;
  for (const f of state.flows) {
    const a = state.nodes[f.src].pos;
    const b = state.nodes[f.dst].pos;
    const mx = (a.x + b.x) * 0.5;
    const my = (a.y + b.y) * 0.5;
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const dstOwner = state.nodes[f.dst].owner;
    const isCross = dstOwner !== null && dstOwner !== f.player;
    sxAll += mx * len;
    syAll += my * len;
    wAll += len;
    if (isCross) {
      sx += mx * len;
      sy += my * len;
      wSum += len;
    }
  }
  if (wSum > 0) return { x: sx / wSum, y: sy / wSum };
  if (wAll > 0) return { x: sxAll / wAll, y: syAll / wAll };
  return { x: 0, y: 0 };
}
