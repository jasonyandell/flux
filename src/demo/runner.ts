import type { GameState } from '../game/state';
import { step } from '../game/step';
import type { SceneLabel } from './champions';
import { setChampion } from '../gpu/evolved';
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
  { label: 'gen0',    caption: 'watch ai battle',         durationSec: 5 },
  { label: 'gen100',  caption: 'the blue one is code',    durationSec: 5 },
  { label: 'gen200',  caption: 'the others are neural nets', durationSec: 5 },
  { label: 'gen1000', caption: 'watch them get smarter',  durationSec: 5 },
  { label: 'gen20k',  caption: 'watch them win',          durationSec: 5 },
];

export const DEMO_SPEED = 100;
const INTRO_FAST_FORWARD_TICKS = 150;
const INTRO_PAN_SEC = 1.0;
const INTRO_TITLE_SEC = 1.5;
const INTRO_ZOOM_OUT_SEC = 0.7;
const CAPTION_FADE_SEC = 0.6;
const HOT_VIEW_SIZE = 8;

export type DemoRunner = {
  isActive: () => boolean;
  enter: () => Promise<void>;
  tick: (dt: number) => void;
  currentScene: () => SceneSpec | null;
};

type LoadSceneFn = (label: SceneLabel) => Promise<void>;
type GetStateFn = () => GameState;

export function createRunner(opts: {
  scene: RenderScene;
  overlay: Overlay;
  getState: GetStateFn;
  loadScene: LoadSceneFn;
}): DemoRunner {
  const { scene, overlay, getState, loadScene } = opts;

  let active = false;
  let phase: Phase = 'intro-pan';
  let phaseElapsed = 0;
  let phaseEntered = false;
  let sceneIdx = 0;
  let initialViewSize = scene.viewSize;
  let hotTarget = { x: 0, y: 0 };
  let panStart = { x: 0, y: 0 };
  let zoomStart = scene.viewSize;
  let titleHidden = false;

  async function preloadIntro(): Promise<void> {
    // Fast-forward an off-screen copy of the current state with the gen0
    // champion (null → fresh random genome). The result is pushed back via
    // loadScene which respawns + applies the champion + sets caption.
    setChampion(null);
    let s = getState();
    for (let i = 0; i < INTRO_FAST_FORWARD_TICKS; i++) s = step(s, 0.1);
    // We can't write into the host's state directly from here, but the host's
    // `loadScene` callback for scene 0 handles a fresh respawn — the
    // fast-forward exists to find a hot area, not to display.
    hotTarget = pickHotArea(s);
  }

  async function enter(): Promise<void> {
    active = true;
    initialViewSize = scene.viewSize;
    await preloadIntro();
    panStart = { x: scene.camera.position.x, y: scene.camera.position.y };
    zoomStart = scene.viewSize;
    phase = 'intro-pan';
    phaseElapsed = 0;
    overlay.hideTitle();
    overlay.hideCaption();
  }

  function tick(dt: number): void {
    if (!active) return;
    phaseElapsed += dt;

    if (phase === 'intro-pan') {
      const t = Math.min(1, phaseElapsed / INTRO_PAN_SEC);
      const e = easeInOut(t);
      const cam = scene.camera.position;
      const targetSize = lerp(zoomStart, HOT_VIEW_SIZE, e);
      if (Math.abs(targetSize - scene.viewSize) > 1e-3) setViewSize(scene, targetSize);
      const wantX = lerp(panStart.x, hotTarget.x, e);
      const wantY = lerp(panStart.y, hotTarget.y, e);
      panBy(scene, wantX - cam.x, wantY - cam.y);
      clampCamera(scene);
      if (t >= 1) advance('intro-title');
      return;
    }

    if (phase === 'intro-title') {
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
      const t = Math.min(1, phaseElapsed / INTRO_ZOOM_OUT_SEC);
      const e = easeInOut(t);
      const cam = scene.camera.position;
      const targetSize = lerp(zoomStart, initialViewSize, e);
      if (Math.abs(targetSize - scene.viewSize) > 1e-3) setViewSize(scene, targetSize);
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

    if (phase === 'scene-caption-in') {
      if (phaseElapsed >= CAPTION_FADE_SEC) advance('scene-hold');
      return;
    }

    if (phase === 'scene-hold') {
      const holdFor = SCENES[sceneIdx].durationSec - 2 * CAPTION_FADE_SEC;
      if (phaseElapsed >= holdFor) {
        overlay.hideCaption();
        advance('scene-caption-out');
      }
      return;
    }

    if (phase === 'scene-caption-out') {
      if (phaseElapsed >= CAPTION_FADE_SEC) {
        const next = (sceneIdx + 1) % SCENES.length;
        sceneIdx = next;
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
    const spec = SCENES[idx];
    void loadScene(spec.label);
    overlay.showCaption(spec.caption);
    advance('scene-caption-in');
  }

  return {
    isActive: () => active,
    enter,
    tick,
    currentScene: () => active ? SCENES[sceneIdx] : null,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// Hot area: weighted centroid of cross-owner flow midpoints, weighted by
// segment length. Cross-owner flows are the visually interesting ones — they're
// where seats are attacking each other. Falls back to centroid of all flows,
// then to origin if there are no flows yet.
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
