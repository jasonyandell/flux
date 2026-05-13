/**
 * flux v2 — trainer-displayer entry.
 *
 * Reads .flxr v2 replays from /v2/replays/, auto-reloads on new replays,
 * and renders against the v1-style hex layout (same colors, no debug GUI).
 */
import { buildBoard } from './board';
import { createPlayer } from './replay/player';
import { createScene, updateScene, rebuildSceneGeometry, render, resizeRenderer } from './render/scene';
import { createTopBar } from './render/topbar';

const REPLAY_BASE = '/v2/replays/';
const INDEX_URL = '/v2/replays/index.json';
const POLL_INTERVAL_MS = 3000;
const TARGET_PLAY_SEC = 4;     // wall-clock seconds per replay

const canvas = document.getElementById('app') as HTMLCanvasElement;

// Initial placeholder board (until first replay loads).
let board = buildBoard(6, 6);
const scene = createScene(canvas, board);
const topBar = createTopBar();
topBar.setStatus('waiting for first replay…');

window.addEventListener('resize', () => resizeRenderer(scene));
resizeRenderer(scene);

const player = createPlayer({
  indexUrl: INDEX_URL,
  replayBaseUrl: REPLAY_BASE,
  pollIntervalMs: POLL_INTERVAL_MS,
  targetPlaySec: TARGET_PLAY_SEC,
});
player.start();

let last = performance.now();
let currentName: string | null = null;

function frame(now: number) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;

  player.tick(dt);

  // Surface live player status even before the first replay loads, so the
  // top bar shows polling / loading state to the user.
  topBar.setStatus(player.status());
  topBar.setRecent(player.recentEntries());

  const r = player.current();
  if (r) {
    const name = player.currentName();
    if (name !== currentName) {
      // Replay swap: rebuild geometry if board shape differs from current.
      if (r.board.N !== scene.nodeCount) {
        rebuildSceneGeometry(scene, r.board);
      }
      board = r.board;
      currentName = name;
    }
    const idx = player.currentFrame();
    const f = r.frames[idx];
    if (f) {
      updateScene(scene, board, f);
      const meta = r.header.metadata as Record<string, unknown>;
      const it = typeof meta.iteration === 'number' ? meta.iteration : 0;
      const fit = typeof meta.best_fitness === 'number' ? meta.best_fitness : 0;
      const mdl = typeof meta.model === 'string' ? meta.model : null;
      topBar.setStats(it, fit, mdl);
    }
  }

  render(scene);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
