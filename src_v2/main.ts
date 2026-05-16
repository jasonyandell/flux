/**
 * flux v2 — trainer-displayer entry.
 *
 * Reads .flxr v2 replays from /v2/replays/, auto-reloads on new replays,
 * and renders against the v1-style hex layout (same colors, no debug GUI).
 */
import { buildBoard } from './board';
import { createPlayer } from './replay/player';
import { createScene, updateScene, rebuildSceneGeometry, render, resizeRenderer, setFadeEnabled } from './render/scene';
import { createTopBar } from './render/topbar';
import { createPlaybackBar } from './render/playback';
import { createPlaylist } from './render/playlist';
import { createRunHeader } from './render/runHeader';
import { createEventsTailer } from './replay/events';

const REPLAY_BASE = '/v2/replays/';
const INDEX_URL = '/v2/replays/index.json';
const EVENTS_URL = '/v2/replays/events.jsonl';
const POLL_INTERVAL_MS = 3000;
const NEW_COUNT_POLL_MS = 3000;
const PLAY_TICKS_PER_SEC = 10; // 1x game time: dt_per_tick_ms is 100
const PLAYBACK_SPEED = 2.0;     // multiplier applied to all playback rates

const canvas = document.getElementById('app') as HTMLCanvasElement;

// Initial placeholder board (until first replay loads).
let board = buildBoard(5, 12);
const scene = createScene(canvas, board);
const topBar = createTopBar();
topBar.setStatus('waiting for first replay…');

window.addEventListener('resize', () => resizeRenderer(scene));
resizeRenderer(scene);

const player = createPlayer({
  indexUrl: INDEX_URL,
  replayBaseUrl: REPLAY_BASE,
  pollIntervalMs: POLL_INTERVAL_MS,
  playTicksPerSec: PLAY_TICKS_PER_SEC,
  playbackSpeed: PLAYBACK_SPEED,
});
const playlist = createPlaylist();
const runHeader = createRunHeader();

function replayParam(): string | null {
  const raw = new URLSearchParams(window.location.search).get('replay');
  return raw && raw.trim() ? raw.trim() : null;
}

function replayUrl(file: string): string {
  const url = new URL(window.location.href);
  url.searchParams.set('replay', file);
  return url.toString();
}

function copyReplayLink(file: string | null): void {
  if (!file) return;
  void navigator.clipboard?.writeText(replayUrl(file)).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = replayUrl(file);
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  });
  topBar.setStatus(`copied link ${file}`);
}

function replaceReplayUrl(file: string): void {
  const current = replayParam();
  if (current === file) return;
  window.history.replaceState(null, '', replayUrl(file));
}

playlist.setOnSelect((file) => {
  // Selecting from the playlist implies the user wants that specific run on
  // screen — unpause if needed and load it.
  if (player.isPaused()) player.setPaused(false);
  player.loadReplay(file);
  replaceReplayUrl(file);
});

playlist.setOnCopyLink((file) => copyReplayLink(file));
runHeader.setOnCopyLink(() => copyReplayLink(player.currentName()));

const requestedReplay = replayParam();
if (requestedReplay) {
  player.loadReplay(requestedReplay);
}

const LAST_CLOSED_KEY = 'flux-v2-playlist-last-closed';
function loadLastClosedMs(): number {
  try {
    const v = localStorage.getItem(LAST_CLOSED_KEY);
    if (v === null) return Date.now();
    const n = Number(v);
    return Number.isFinite(n) ? n : Date.now();
  } catch { return Date.now(); }
}

function saveLastClosedMs(ms: number): void {
  try { localStorage.setItem(LAST_CLOSED_KEY, String(ms)); } catch { /* ignore */ }
}

let lastClosedMs = loadLastClosedMs();
if (!localStorage.getItem(LAST_CLOSED_KEY)) saveLastClosedMs(lastClosedMs);
playlist.setNewSince(lastClosedMs);

playlist.setOnClose(() => {
  lastClosedMs = Date.now();
  saveLastClosedMs(lastClosedMs);
  playlist.setNewSince(lastClosedMs);
  playlist.setNewCount(0);
});

const eventsTailer = createEventsTailer(EVENTS_URL);
let lastEventsPoll = 0;
let eventsInFlight = false;

function pollNewCount(now: number): void {
  if (eventsInFlight) return;
  if (now - lastEventsPoll < NEW_COUNT_POLL_MS) return;
  lastEventsPoll = now;
  eventsInFlight = true;
  eventsTailer.fetchNewer(lastClosedMs)
    .then((events) => {
      if (playlist.isOpen()) return;
      playlist.setNewCount(events.length);
    })
    .catch(() => { /* transient; retry on next poll */ })
    .finally(() => { eventsInFlight = false; });
}

player.start();

function stepFrame(delta: number) {
  // Stepping a single frame implies the user wants the new frame held still.
  if (!player.isPaused()) player.setPaused(true);
  player.stepFrames(delta);
}

const FADE_STORAGE_KEY = 'flux-v2-fade-enabled';
function loadFadeEnabled(): boolean {
  try {
    const v = localStorage.getItem(FADE_STORAGE_KEY);
    return v === null ? true : v === '1';
  } catch { return true; }
}
function saveFadeEnabled(enabled: boolean): void {
  try { localStorage.setItem(FADE_STORAGE_KEY, enabled ? '1' : '0'); } catch { /* ignore */ }
}

const playbackBar = createPlaybackBar({
  onTogglePlay: () => player.togglePaused(),
  onPrev: () => player.prevReplay(),
  onNext: () => player.nextReplay(),
  onStepBack: () => stepFrame(-1),
  onStepForward: () => stepFrame(1),
  onSeek: (t) => {
    // Scrubbing implies user wants the frame frozen; pause if not already.
    if (!player.isPaused()) player.setPaused(true);
    player.seekFraction(t);
  },
  onSpeedChange: (m) => player.setSpeedMultiplier(m),
  onToggleFade: (enabled) => {
    setFadeEnabled(scene, enabled);
    saveFadeEnabled(enabled);
  },
});

// Restore the user's fade-toggle preference (default on).
{
  const initialFade = loadFadeEnabled();
  setFadeEnabled(scene, initialFade);
  playbackBar.setFadeEnabled(initialFade);
}

// Standard media-player keys: Space toggles, arrows jog by frame,
// Shift+arrows swap to the prev/next replay.
window.addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement | null;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
  if (e.key === ' ' || e.code === 'Space') {
    e.preventDefault();
    player.togglePaused();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    if (e.shiftKey) player.prevReplay();
    else stepFrame(-1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    if (e.shiftKey) player.nextReplay();
    else stepFrame(1);
  }
});

let last = performance.now();
let currentName: string | null = null;
let currentBoardKey = `${board.radius}:${board.numPlayers}:${board.N}`;

function boardKey(b: typeof board): string {
  return `${b.radius}:${b.numPlayers}:${b.N}`;
}

function frame(now: number) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;

  player.tick(dt);
  pollNewCount(now);

  // Surface live player status even before the first replay loads, so the
  // top bar shows polling / loading state to the user.
  topBar.setStatus(player.status());
  playlist.setEntries(player.recentEntries(), player.currentName());

  const r = player.current();
  if (r) {
    const name = player.currentName();
    if (name !== currentName) {
      if (name) replaceReplayUrl(name);
      const nextBoardKey = boardKey(r.board);
      // Replay swap: rebuild geometry if board shape differs from current.
      // Mixed radius streams are normal while experiments pivot, so compare
      // the full board signature instead of only the node count.
      if (nextBoardKey !== currentBoardKey || r.board.N !== scene.nodeCount) {
        rebuildSceneGeometry(scene, r.board);
        currentBoardKey = nextBoardKey;
      }
      board = r.board;
      currentName = name;
      runHeader.setReplay(name, r);
    }
    const idx = player.currentFrame();
    const f = r.frames[idx];
    if (f) {
      updateScene(scene, board, f, idx, dt);
      const meta = r.header.metadata as Record<string, unknown>;
      const it = typeof meta.iteration === 'number' ? meta.iteration : 0;
      const fit = typeof meta.best_fitness === 'number' ? meta.best_fitness : 0;
      const mdl = typeof meta.model === 'string' ? meta.model : null;
      topBar.setStats(it, fit, mdl);
      topBar.setBoard(
        r.header.radius,
        r.header.numPlayers,
        r.header.numNodes,
        r.header.tickStride,
        r.header.dtPerTickMs,
      );
    }
  } else {
    runHeader.setReplay(null, null);
  }

  playbackBar.setFrame(player.currentFrame(), player.frameCount());
  playbackBar.setPaused(player.isPaused());
  playbackBar.setSpeed(player.speedMultiplier());

  render(scene);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
