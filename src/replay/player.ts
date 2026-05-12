import { applyFrame, parseReplay, type Replay } from './format';
import type { GameState } from '../game/state';

type IndexEntry = {
  file: string;
  saved_at?: string;
  kind?: string;
  ticks?: number;
  radius?: number;
  num_players?: number;
};

export type ReplayPlayer = {
  /** Currently rendered state, or null if no replay is loaded yet. */
  current(): GameState | null;
  /** Identifier of the replay being played, or null. */
  currentName(): string | null;
  /** Frame index within the current replay (0..numFrames-1), or -1. */
  currentFrame(): number;
  /** Parsed metadata from the current replay (generation, fitness, etc.). */
  metadata(): Record<string, unknown> | null;
  /** Advance playback by wall-clock dt (seconds). Idempotent if no replay loaded. */
  tick(dt: number): void;
  /** Begin polling for new replays and play them. */
  start(): void;
  /** Stop polling and pause playback. */
  stop(): void;
  /** Status string for HUD. */
  status(): string;
  /** Switch to a different index URL (e.g. greatest-hits.json). Resets cache. */
  setIndexUrl(url: string): void;
};

export type ReplayPlayerOpts = {
  indexUrl: string;          // e.g. '/replays/index.json'
  replayBaseUrl: string;     // e.g. '/replays/'
  speed: number | (() => number);  // wall-clock multiplier; pass a getter for live updates
  pollIntervalMs: number;    // index poll cadence
};

export function createReplayPlayer(opts: ReplayPlayerOpts): ReplayPlayer {
  let active = false;
  let replay: Replay | null = null;
  let replayName: string | null = null;
  let frameIdx = 0;
  let frameAccSec = 0;
  let lastPoll = 0;
  let pendingFile: string | null = null;
  let loading = false;
  let statusMsg = 'idle';
  let entriesCache: IndexEntry[] = [];   // newest-first
  let indexUrl = opts.indexUrl;

  function setStatus(s: string) { statusMsg = s; }

  function pickNextReplay(): string | null {
    if (entriesCache.length === 0) return null;
    if (entriesCache.length === 1) return entriesCache[0].file;
    const curIdx = replayName
      ? entriesCache.findIndex(e => e.file === replayName)
      : -1;
    if (curIdx < 0) return entriesCache[0].file;
    // Walk to the next-older entry; wrap back to newest when we hit the end.
    const nextIdx = (curIdx + 1) % entriesCache.length;
    return entriesCache[nextIdx].file;
  }

  async function pollIndex(now: number): Promise<void> {
    if (now - lastPoll < opts.pollIntervalMs) return;
    lastPoll = now;
    try {
      const res = await fetch(`${indexUrl}?t=${Date.now()}`, { cache: 'no-cache' });
      if (!res.ok) { setStatus(`index http ${res.status}`); return; }
      const entries = await res.json() as IndexEntry[];
      if (!Array.isArray(entries) || entries.length === 0) {
        setStatus('no replays yet');
        return;
      }
      entriesCache = entries;
      const newest = entries[0];
      if (!newest || !newest.file) return;
      if (newest.file === replayName) return;
      // A newer replay appeared — interrupt and play it.
      pendingFile = newest.file;
    } catch (err) {
      setStatus(`index error: ${(err as Error).message}`);
    }
  }

  async function loadPending(): Promise<void> {
    if (!pendingFile || loading) return;
    const file = pendingFile;
    loading = true;
    setStatus(`loading ${file}`);
    try {
      const res = await fetch(`${opts.replayBaseUrl}${file}?t=${Date.now()}`, { cache: 'no-cache' });
      if (!res.ok) { setStatus(`replay http ${res.status}`); loading = false; return; }
      const buf = await res.arrayBuffer();
      const r = parseReplay(buf);
      replay = r;
      replayName = file;
      frameIdx = 0;
      frameAccSec = 0;
      pendingFile = null;
      setStatus(`playing ${file} (${r.header.numFrames} frames)`);
    } catch (err) {
      setStatus(`load error: ${(err as Error).message}`);
    } finally {
      loading = false;
    }
  }

  function advance(dt: number): void {
    if (!replay) return;
    const h = replay.header;
    // game-time per recorded frame in seconds
    const frameGameTimeSec = (h.tickStride * h.dtPerTickMs) / 1000;
    const speed = typeof opts.speed === 'function' ? opts.speed() : opts.speed;
    // wall-clock seconds per frame after speed multiplier
    const frameWallSec = frameGameTimeSec / Math.max(speed, 0.0001);
    frameAccSec += dt;
    while (frameAccSec >= frameWallSec) {
      frameAccSec -= frameWallSec;
      if (frameIdx + 1 < replay.frames.length) {
        frameIdx++;
      } else {
        // End of replay — try to advance to the next entry in the cached
        // index (walking back through history). If it's a different file
        // from current, queue it; otherwise hold on the last frame.
        const next = pickNextReplay();
        if (next && next !== replayName) {
          pendingFile = next;
        }
        frameAccSec = 0;
        break;
      }
    }
    applyFrame(replay.base, replay.frames[frameIdx]);
    replay.base.tick = frameIdx * h.tickStride;
  }

  return {
    current() { return replay ? replay.base : null; },
    currentName() { return replayName; },
    currentFrame() { return replay ? frameIdx : -1; },
    metadata() { return replay ? replay.header.metadata : null; },
    tick(dt: number) {
      if (!active) return;
      const now = performance.now();
      // Fire-and-forget index polling; never block frame.
      void pollIndex(now);
      // If we have a pending file and aren't loading, start loading.
      if (pendingFile && !loading) void loadPending();
      advance(dt);
    },
    start() {
      active = true;
      lastPoll = 0; // force immediate poll
      setStatus('starting');
    },
    stop() {
      active = false;
      setStatus('stopped');
    },
    status() { return statusMsg; },
    setIndexUrl(url: string) {
      if (url === indexUrl) return;
      indexUrl = url;
      // Force a re-poll on next tick; drop the current replay so we pick a
      // fresh entry from the new index instead of hanging on the old file.
      entriesCache = [];
      replayName = null;
      replay = null;
      pendingFile = null;
      frameIdx = 0;
      frameAccSec = 0;
      lastPoll = 0;
      setStatus('switching index');
    },
  };
}
