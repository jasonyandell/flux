/**
 * Replay player — polls the index, loads new replays as they arrive, and
 * advances frames against wall-clock dt.
 *
 * Mirrors src/replay/player.ts but for v2 binary format and v2-only state.
 */
import { parseReplay, type Replay } from './format';

export type IndexEntry = {
  file: string;
  saved_at?: string;
  kind?: string;
  iteration?: number;
  generation?: number;
  radius?: number;
  num_players?: number;
};

export type Player = {
  current(): Replay | null;
  currentFrame(): number;
  currentName(): string | null;
  recentEntries(): IndexEntry[];
  status(): string;
  start(): void;
  stop(): void;
  tick(dt: number): void;
  setSpeed(speedFramesPerSec: number): void;
};

export type Opts = {
  indexUrl: string;
  replayBaseUrl: string;
  pollIntervalMs: number;
  // Auto speed: target wall-clock seconds to play each replay end-to-end.
  // Effective speed is computed per replay so we always finish in about
  // `targetPlaySec` seconds.
  targetPlaySec: number;
};

export function createPlayer(opts: Opts): Player {
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
  let framesPerSec = 30;                  // default; recomputed per replay
  let manualSpeedOverride: number | null = null;

  function setStatus(s: string) { statusMsg = s; }

  function pickNextReplay(): string | null {
    if (entriesCache.length === 0) return null;
    if (entriesCache.length === 1) return entriesCache[0].file;
    const curIdx = replayName
      ? entriesCache.findIndex(e => e.file === replayName)
      : -1;
    if (curIdx < 0) return entriesCache[0].file;
    const nextIdx = (curIdx + 1) % entriesCache.length;
    return entriesCache[nextIdx].file;
  }

  async function pollIndex(now: number): Promise<void> {
    if (now - lastPoll < opts.pollIntervalMs) return;
    lastPoll = now;
    try {
      const res = await fetch(`${opts.indexUrl}?t=${Date.now()}`, { cache: 'no-cache' });
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
      // Recompute auto-speed: play full replay in `targetPlaySec`.
      if (manualSpeedOverride !== null) {
        framesPerSec = manualSpeedOverride;
      } else {
        framesPerSec = Math.max(1, r.frames.length / Math.max(0.5, opts.targetPlaySec));
      }
      setStatus(`playing ${file} (${r.frames.length} frames @ ${framesPerSec.toFixed(0)} fps)`);
    } catch (err) {
      setStatus(`load error: ${(err as Error).message}`);
    } finally {
      loading = false;
    }
  }

  function advance(dt: number): void {
    if (!replay) return;
    const wallSec = 1 / Math.max(framesPerSec, 0.1);
    frameAccSec += dt;
    while (frameAccSec >= wallSec) {
      frameAccSec -= wallSec;
      if (frameIdx + 1 < replay.frames.length) {
        frameIdx++;
      } else {
        if (!pendingFile) {
          const next = pickNextReplay();
          if (next && next !== replayName) pendingFile = next;
        }
        frameAccSec = 0;
        break;
      }
    }
  }

  return {
    current() { return replay; },
    currentFrame() { return frameIdx; },
    currentName() { return replayName; },
    recentEntries() { return entriesCache.slice(0, 8); },
    status() { return statusMsg; },
    setSpeed(speed: number) {
      manualSpeedOverride = speed;
      framesPerSec = speed;
    },
    tick(dt: number) {
      if (!active) return;
      const now = performance.now();
      void pollIndex(now);
      const atEnd = replay !== null && frameIdx + 1 >= replay.frames.length;
      if (pendingFile && !loading && (replay === null || atEnd)) {
        void loadPending();
      }
      advance(dt);
    },
    start() {
      active = true;
      lastPoll = 0;
      setStatus('starting');
    },
    stop() {
      active = false;
      setStatus('stopped');
    },
  };
}
