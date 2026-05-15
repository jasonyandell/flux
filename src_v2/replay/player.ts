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
  // Transport controls (media-style playback).
  isPaused(): boolean;
  setPaused(p: boolean): void;
  togglePaused(): void;
  frameCount(): number;
  seekFrame(idx: number): void;
  seekFraction(t: number): void;
  stepFrames(delta: number): void;
  speedMultiplier(): number;
  setSpeedMultiplier(m: number): void;
  nextReplay(): void;
  prevReplay(): void;
  // Jump to a specific replay file (newest-first list selection).
  loadReplay(file: string): void;
};

export type Opts = {
  indexUrl: string;
  replayBaseUrl: string;
  pollIntervalMs: number;
  // Real-time playback rate in game ticks per wall-clock second. With the
  // trainer's dt_per_tick_ms=100, 10 means one physics tick every 0.1s.
  // Used as a fallback until the replay index has enough timestamps to
  // estimate the live trainer's drop cadence.
  playTicksPerSec: number;
  // Global multiplier applied to all computed playback rates (auto-cadence
  // and fallback). 2.0 = play everything twice as fast as real-time.
  playbackSpeed?: number;
};

export function createPlayer(opts: Opts): Player {
  let active = false;
  let replay: Replay | null = null;
  let replayName: string | null = null;
  let frameIdx = 0;
  let frameAccSec = 0;
  let lastPoll = 0;
  let pendingFile: string | null = null;
  let pendingIsShapeChange = false;
  let forceLoad = false;                  // user-requested switch (prev/next)
  let loading = false;
  let statusMsg = 'idle';
  let entriesCache: IndexEntry[] = [];   // newest-first
  let framesPerSec = 30;                  // default; recomputed per replay
  let manualSpeedOverride: number | null = null;
  // Replays that parsed to <2 frames — nothing to render. Skip on future polls.
  const skippedFiles = new Set<string>();
  const speedMult = opts.playbackSpeed ?? 1.0;
  // User-controllable transport multiplier, layered on top of the auto/manual
  // framesPerSec. 1.0 = whatever auto picked; 2.0 = double; 0.25 = quarter.
  let runtimeSpeedMult = 1.0;
  let paused = false;
  // Track which file we last saw at the top of the index so we only preempt
  // for *genuinely new* arrivals. Without this, a static newest entry with a
  // different board shape preempts every poll and the round-robin never
  // advances past it.
  let lastSeenNewestFile: string | null = null;

  function setStatus(s: string) { statusMsg = s; }

  function parsedSavedAt(e: IndexEntry | undefined): number | null {
    if (!e?.saved_at) return null;
    const ms = Date.parse(e.saved_at);
    return Number.isFinite(ms) ? ms : null;
  }

  function estimateReplayCadenceSec(file: string): number | null {
    const idx = entriesCache.findIndex(e => e.file === file);
    if (idx < 0) return null;
    const curMs = parsedSavedAt(entriesCache[idx]);
    if (curMs === null) return null;

    const candidates: number[] = [];
    const newerMs = parsedSavedAt(entriesCache[idx - 1]);
    if (newerMs !== null && newerMs > curMs) {
      candidates.push((newerMs - curMs) / 1000);
    }
    const olderMs = parsedSavedAt(entriesCache[idx + 1]);
    if (olderMs !== null && curMs > olderMs) {
      candidates.push((curMs - olderMs) / 1000);
    }
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => a - b);
    return candidates[Math.floor(candidates.length / 2)];
  }

  function autoFramesPerSec(r: Replay, file: string): { fps: number; targetSec: number | null } {
    const baselineFps =
      (opts.playTicksPerSec / Math.max(1, r.header.tickStride)) * speedMult;
    const cadenceSec = estimateReplayCadenceSec(file);
    if (cadenceSec !== null) {
      // Finish slightly before the next replay is expected so polling/load
      // jitter doesn't accumulate into a backlog. Cadence may legitimately
      // be short (live trainer) or unrealistically long (sparse one-off
      // artifacts in the index); only ever use it to speed playback UP, never
      // to slow it below the configured baseline tick rate.
      const targetSec = Math.max(1.0, cadenceSec * 0.9) / speedMult;
      const cadenceFps = Math.max(1, r.frames.length / targetSec);
      if (cadenceFps > baselineFps) {
        return { fps: cadenceFps, targetSec };
      }
    }
    return { fps: baselineFps, targetSec: null };
  }

  function pickNeighborReplay(delta: number): string | null {
    if (entriesCache.length === 0) return null;
    const usable = entriesCache.filter(e => e.file && !skippedFiles.has(e.file));
    if (usable.length === 0) return null;
    if (usable.length === 1) return usable[0].file;
    const curIdx = replayName
      ? usable.findIndex(e => e.file === replayName)
      : -1;
    if (curIdx < 0) return usable[0].file;
    const n = usable.length;
    const nextIdx = ((curIdx + delta) % n + n) % n;
    return usable[nextIdx].file;
  }

  function pickNextReplay(): string | null {
    return pickNeighborReplay(1);
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
      // Skip entries we already know are unplayable (parsed to <2 frames).
      const newest = entries.find(e => e.file && !skippedFiles.has(e.file));
      if (!newest || !newest.file) return;
      const isNewArrival = newest.file !== lastSeenNewestFile;
      lastSeenNewestFile = newest.file;
      if (newest.file === replayName) return;
      // Mid-playback preemption is only worth it when the newest entry is
      // brand new (live trainer drop). For a static index that's been sitting
      // at the same newest for a while, let the round-robin in advance() pick
      // it up at the natural end-of-replay boundary — otherwise we ping-pong
      // back to newest every poll instead of cycling through the catalog.
      // Also skip auto-preemption while paused; user expects a frozen frame.
      if (replay !== null && !isNewArrival) return;
      if (paused && replay !== null) return;
      pendingFile = newest.file;
      pendingIsShapeChange = replay !== null && (
        (typeof newest.radius === 'number' && newest.radius !== replay.header.radius)
        || (typeof newest.num_players === 'number' && newest.num_players !== replay.header.numPlayers)
      );
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
      const r = await parseReplay(buf);
      if (r.frames.length < 2) {
        // No-frames stub (header-only file from a crashed/aborted writer).
        // Mark it skipped so future polls walk past it to the next candidate.
        skippedFiles.add(file);
        setStatus(`skipped ${file} (${r.frames.length} frames)`);
        pendingFile = null;
        pendingIsShapeChange = false;
        forceLoad = false;
        loading = false;
        return;
      }
      replay = r;
      replayName = file;
      frameIdx = 0;
      frameAccSec = 0;
      pendingFile = null;
      pendingIsShapeChange = false;
      forceLoad = false;
      // Recompute auto-speed from the recorded stride so playback follows
      // the trainer's replay cadence once the index has timestamps.
      if (manualSpeedOverride !== null) {
        framesPerSec = manualSpeedOverride;
        setStatus(
          `playing ${file} (${r.frames.length} frames, manual ${framesPerSec.toFixed(1)} fps)`,
        );
      } else {
        const auto = autoFramesPerSec(r, file);
        framesPerSec = auto.fps;
        const target = auto.targetSec === null
          ? 'real time'
          : `~${auto.targetSec.toFixed(1)}s`;
        setStatus(
          `playing ${file} (${r.frames.length} frames over ${target}, ${framesPerSec.toFixed(1)} fps)`,
        );
      }
    } catch (err) {
      setStatus(`load error: ${(err as Error).message}`);
    } finally {
      loading = false;
    }
  }

  function effectiveFps(): number {
    return Math.max(0.1, framesPerSec * runtimeSpeedMult);
  }

  function advance(dt: number): void {
    if (!replay || paused) return;
    const wallSec = 1 / effectiveFps();
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
    recentEntries() { return entriesCache.slice(); },
    status() { return statusMsg; },
    setSpeed(speed: number) {
      manualSpeedOverride = speed;
      framesPerSec = speed;
    },
    isPaused() { return paused; },
    setPaused(p: boolean) {
      paused = p;
      // Drop fractional accumulator so unpause doesn't snap forward.
      frameAccSec = 0;
    },
    togglePaused() { this.setPaused(!paused); },
    frameCount() { return replay ? replay.frames.length : 0; },
    seekFrame(idx: number) {
      if (!replay) return;
      const max = replay.frames.length - 1;
      frameIdx = Math.max(0, Math.min(max, Math.floor(idx)));
      frameAccSec = 0;
    },
    seekFraction(t: number) {
      if (!replay) return;
      const max = replay.frames.length - 1;
      this.seekFrame(Math.round(Math.max(0, Math.min(1, t)) * max));
    },
    stepFrames(delta: number) {
      if (!replay) return;
      this.seekFrame(frameIdx + delta);
    },
    speedMultiplier() { return runtimeSpeedMult; },
    setSpeedMultiplier(m: number) {
      runtimeSpeedMult = Math.max(0.05, m);
      frameAccSec = 0;
    },
    nextReplay() {
      const f = pickNeighborReplay(1);
      if (f && f !== replayName) { pendingFile = f; forceLoad = true; }
    },
    prevReplay() {
      const f = pickNeighborReplay(-1);
      if (f && f !== replayName) { pendingFile = f; forceLoad = true; }
    },
    loadReplay(file: string) {
      if (!file || file === replayName) return;
      if (skippedFiles.has(file)) return;
      pendingFile = file;
      forceLoad = true;
    },
    tick(dt: number) {
      if (!active) return;
      const now = performance.now();
      void pollIndex(now);
      const atEnd = replay !== null && frameIdx + 1 >= replay.frames.length;
      // Auto-cycle to the next replay at end only while playing. User-driven
      // switches (forceLoad) or shape changes still load immediately.
      const shouldLoad =
        replay === null ||
        forceLoad ||
        pendingIsShapeChange ||
        (atEnd && !paused);
      if (pendingFile && !loading && shouldLoad) {
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
