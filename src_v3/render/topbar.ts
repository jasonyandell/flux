/**
 * Top bar — shows iter/gen + an interactive newest-first list of recent
 * replays, so any of the indexed runs (cap 50) can be jumped to with one
 * click, not just the freshest.
 *
 * Stripped from src/render/gameui.ts: no restart/evolve buttons.
 */
export type RecentEntry = {
  file: string;
  iteration?: number;
  saved_at?: string;
  kind?: string;
  radius?: number;
  num_players?: number;
};

export type TopBar = {
  root: HTMLDivElement;
  recentRoot: HTMLDivElement;
  setStats: (iteration: number, fitness: number, model?: string | null) => void;
  setBoard: (
    radius: number,
    numPlayers: number,
    numNodes: number,
    tickStride: number,
    dtPerTickMs: number,
  ) => void;
  setRecent: (entries: RecentEntry[], currentFile: string | null) => void;
  setOnSelect: (handler: (file: string) => void) => void;
  setStatus: (status: string) => void;
};

function fmtRelativeMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h`;
  const d = Math.round(h / 24);
  return `${d}d`;
}

function entryDigest(e: RecentEntry, now: number): string {
  const bits: string[] = [];
  if (e.saved_at) {
    const t = Date.parse(e.saved_at);
    if (Number.isFinite(t)) {
      const rel = fmtRelativeMs(now - t);
      if (rel) bits.push(`${rel} ago`);
    }
  }
  if (typeof e.radius === 'number') bits.push(`r${e.radius}`);
  if (typeof e.num_players === 'number') bits.push(`p${e.num_players}`);
  if (e.kind) bits.push(e.kind);
  return bits.join(' · ');
}

export function createTopBar(): TopBar {
  const root = document.createElement('div');
  root.id = 'flux-v2-topbar';
  root.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);' +
    'display:flex;flex-direction:column;align-items:center;gap:6px;white-space:nowrap;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:0;' +
    'color:#ddd;pointer-events:none;z-index:8;width:min(1200px,calc(100vw - 24px));';

  const headRow = document.createElement('div');
  headRow.style.cssText =
    'display:flex;align-items:baseline;justify-content:center;gap:12px;flex-wrap:wrap;';

  const tag = document.createElement('div');
  tag.style.cssText = 'opacity:0.55;letter-spacing:0.8px;';
  tag.textContent = 'flux v2';
  headRow.appendChild(tag);

  const stats = document.createElement('div');
  stats.style.cssText = 'opacity:0.55;font-variant-numeric:tabular-nums;min-width:18ch;text-align:left;';
  stats.textContent = 'iter 0';
  headRow.appendChild(stats);

  const board = document.createElement('div');
  board.style.cssText = 'opacity:0.45;font-variant-numeric:tabular-nums;';
  board.textContent = 'r? · n? · stride ?';
  headRow.appendChild(board);

  const status = document.createElement('div');
  status.style.cssText = 'opacity:0.35;font-variant-numeric:tabular-nums;max-width:min(40vw,520px);overflow:hidden;text-overflow:ellipsis;';
  status.textContent = 'idle';
  headRow.appendChild(status);

  root.appendChild(headRow);

  // Recent-runs strip — newest-first, horizontally scrollable so older runs
  // are reachable without losing the screen real estate the canvas needs.
  const recent = document.createElement('div');
  recent.id = 'flux-v2-recent';
  recent.style.cssText =
    'pointer-events:auto;display:flex;gap:6px;align-items:center;' +
    'max-width:100%;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;' +
    'padding:2px 4px;opacity:0.6;transition:opacity 0.15s ease;';
  recent.addEventListener('mouseenter', () => { recent.style.opacity = '1'; });
  recent.addEventListener('mouseleave', () => { recent.style.opacity = '0.6'; });
  root.appendChild(recent);

  document.body.appendChild(root);

  let onSelect: (file: string) => void = () => {};
  let lastSignature = '';
  let lastCurrent: string | null = null;

  function setRecent(entries: RecentEntry[], currentFile: string | null) {
    const sig = entries.map(e => `${e.file}:${e.iteration ?? ''}`).join('|');
    if (sig === lastSignature && currentFile === lastCurrent) return;
    lastSignature = sig;
    lastCurrent = currentFile;

    recent.replaceChildren();
    if (entries.length === 0) {
      const empty = document.createElement('span');
      empty.style.cssText = 'opacity:0.4;';
      empty.textContent = 'no replays yet';
      recent.appendChild(empty);
      return;
    }
    const now = Date.now();
    for (const e of entries) {
      const btn = document.createElement('button');
      btn.type = 'button';
      const isCurrent = e.file === currentFile;
      btn.style.cssText =
        'background:transparent;border:1px solid ' +
        (isCurrent ? '#4a90e2' : '#2a2a3a') + ';color:#ddd;' +
        'font:inherit;font-size:11px;line-height:1;padding:4px 7px;cursor:pointer;' +
        'border-radius:4px;opacity:' + (isCurrent ? '1' : '0.78') + ';' +
        'white-space:nowrap;transition:opacity 0.12s ease, background 0.12s ease;' +
        'font-variant-numeric:tabular-nums;flex:0 0 auto;';
      const iter = typeof e.iteration === 'number' ? `i${e.iteration}` : e.file;
      btn.textContent = iter;
      btn.title = `${e.file}${entryDigest(e, now) ? ' — ' + entryDigest(e, now) : ''}`;
      btn.addEventListener('mouseenter', () => {
        btn.style.background = 'rgba(255,255,255,0.08)';
        btn.style.opacity = '1';
      });
      btn.addEventListener('mouseleave', () => {
        btn.style.background = 'transparent';
        btn.style.opacity = isCurrent ? '1' : '0.78';
      });
      btn.addEventListener('click', () => onSelect(e.file));
      recent.appendChild(btn);
    }
  }

  return {
    root,
    recentRoot: recent,
    setStats(iteration, fitness, model) {
      const m = model ? `[${model}] ` : '';
      const f = Number.isFinite(fitness) ? fitness.toFixed(2) : '–';
      stats.textContent = `${m}iter ${iteration} · R ${f}`;
    },
    setBoard(radius, numPlayers, numNodes, tickStride, dtPerTickMs) {
      const ticksPerFrame = tickStride;
      const frameSec = (tickStride * dtPerTickMs) / 1000;
      board.textContent = `r${radius} · p${numPlayers} · n${numNodes} · ${ticksPerFrame}t/frame · ${frameSec.toFixed(1)}s`;
    },
    setRecent,
    setOnSelect(handler) { onSelect = handler; },
    setStatus(s) { status.textContent = s; },
  };
}
