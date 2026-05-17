/**
 * Searchable replay playlist.
 *
 * The viewer produces a lot of run artifacts. This panel keeps the canvas
 * clean while making runs addressable, filterable, and copy-linkable.
 */
export type PlaylistEntry = {
  file: string;
  iteration?: number;
  generation?: number;
  saved_at?: string;
  kind?: string;
  radius?: number;
  num_players?: number;
  ruleset?: string;
  edge_alpha?: number;
  model?: string;
  winner?: number;
  ticks?: number;
  seat_solvers?: string[];
};

export type Playlist = {
  setEntries(entries: PlaylistEntry[], currentFile: string | null): void;
  setOnSelect(handler: (file: string) => void): void;
  setOnCopyLink(handler: (file: string) => void): void;
  setNewCount(n: number): void;
  setNewSince(ms: number): void;
  setOnClose(handler: () => void): void;
  open(): void;
  close(): void;
  isOpen(): boolean;
};

type Filter = 'all' | 'solver' | 'train' | 'fluid' | 'new' | 'current';

function fmtRelativeMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

function entryPrimary(e: PlaylistEntry): string {
  if (typeof e.iteration === 'number' && e.iteration > 0) return `iter ${e.iteration}`;
  if (typeof e.generation === 'number' && e.generation > 0) return `gen ${e.generation}`;
  if (typeof e.radius === 'number' && typeof e.num_players === 'number') {
    return `r${e.radius} · p${e.num_players}`;
  }
  return e.file;
}

function entrySecondaryParts(e: PlaylistEntry, now: number): string[] {
  const parts: string[] = [];
  if (e.saved_at) {
    const t = Date.parse(e.saved_at);
    if (Number.isFinite(t)) {
      const rel = fmtRelativeMs(now - t);
      if (rel) parts.push(rel);
    }
  }
  if (typeof e.radius === 'number' && typeof e.num_players === 'number') {
    parts.push(`r${e.radius}p${e.num_players}`);
  }
  if (typeof e.edge_alpha === 'number') parts.push(`ea ${e.edge_alpha}`);
  if (e.ruleset) parts.push(e.ruleset);
  if (typeof e.winner === 'number' && e.winner >= 0) parts.push(`winner ${e.winner}`);
  if (typeof e.ticks === 'number' && e.ticks > 0) parts.push(`${e.ticks}t`);
  return parts;
}

function searchableText(e: PlaylistEntry): string {
  return [
    e.file,
    e.kind ?? '',
    e.ruleset ?? '',
    e.model ?? '',
    ...(e.seat_solvers ?? []),
    typeof e.radius === 'number' ? `r${e.radius}` : '',
    typeof e.num_players === 'number' ? `p${e.num_players}` : '',
    typeof e.edge_alpha === 'number' ? `ea${String(e.edge_alpha).replace('.', 'p')}` : '',
  ].join(' ').toLowerCase();
}

function passesFilter(e: PlaylistEntry, filter: Filter, currentFile: string | null, newSinceMs: number): boolean {
  if (filter === 'current') return e.file === currentFile;
  if (filter === 'solver') return e.kind === 'solver_v2' || e.file.startsWith('solver_');
  if (filter === 'train') return e.kind === 'train_v2' || e.file.startsWith('train_');
  if (filter === 'fluid') return (e.ruleset ?? '').includes('fluid') || typeof e.edge_alpha === 'number' && e.edge_alpha < 1;
  if (filter === 'new') {
    if (!e.saved_at || !Number.isFinite(newSinceMs)) return false;
    const ms = Date.parse(e.saved_at);
    return Number.isFinite(ms) && ms > newSinceMs;
  }
  return true;
}

export function createPlaylist(): Playlist {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.id = 'flux-v2-playlist-toggle';
  btn.setAttribute('aria-label', 'Open replay browser');
  btn.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));' +
    'left:max(10px,env(safe-area-inset-left));' +
    'width:36px;height:36px;display:flex;flex-direction:column;' +
    'align-items:center;justify-content:center;gap:4px;' +
    'background:rgba(20,20,28,0.72);border:1px solid #2a2a3a;border-radius:8px;' +
    'cursor:pointer;z-index:10;opacity:0.72;pointer-events:auto;' +
    'transition:opacity 0.18s ease, background 0.18s ease;' +
    'backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);' +
    'box-shadow:0 2px 12px rgba(0,0,0,0.35);';
  const bars = document.createElement('div');
  bars.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:4px;';
  for (let i = 0; i < 3; i++) {
    const bar = document.createElement('div');
    bar.style.cssText = 'width:16px;height:2px;background:#ddd;border-radius:1px;';
    bars.appendChild(bar);
  }
  btn.appendChild(bars);

  const badge = document.createElement('div');
  badge.style.cssText =
    'position:absolute;top:-6px;right:-6px;min-width:18px;height:18px;' +
    'padding:0 5px;display:none;align-items:center;justify-content:center;' +
    'background:#4a90e2;color:#fff;font:inherit;font-size:10px;font-weight:600;' +
    'line-height:1;border-radius:9px;box-shadow:0 1px 4px rgba(0,0,0,0.4);' +
    'pointer-events:none;font-variant-numeric:tabular-nums;';
  btn.appendChild(badge);

  if (!document.getElementById('flux-v2-playlist-flash-style')) {
    const style = document.createElement('style');
    style.id = 'flux-v2-playlist-flash-style';
    style.textContent =
      '@keyframes flux-v2-playlist-flash {' +
      '0%{box-shadow:0 0 0 0 rgba(74,144,226,0.6),0 2px 12px rgba(0,0,0,0.35);}' +
      '60%{box-shadow:0 0 0 10px rgba(74,144,226,0),0 2px 12px rgba(0,0,0,0.35);}' +
      '100%{box-shadow:0 0 0 0 rgba(74,144,226,0),0 2px 12px rgba(0,0,0,0.35);}' +
      '}' +
      '.flux-v2-playlist-flash{animation:flux-v2-playlist-flash 0.9s ease-out 1;}';
    document.head.appendChild(style);
  }

  btn.addEventListener('mouseenter', () => { btn.style.opacity = '1'; btn.style.background = 'rgba(40,40,52,0.85)'; });
  btn.addEventListener('mouseleave', () => { btn.style.opacity = '0.72'; btn.style.background = 'rgba(20,20,28,0.72)'; });
  document.body.appendChild(btn);

  const backdrop = document.createElement('div');
  backdrop.style.cssText =
    'position:fixed;inset:0;background:rgba(0,0,0,0.32);z-index:11;' +
    'opacity:0;pointer-events:none;transition:opacity 0.2s ease;';
  document.body.appendChild(backdrop);

  const panel = document.createElement('div');
  panel.id = 'flux-v2-playlist-panel';
  panel.setAttribute('aria-hidden', 'true');
  panel.style.cssText =
    'position:fixed;top:0;left:0;height:100vh;height:100dvh;' +
    'width:min(420px,92vw);background:rgba(14,14,20,0.96);' +
    'border-right:1px solid #2a2a3a;z-index:12;' +
    'display:flex;flex-direction:column;color:#ddd;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;' +
    'transform:translateX(-100%);transition:transform 0.22s ease;' +
    'backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
    'box-shadow:4px 0 24px rgba(0,0,0,0.4);' +
    'padding-top:env(safe-area-inset-top);padding-left:env(safe-area-inset-left);' +
    'visibility:hidden;pointer-events:none;';

  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;align-items:center;justify-content:space-between;' +
    'padding:14px 16px;border-bottom:1px solid #22222e;flex:0 0 auto;';
  const title = document.createElement('div');
  title.style.cssText = 'font-size:13px;letter-spacing:0.5px;opacity:0.85;';
  title.textContent = 'replays';
  header.appendChild(title);
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.setAttribute('aria-label', 'Close replay browser');
  closeBtn.textContent = 'x';
  closeBtn.style.cssText =
    'background:transparent;border:none;color:#ddd;font:inherit;' +
    'font-size:18px;line-height:1;padding:0 6px;cursor:pointer;opacity:0.75;';
  header.appendChild(closeBtn);
  panel.appendChild(header);

  const tools = document.createElement('div');
  tools.style.cssText = 'padding:10px 16px 8px;border-bottom:1px solid #1a1a24;display:flex;flex-direction:column;gap:8px;';
  const search = document.createElement('input');
  search.type = 'search';
  search.placeholder = 'Search filename, solver, ruleset...';
  search.setAttribute('aria-label', 'Search replays');
  search.style.cssText =
    'width:100%;box-sizing:border-box;background:#101018;border:1px solid #2a2a3a;' +
    'border-radius:6px;color:#eee;font:inherit;padding:8px 9px;outline:none;';
  tools.appendChild(search);

  const chipRow = document.createElement('div');
  chipRow.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';
  const chips: Array<[Filter, string]> = [
    ['all', 'All'],
    ['solver', 'Solver'],
    ['train', 'Train'],
    ['fluid', 'Fluid'],
    ['new', 'New'],
    ['current', 'Current'],
  ];
  let activeFilter: Filter = 'all';
  const chipButtons = new Map<Filter, HTMLButtonElement>();
  for (const [key, label] of chips) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.textContent = label;
    chip.style.cssText =
      'border:1px solid #2a2a3a;border-radius:999px;background:transparent;' +
      'color:#bbb;font:inherit;font-size:11px;padding:4px 8px;cursor:pointer;';
    chip.addEventListener('click', () => {
      activeFilter = key;
      updateChipStyles();
      render(lastEntries, lastCurrent, true);
    });
    chipButtons.set(key, chip);
    chipRow.appendChild(chip);
  }
  tools.appendChild(chipRow);
  panel.appendChild(tools);

  const summary = document.createElement('div');
  summary.style.cssText =
    'padding:8px 16px;opacity:0.58;font-size:11px;border-bottom:1px solid #1a1a24;flex:0 0 auto;';
  panel.appendChild(summary);

  const list = document.createElement('div');
  list.style.cssText =
    'flex:1 1 auto;overflow-y:auto;overflow-x:hidden;padding:6px 0;scrollbar-width:thin;';
  panel.appendChild(list);

  document.body.appendChild(panel);

  let isOpen = false;
  let onSelect: (file: string) => void = () => {};
  let onCopyLink: (file: string) => void = () => {};
  let onClose: () => void = () => {};
  let lastSignature = '';
  let lastCurrent: string | null = null;
  let lastEntries: PlaylistEntry[] = [];
  let lastNewCount = 0;
  let newSinceMs = 0;

  function updateChipStyles() {
    for (const [key, chip] of chipButtons) {
      const active = key === activeFilter;
      chip.style.background = active ? '#263852' : 'transparent';
      chip.style.color = active ? '#fff' : '#bbb';
      chip.style.borderColor = active ? '#4a90e2' : '#2a2a3a';
    }
  }
  updateChipStyles();

  function setOpen(open: boolean) {
    isOpen = open;
    if (open) {
      panel.style.visibility = 'visible';
      panel.style.pointerEvents = 'auto';
      panel.style.transform = 'translateX(0)';
      panel.setAttribute('aria-hidden', 'false');
      backdrop.style.opacity = '1';
      backdrop.style.pointerEvents = 'auto';
      btn.setAttribute('aria-label', 'Close replay browser');
      render(lastEntries, lastCurrent, true);
      scrollCurrentIntoView();
      search.focus();
    } else {
      panel.style.transform = 'translateX(-100%)';
      panel.style.pointerEvents = 'none';
      panel.setAttribute('aria-hidden', 'true');
      backdrop.style.opacity = '0';
      backdrop.style.pointerEvents = 'none';
      btn.setAttribute('aria-label', 'Open replay browser');
      window.setTimeout(() => {
        if (!isOpen) panel.style.visibility = 'hidden';
      }, 240);
      onClose();
    }
  }

  function scrollCurrentIntoView() {
    const cur = list.querySelector('[data-current="1"]') as HTMLElement | null;
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }

  btn.addEventListener('click', () => setOpen(!isOpen));
  closeBtn.addEventListener('click', () => setOpen(false));
  backdrop.addEventListener('click', () => setOpen(false));
  search.addEventListener('input', () => render(lastEntries, lastCurrent, true));
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isOpen) {
      e.preventDefault();
      setOpen(false);
    }
  });

  function filteredEntries(entries: PlaylistEntry[], currentFile: string | null): PlaylistEntry[] {
    const q = search.value.trim().toLowerCase();
    return entries.filter((e) => {
      if (!passesFilter(e, activeFilter, currentFile, newSinceMs)) return false;
      if (!q) return true;
      return searchableText(e).includes(q);
    });
  }

  function render(entries: PlaylistEntry[], currentFile: string | null, force = false) {
    const sig = entries.map(e => `${e.file}:${e.iteration ?? ''}:${e.saved_at ?? ''}`).join('|') +
      `#${currentFile ?? ''}#${search.value}#${activeFilter}`;
    if (!force && sig === lastSignature && currentFile === lastCurrent) return;
    lastSignature = sig;
    lastCurrent = currentFile;
    lastEntries = entries;

    list.replaceChildren();
    const shown = filteredEntries(entries, currentFile);
    summary.textContent = `${shown.length}/${entries.length} replay${entries.length === 1 ? '' : 's'} shown`;
    if (shown.length === 0) return;

    const now = Date.now();
    for (const e of shown) {
      const row = document.createElement('div');
      const isCurrent = e.file === currentFile;
      if (isCurrent) row.dataset.current = '1';
      row.style.cssText =
        'display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;' +
        'width:100%;box-sizing:border-box;padding:10px 12px 10px 16px;' +
        'border-left:3px solid ' + (isCurrent ? '#4a90e2' : 'transparent') + ';' +
        'opacity:' + (isCurrent ? '1' : '0.88') + ';';

      const runBtn = document.createElement('button');
      runBtn.type = 'button';
      runBtn.setAttribute('aria-label', `Load ${e.file}`);
      runBtn.style.cssText =
        'display:flex;flex-direction:column;align-items:flex-start;gap:3px;width:100%;' +
        'min-width:0;text-align:left;background:transparent;border:none;color:#ddd;' +
        'font:inherit;cursor:pointer;padding:0;';
      runBtn.addEventListener('mouseenter', () => { row.style.background = 'rgba(74,144,226,0.10)'; });
      runBtn.addEventListener('mouseleave', () => { row.style.background = 'transparent'; });

      const top = document.createElement('div');
      top.style.cssText = 'display:flex;align-items:baseline;gap:8px;width:100%;';
      const primary = document.createElement('span');
      primary.style.cssText = 'font-size:13px;font-variant-numeric:tabular-nums;' + (isCurrent ? 'color:#9ec5ff;' : '');
      primary.textContent = entryPrimary(e);
      top.appendChild(primary);
      if (e.kind) {
        const kind = document.createElement('span');
        kind.style.cssText =
          'font-size:10px;letter-spacing:0.3px;opacity:0.65;padding:1px 6px;' +
          'border:1px solid #2a2a3a;border-radius:3px;';
        kind.textContent = e.kind;
        top.appendChild(kind);
      }
      runBtn.appendChild(top);

      const sub = entrySecondaryParts(e, now);
      if (sub.length > 0) {
        const sec = document.createElement('div');
        sec.style.cssText = 'opacity:0.58;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;';
        sec.textContent = sub.join(' · ');
        runBtn.appendChild(sec);
      }

      const fileLine = document.createElement('div');
      fileLine.style.cssText = 'opacity:0.35;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:100%;';
      fileLine.textContent = e.file;
      runBtn.appendChild(fileLine);
      runBtn.addEventListener('click', () => {
        onSelect(e.file);
        setOpen(false);
      });
      row.appendChild(runBtn);

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = 'link';
      copyBtn.setAttribute('aria-label', `Copy link for ${e.file}`);
      copyBtn.style.cssText =
        'background:rgba(74,144,226,0.10);border:1px solid #2a2a3a;border-radius:5px;' +
        'color:#b9d6ff;font:inherit;font-size:10px;padding:5px 7px;cursor:pointer;';
      copyBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        onCopyLink(e.file);
      });
      row.appendChild(copyBtn);
      list.appendChild(row);
    }
  }

  return {
    setEntries(entries, currentFile) {
      lastEntries = entries;
      if (isOpen) render(entries, currentFile);
      else lastCurrent = currentFile;
    },
    setOnSelect(handler) { onSelect = handler; },
    setOnCopyLink(handler) { onCopyLink = handler; },
    setNewSince(ms) {
      newSinceMs = ms;
      if (isOpen && activeFilter === 'new') render(lastEntries, lastCurrent, true);
    },
    setOnClose(handler) { onClose = handler; },
    setNewCount(n) {
      const clamped = Math.max(0, n | 0);
      if (clamped === lastNewCount) return;
      const increased = clamped > lastNewCount;
      lastNewCount = clamped;
      if (clamped === 0) {
        badge.style.display = 'none';
        badge.textContent = '';
        return;
      }
      badge.style.display = 'flex';
      badge.textContent = `+${clamped > 99 ? '99' : clamped}`;
      if (increased) {
        btn.classList.remove('flux-v2-playlist-flash');
        void btn.offsetWidth;
        btn.classList.add('flux-v2-playlist-flash');
      }
    },
    open() { setOpen(true); },
    close() { setOpen(false); },
    isOpen() { return isOpen; },
  };
}
