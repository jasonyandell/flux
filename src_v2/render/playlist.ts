/**
 * Playlist — a hamburger-toggled side panel with a vertical, info-rich list
 * of recent replays. Decoupled from the top bar (which now only shows
 * playback stats) and from the playback transport (which now only drives
 * the currently-loaded replay).
 */
export type PlaylistEntry = {
  file: string;
  iteration?: number;
  generation?: number;
  saved_at?: string;
  kind?: string;
  radius?: number;
  num_players?: number;
};

export type Playlist = {
  setEntries(entries: PlaylistEntry[], currentFile: string | null): void;
  setOnSelect(handler: (file: string) => void): void;
  // Display "+N" badge on the hamburger and flash if N increased.
  setNewCount(n: number): void;
  // Called when the panel closes — caller persists "last seen" timestamp.
  setOnClose(handler: () => void): void;
  open(): void;
  close(): void;
  isOpen(): boolean;
};

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
  if (typeof e.generation === 'number' && e.generation > 0
      && (typeof e.iteration !== 'number' || e.iteration === 0)) {
    // already shown as primary if no iter
  } else if (typeof e.generation === 'number' && e.generation > 0) {
    parts.push(`g${e.generation}`);
  }
  return parts;
}

export function createPlaylist(): Playlist {
  // Hamburger button — top-left, unobtrusive, doesn't fight the canvas.
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.id = 'flux-v2-playlist-toggle';
  btn.setAttribute('aria-label', 'Open playlist');
  btn.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));' +
    'left:max(10px,env(safe-area-inset-left));' +
    'width:36px;height:36px;display:flex;flex-direction:column;' +
    'align-items:center;justify-content:center;gap:4px;' +
    'background:rgba(20,20,28,0.72);border:1px solid #2a2a3a;border-radius:8px;' +
    'cursor:pointer;z-index:10;opacity:0.62;pointer-events:auto;' +
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

  // "+N" arrival badge — top-right of the hamburger. Hidden when N=0.
  const badge = document.createElement('div');
  badge.style.cssText =
    'position:absolute;top:-6px;right:-6px;min-width:18px;height:18px;' +
    'padding:0 5px;display:none;align-items:center;justify-content:center;' +
    'background:#4a90e2;color:#fff;font:inherit;font-size:10px;font-weight:600;' +
    'line-height:1;border-radius:9px;box-shadow:0 1px 4px rgba(0,0,0,0.4);' +
    'pointer-events:none;font-variant-numeric:tabular-nums;';
  badge.textContent = '';
  btn.style.position = 'fixed';
  btn.appendChild(badge);

  // Flash keyframes injected once. The pulse lasts ~0.9s so it reads at the
  // edge of vision without being annoying when new replays land in bursts.
  if (!document.getElementById('flux-v2-playlist-flash-style')) {
    const style = document.createElement('style');
    style.id = 'flux-v2-playlist-flash-style';
    style.textContent =
      '@keyframes flux-v2-playlist-flash {' +
      '  0%   { box-shadow: 0 0 0 0 rgba(74,144,226,0.6), 0 2px 12px rgba(0,0,0,0.35); }' +
      '  60%  { box-shadow: 0 0 0 10px rgba(74,144,226,0.0), 0 2px 12px rgba(0,0,0,0.35); }' +
      '  100% { box-shadow: 0 0 0 0 rgba(74,144,226,0.0), 0 2px 12px rgba(0,0,0,0.35); }' +
      '}' +
      '.flux-v2-playlist-flash { animation: flux-v2-playlist-flash 0.9s ease-out 1; }';
    document.head.appendChild(style);
  }

  btn.addEventListener('mouseenter', () => { btn.style.opacity = '1'; btn.style.background = 'rgba(40,40,52,0.85)'; });
  btn.addEventListener('mouseleave', () => { btn.style.opacity = '0.62'; btn.style.background = 'rgba(20,20,28,0.72)'; });
  document.body.appendChild(btn);

  // Dim backdrop — click-to-close, fades in when panel opens.
  const backdrop = document.createElement('div');
  backdrop.style.cssText =
    'position:fixed;inset:0;background:rgba(0,0,0,0.32);z-index:11;' +
    'opacity:0;pointer-events:none;transition:opacity 0.2s ease;';
  document.body.appendChild(backdrop);

  // Panel — slides in from left.
  const panel = document.createElement('div');
  panel.id = 'flux-v2-playlist-panel';
  panel.style.cssText =
    'position:fixed;top:0;left:0;height:100vh;height:100dvh;' +
    'width:min(340px,86vw);background:rgba(14,14,20,0.94);' +
    'border-right:1px solid #2a2a3a;z-index:12;' +
    'display:flex;flex-direction:column;color:#ddd;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;' +
    'transform:translateX(-100%);transition:transform 0.22s ease;' +
    'backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
    'box-shadow:4px 0 24px rgba(0,0,0,0.4);' +
    'padding-top:env(safe-area-inset-top);' +
    'padding-left:env(safe-area-inset-left);';

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
  closeBtn.setAttribute('aria-label', 'Close playlist');
  closeBtn.textContent = '×';
  closeBtn.style.cssText =
    'background:transparent;border:none;color:#ddd;font:inherit;' +
    'font-size:22px;line-height:1;padding:0 6px;cursor:pointer;opacity:0.75;';
  closeBtn.addEventListener('mouseenter', () => { closeBtn.style.opacity = '1'; });
  closeBtn.addEventListener('mouseleave', () => { closeBtn.style.opacity = '0.75'; });
  header.appendChild(closeBtn);
  panel.appendChild(header);

  const summary = document.createElement('div');
  summary.style.cssText =
    'padding:8px 16px;opacity:0.55;font-size:11px;border-bottom:1px solid #1a1a24;flex:0 0 auto;';
  summary.textContent = '';
  panel.appendChild(summary);

  const list = document.createElement('div');
  list.style.cssText =
    'flex:1 1 auto;overflow-y:auto;overflow-x:hidden;padding:6px 0;' +
    'scrollbar-width:thin;';
  panel.appendChild(list);

  document.body.appendChild(panel);

  let isOpen = false;
  let onSelect: (file: string) => void = () => {};
  let onClose: () => void = () => {};
  let lastSignature = '';
  let lastCurrent: string | null = null;
  let lastEntries: PlaylistEntry[] = [];
  let lastNewCount = 0;

  function setOpen(open: boolean) {
    isOpen = open;
    if (open) {
      panel.style.transform = 'translateX(0)';
      backdrop.style.opacity = '1';
      backdrop.style.pointerEvents = 'auto';
      btn.setAttribute('aria-label', 'Close playlist');
      // Re-render in case entries changed while closed; scroll current into view.
      render(lastEntries, lastCurrent, /* force */ true);
      scrollCurrentIntoView();
    } else {
      panel.style.transform = 'translateX(-100%)';
      backdrop.style.opacity = '0';
      backdrop.style.pointerEvents = 'none';
      btn.setAttribute('aria-label', 'Open playlist');
      // Hand control of "last seen" back to the caller. We don't clear the
      // badge here — the caller will set newCount=0 on the next tick after
      // it updates its lastClosedAt cursor.
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
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isOpen) {
      e.preventDefault();
      setOpen(false);
    }
  });

  function render(entries: PlaylistEntry[], currentFile: string | null, force = false) {
    const sig = entries.map(e => `${e.file}:${e.iteration ?? ''}:${e.saved_at ?? ''}`).join('|') + `#${currentFile ?? ''}`;
    if (!force && sig === lastSignature && currentFile === lastCurrent) return;
    lastSignature = sig;
    lastCurrent = currentFile;
    lastEntries = entries;

    list.replaceChildren();
    summary.textContent = entries.length === 0
      ? 'no replays yet'
      : `${entries.length} replay${entries.length === 1 ? '' : 's'}`;
    if (entries.length === 0) return;

    const now = Date.now();
    for (const e of entries) {
      const row = document.createElement('button');
      row.type = 'button';
      const isCurrent = e.file === currentFile;
      if (isCurrent) row.dataset.current = '1';
      row.style.cssText =
        'display:flex;flex-direction:column;align-items:flex-start;gap:3px;' +
        'width:100%;text-align:left;padding:10px 16px;background:transparent;' +
        'border:none;border-left:3px solid ' + (isCurrent ? '#4a90e2' : 'transparent') + ';' +
        'color:#ddd;font:inherit;cursor:pointer;' +
        'transition:background 0.12s ease;' +
        'opacity:' + (isCurrent ? '1' : '0.85') + ';';
      row.addEventListener('mouseenter', () => { row.style.background = 'rgba(74,144,226,0.10)'; });
      row.addEventListener('mouseleave', () => { row.style.background = 'transparent'; });

      const top = document.createElement('div');
      top.style.cssText = 'display:flex;align-items:baseline;gap:8px;width:100%;';
      const primary = document.createElement('span');
      primary.style.cssText = 'font-size:13px;font-variant-numeric:tabular-nums;' +
        (isCurrent ? 'color:#9ec5ff;' : '');
      primary.textContent = entryPrimary(e);
      top.appendChild(primary);
      if (e.kind) {
        const kind = document.createElement('span');
        kind.style.cssText =
          'font-size:10px;letter-spacing:0.3px;opacity:0.65;' +
          'padding:1px 6px;border:1px solid #2a2a3a;border-radius:3px;';
        kind.textContent = e.kind;
        top.appendChild(kind);
      }
      row.appendChild(top);

      const sub = entrySecondaryParts(e, now);
      if (sub.length > 0) {
        const sec = document.createElement('div');
        sec.style.cssText = 'opacity:0.55;font-size:11px;';
        sec.textContent = sub.join(' · ');
        row.appendChild(sec);
      }

      const fileLine = document.createElement('div');
      fileLine.style.cssText =
        'opacity:0.32;font-size:10px;white-space:nowrap;' +
        'overflow:hidden;text-overflow:ellipsis;width:100%;';
      fileLine.textContent = e.file;
      row.appendChild(fileLine);

      row.addEventListener('click', () => {
        onSelect(e.file);
        setOpen(false);
      });
      list.appendChild(row);
    }
  }

  return {
    setEntries(entries, currentFile) {
      // Only do full work when the panel is open or when contents change in a
      // way that affects the "current" badge. We still keep lastEntries fresh
      // so re-opening shows the latest list.
      lastEntries = entries;
      if (isOpen) render(entries, currentFile);
      else {
        lastCurrent = currentFile;
      }
    },
    setOnSelect(handler) { onSelect = handler; },
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
      // Flash only when N grew — refreshes shouldn't re-animate the button.
      if (increased) {
        btn.classList.remove('flux-v2-playlist-flash');
        // Force reflow so re-adding the class restarts the animation.
        void btn.offsetWidth;
        btn.classList.add('flux-v2-playlist-flash');
      }
    },
    open() { setOpen(true); },
    close() { setOpen(false); },
    isOpen() { return isOpen; },
  };
}
