/**
 * Top bar — shows iter/gen + a drip-feed of the last few incoming replays.
 *
 * Stripped from src/render/gameui.ts: no restart/evolve buttons.
 */
export type TopBar = {
  root: HTMLDivElement;
  setStats: (iteration: number, fitness: number, model?: string | null) => void;
  setRecent: (entries: { file: string; iteration?: number; saved_at?: string }[]) => void;
  setStatus: (status: string) => void;
};

export function createTopBar(): TopBar {
  const root = document.createElement('div');
  root.id = 'flux-v2-topbar';
  root.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);' +
    'display:flex;align-items:baseline;gap:14px;white-space:nowrap;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:0.5px;' +
    'color:#ddd;pointer-events:none;z-index:8;max-width:100vw;';

  const tag = document.createElement('div');
  tag.style.cssText = 'opacity:0.55;letter-spacing:0.8px;';
  tag.textContent = 'flux v2';
  root.appendChild(tag);

  const stats = document.createElement('div');
  stats.style.cssText = 'opacity:0.55;font-variant-numeric:tabular-nums;min-width:18ch;text-align:left;';
  stats.textContent = 'iter 0';
  root.appendChild(stats);

  const recent = document.createElement('div');
  recent.style.cssText = 'opacity:0.35;font-variant-numeric:tabular-nums;display:flex;gap:8px;max-width:50vw;overflow:hidden;';
  recent.textContent = '';
  root.appendChild(recent);

  const status = document.createElement('div');
  status.style.cssText = 'opacity:0.35;font-variant-numeric:tabular-nums;';
  status.textContent = 'idle';
  root.appendChild(status);

  document.body.appendChild(root);

  return {
    root,
    setStats(iteration, fitness, model) {
      const m = model ? `[${model}] ` : '';
      const f = Number.isFinite(fitness) ? fitness.toFixed(2) : '–';
      stats.textContent = `${m}iter ${iteration} · R ${f}`;
    },
    setRecent(entries) {
      const items = entries.slice(0, 3).map(e => {
        const it = (e.iteration ?? 0).toString();
        return `<span style="opacity:0.8">i${it}</span>`;
      });
      recent.innerHTML = items.join(' · ');
    },
    setStatus(s) { status.textContent = s; },
  };
}
