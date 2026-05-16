/**
 * Top bar — playback stats only (iter, fitness, board signature, status).
 *
 * The interactive replay list lives in the playlist panel
 * (see `src_v2/render/playlist.ts`), reachable via the hamburger button.
 */
export type TopBar = {
  root: HTMLDivElement;
  setStats: (iteration: number, fitness: number, model?: string | null) => void;
  setBoard: (
    radius: number,
    numPlayers: number,
    numNodes: number,
    tickStride: number,
    dtPerTickMs: number,
  ) => void;
  setStatus: (status: string) => void;
};

export function createTopBar(): TopBar {
  const root = document.createElement('div');
  root.id = 'flux-v2-topbar';
  root.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);' +
    'display:flex;align-items:baseline;justify-content:center;gap:12px;flex-wrap:wrap;' +
    'white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;' +
    'letter-spacing:0;color:#ddd;pointer-events:none;z-index:8;' +
    'width:min(1200px,calc(100vw - 120px));';

  const tag = document.createElement('div');
  tag.style.cssText = 'opacity:0.55;letter-spacing:0.8px;';
  tag.textContent = 'flux v2';
  root.appendChild(tag);

  const stats = document.createElement('div');
  stats.style.cssText = 'opacity:0.55;font-variant-numeric:tabular-nums;min-width:18ch;text-align:left;';
  stats.textContent = 'iter 0';
  root.appendChild(stats);

  const board = document.createElement('div');
  board.style.cssText = 'opacity:0.45;font-variant-numeric:tabular-nums;';
  board.textContent = 'r? · n? · stride ?';
  root.appendChild(board);

  const status = document.createElement('div');
  status.style.cssText = 'opacity:0.35;font-variant-numeric:tabular-nums;max-width:min(40vw,520px);overflow:hidden;text-overflow:ellipsis;';
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
    setBoard(radius, numPlayers, numNodes, tickStride, dtPerTickMs) {
      const ticksPerFrame = tickStride;
      const frameSec = (tickStride * dtPerTickMs) / 1000;
      board.textContent = `r${radius} · p${numPlayers} · n${numNodes} · ${ticksPerFrame}t/frame · ${frameSec.toFixed(1)}s`;
    },
    setStatus(s) { status.textContent = s; },
  };
}
