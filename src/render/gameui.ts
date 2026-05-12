import type { GameState, Player } from '../game/state';
import { COLORS } from './scene';

export type GameUI = {
  banner: HTMLDivElement;
  bannerText: HTMLDivElement;
  bannerButton: HTMLButtonElement;
  pause: HTMLDivElement;
  hint: HTMLDivElement;
  onPlayAgain: (cb: () => void) => void;
  onPauseClick: (cb: () => void) => void;
};

export function getWinner(state: GameState): Player | null {
  const counts = new Map<Player, number>();
  for (const n of state.nodes) {
    if (n.owner === null) continue;
    counts.set(n.owner, (counts.get(n.owner) ?? 0) + 1);
  }
  const owners = [...counts.entries()].filter(([, c]) => c > 0);
  if (owners.length !== 1) return null;
  return owners[0][0];
}

export function createGameUI(): GameUI {
  const banner = document.createElement('div');
  banner.style.cssText =
    'position:fixed;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;gap:24px;background:rgba(0,0,0,0.6);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#fff;z-index:10;';

  const bannerText = document.createElement('div');
  bannerText.style.cssText = 'font-size:64px;letter-spacing:4px;font-weight:bold;text-shadow:0 2px 8px rgba(0,0,0,0.8);';
  banner.appendChild(bannerText);

  const bannerButton = document.createElement('button');
  bannerButton.textContent = 'play again';
  bannerButton.style.cssText =
    'font:inherit;font-size:18px;padding:10px 24px;background:#222;color:#fff;border:1px solid #555;border-radius:4px;cursor:pointer;letter-spacing:2px;';
  bannerButton.onmouseenter = () => { bannerButton.style.background = '#333'; };
  bannerButton.onmouseleave = () => { bannerButton.style.background = '#222'; };
  banner.appendChild(bannerButton);

  document.body.appendChild(banner);

  const pause = document.createElement('div');
  pause.style.cssText =
    'position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(10,10,10,0.55);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#fff;font-size:56px;letter-spacing:6px;font-weight:bold;cursor:pointer;z-index:9;text-shadow:0 2px 8px rgba(0,0,0,0.8);';
  pause.textContent = 'PAUSED';
  document.body.appendChild(pause);

  const hint = document.createElement('div');
  hint.style.cssText =
    'position:fixed;bottom:12px;left:50%;transform:translateX(-50%);pointer-events:none;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#fff;opacity:0.35;letter-spacing:1px;transition:opacity 0.6s ease-out;z-index:5;';
  hint.textContent = 'click your node, then a neighbor to toggle a flow';
  document.body.appendChild(hint);

  return {
    banner,
    bannerText,
    bannerButton,
    pause,
    hint,
    onPlayAgain: (cb) => { bannerButton.onclick = cb; },
    onPauseClick: (cb) => { pause.onclick = cb; },
  };
}

export function showBanner(ui: GameUI, winner: Player, label?: string): void {
  ui.bannerText.textContent = label ? `${label.toUpperCase()} WINS` : `PLAYER ${winner} WINS`;
  ui.bannerText.style.fontSize = '64px';
  const hex = COLORS[winner % COLORS.length].toString(16).padStart(6, '0');
  ui.bannerText.style.color = `#${hex}`;
  ui.banner.style.display = 'flex';
}

export function showStasisBanner(ui: GameUI): void {
  ui.bannerText.textContent = 'STASIS';
  ui.bannerText.style.fontSize = '48px';
  ui.bannerText.style.color = '#999';
  ui.banner.style.display = 'flex';
}

export function hideBanner(ui: GameUI): void {
  ui.banner.style.display = 'none';
}

export function setPaused(ui: GameUI, paused: boolean): void {
  ui.pause.style.display = paused ? 'flex' : 'none';
}

export function fadeHint(ui: GameUI): void {
  ui.hint.style.opacity = '0';
}

export type TopBar = {
  root: HTMLDivElement;
  evolveBtn: HTMLButtonElement;
  evolveSub: HTMLSpanElement;
  stats: HTMLDivElement;
  setEvolveOn: (on: boolean) => void;
  setEvolveAvailable: (available: boolean, reason?: string) => void;
  setStats: (generation: number, best: number, model?: string | null) => void;
  setMode: (mode: 'normal' | 'replay') => void;
  onRestart: (cb: () => void) => void;
  onEvolveToggle: (cb: (next: boolean) => void) => void;
};

export function createTopBar(): TopBar {
  const root = document.createElement('div');
  root.id = 'flux-topbar';
  root.style.cssText =
    'position:fixed;top:max(10px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);' +
    'display:flex;align-items:baseline;gap:14px;white-space:nowrap;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:0.5px;' +
    'color:#ddd;pointer-events:none;z-index:8;max-width:100vw;';

  const btnCss =
    'font:inherit;letter-spacing:inherit;color:inherit;background:none;border:0;padding:6px 4px;' +
    'cursor:pointer;pointer-events:auto;opacity:0.45;transition:opacity 0.2s ease;' +
    'display:inline-flex;align-items:baseline;gap:6px;';

  const restartBtn = document.createElement('button');
  restartBtn.type = 'button';
  restartBtn.style.cssText = btnCss;
  restartBtn.textContent = '↻ restart';
  restartBtn.onmouseenter = () => { restartBtn.style.opacity = '0.95'; };
  restartBtn.onmouseleave = () => { restartBtn.style.opacity = '0.45'; };

  const evolveBtn = document.createElement('button');
  evolveBtn.type = 'button';
  evolveBtn.style.cssText =
    'font:inherit;letter-spacing:inherit;color:inherit;background:none;border:0;padding:4px 4px;' +
    'cursor:pointer;pointer-events:auto;opacity:0.6;transition:opacity 0.2s ease;' +
    'display:inline-flex;flex-direction:column;align-items:flex-start;gap:0;line-height:1.15;';
  const evolveTopRow = document.createElement('span');
  evolveTopRow.style.cssText = 'display:inline-flex;align-items:baseline;gap:6px;';
  const evolveDot = document.createElement('span');
  evolveDot.style.cssText = 'display:inline-block;width:7px;height:7px;border-radius:50%;background:transparent;border:1px solid currentColor;opacity:0.7;transform:translateY(1px);transition:background 0.2s ease, box-shadow 0.2s ease;';
  const evolveLabel = document.createElement('span');
  evolveLabel.style.cssText = 'display:inline-block;min-width:8ch;text-align:left;';
  evolveLabel.textContent = 'evolve';
  const evolveSub = document.createElement('span');
  evolveSub.style.cssText = 'font-size:10px;opacity:0.5;letter-spacing:0.4px;display:inline-block;min-width:14ch;text-align:left;';
  evolveSub.textContent = 'drains battery';
  evolveTopRow.appendChild(evolveDot);
  evolveTopRow.appendChild(evolveLabel);
  evolveBtn.appendChild(evolveTopRow);
  evolveBtn.appendChild(evolveSub);
  evolveBtn.onmouseenter = () => { if (!evolveBtn.disabled) evolveBtn.style.opacity = '0.95'; };
  evolveBtn.onmouseleave = () => { if (!evolveBtn.disabled) evolveBtn.style.opacity = '0.6'; };

  const stats = document.createElement('div');
  stats.style.cssText = 'opacity:0.4;white-space:nowrap;min-width:22ch;text-align:left;font-variant-numeric:tabular-nums;';
  stats.textContent = 'gen 0';

  root.appendChild(restartBtn);
  root.appendChild(evolveBtn);
  root.appendChild(stats);
  document.body.appendChild(root);

  let evolveOn = false;
  const applyEvolveStyle = () => {
    if (evolveBtn.disabled) {
      evolveBtn.style.opacity = '0.25';
      evolveBtn.style.cursor = 'not-allowed';
      evolveDot.style.background = 'transparent';
      evolveDot.style.boxShadow = 'none';
      evolveLabel.textContent = 'evolve';
    } else if (evolveOn) {
      evolveBtn.style.opacity = '0.85';
      evolveBtn.style.cursor = 'pointer';
      evolveDot.style.background = '#e2c44a';
      evolveDot.style.boxShadow = '0 0 6px rgba(226,196,74,0.6)';
      evolveLabel.textContent = 'evolving';
    } else {
      evolveBtn.style.opacity = '0.6';
      evolveBtn.style.cursor = 'pointer';
      evolveDot.style.background = 'transparent';
      evolveDot.style.boxShadow = 'none';
      evolveLabel.textContent = 'evolve';
    }
  };

  return {
    root, evolveBtn, evolveSub, stats,
    setMode: (mode) => {
      const showButtons = mode === 'normal';
      restartBtn.style.display = showButtons ? '' : 'none';
      evolveBtn.style.display = showButtons ? '' : 'none';
    },
    setEvolveOn: (on) => { evolveOn = on; applyEvolveStyle(); },
    setEvolveAvailable: (available, reason) => {
      evolveBtn.disabled = !available;
      evolveBtn.title = available ? '' : (reason ?? 'unavailable');
      evolveSub.textContent = available ? 'drains battery' : (reason ?? 'unavailable');
      applyEvolveStyle();
    },
    setStats: (generation, best, model) => {
      const modelTag = model ? `[${model}] ` : '';
      stats.textContent = generation > 0
        ? `${modelTag}gen ${generation} · best ${best}`
        : `${modelTag}gen 0`;
    },
    onRestart: (cb) => { restartBtn.onclick = cb; },
    onEvolveToggle: (cb) => { evolveBtn.onclick = () => { if (!evolveBtn.disabled) cb(!evolveOn); }; },
  };
}

