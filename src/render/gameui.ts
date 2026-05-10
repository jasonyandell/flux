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

