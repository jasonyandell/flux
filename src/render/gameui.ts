import type { GameState, Player } from '../game/state';

export type GameUI = {
  banner: HTMLDivElement;
  bannerText: HTMLDivElement;
  bannerButton: HTMLButtonElement;
  onPlayAgain: (cb: () => void) => void;
};

const HUMAN: Player = 0;

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

  return {
    banner,
    bannerText,
    bannerButton,
    onPlayAgain: (cb) => { bannerButton.onclick = cb; },
  };
}

export function showBanner(ui: GameUI, winner: Player): void {
  ui.bannerText.textContent = winner === HUMAN ? 'YOU WIN' : 'YOU LOSE';
  ui.bannerText.style.color = winner === HUMAN ? '#7fc7ff' : '#ff7f7f';
  ui.banner.style.display = 'flex';
}

export function hideBanner(ui: GameUI): void {
  ui.banner.style.display = 'none';
}
