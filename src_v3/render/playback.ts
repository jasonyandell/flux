/**
 * Bottom-of-screen transport bar: prev / play-pause / next, time scrubber,
 * frame counter, speed cycle. Standard media-player layout.
 *
 * Pure DOM. The bar pulls state from the player each frame via setFrame /
 * setPaused / setSpeed and pushes user actions back through PlaybackHandlers.
 */
export type PlaybackHandlers = {
  onTogglePlay(): void;
  onPrev(): void;
  onNext(): void;
  onStepBack(): void;
  onStepForward(): void;
  // fraction in [0,1]; called on each scrub event while user drags
  onSeek(fraction: number): void;
  // user picked a new speed from the cycle
  onSpeedChange(multiplier: number): void;
};

export type PlaybackBar = {
  root: HTMLDivElement;
  setFrame(idx: number, total: number): void;
  setPaused(paused: boolean): void;
  setSpeed(multiplier: number): void;
  setEnabled(enabled: boolean): void;
};

const SPEEDS = [0.25, 0.5, 1, 2, 4];

function fmtSpeed(m: number): string {
  // 1 → "1×", 0.5 → "0.5×", 2 → "2×"
  const s = Number.isInteger(m) ? m.toFixed(0) : m.toString();
  return `${s}×`;
}

function makeButton(label: string, title: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  b.title = title;
  b.style.cssText =
    'background:transparent;border:none;color:#ddd;font:inherit;font-size:16px;' +
    'line-height:1;padding:6px 8px;cursor:pointer;border-radius:4px;' +
    'opacity:0.85;transition:opacity 0.15s ease, background 0.15s ease;';
  b.addEventListener('mouseenter', () => { b.style.background = 'rgba(255,255,255,0.08)'; b.style.opacity = '1'; });
  b.addEventListener('mouseleave', () => { b.style.background = 'transparent'; b.style.opacity = '0.85'; });
  return b;
}

export function createPlaybackBar(handlers: PlaybackHandlers): PlaybackBar {
  const root = document.createElement('div');
  root.id = 'flux-v2-playback';
  root.style.cssText =
    'position:fixed;left:50%;bottom:max(12px,env(safe-area-inset-bottom));' +
    'transform:translateX(-50%);display:flex;align-items:center;gap:10px;' +
    'padding:6px 12px;background:rgba(20,20,28,0.78);border:1px solid #2a2a3a;' +
    'border-radius:10px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;' +
    'color:#ddd;z-index:9;pointer-events:auto;user-select:none;-webkit-user-select:none;' +
    'opacity:0.55;transition:opacity 0.18s ease;max-width:calc(100vw - 24px);' +
    'box-shadow:0 4px 20px rgba(0,0,0,0.4);';
  root.addEventListener('mouseenter', () => { root.style.opacity = '1'; });
  root.addEventListener('mouseleave', () => { root.style.opacity = '0.55'; });

  const prev = makeButton('⏮', 'Previous replay (Shift+←)');
  prev.addEventListener('click', () => handlers.onPrev());

  const stepBack = makeButton('⏪', 'Step back one frame (←)');
  stepBack.addEventListener('click', () => handlers.onStepBack());

  const play = makeButton('▶', 'Play / Pause (Space)');
  play.addEventListener('click', () => handlers.onTogglePlay());

  const stepFwd = makeButton('⏩', 'Step forward one frame (→)');
  stepFwd.addEventListener('click', () => handlers.onStepForward());

  const next = makeButton('⏭', 'Next replay (Shift+→)');
  next.addEventListener('click', () => handlers.onNext());

  const scrubber = document.createElement('input');
  scrubber.type = 'range';
  scrubber.min = '0';
  scrubber.max = '1000';
  scrubber.step = '1';
  scrubber.value = '0';
  scrubber.style.cssText =
    'flex:1;min-width:160px;max-width:360px;height:14px;cursor:pointer;' +
    'accent-color:#4a90e2;';
  scrubber.title = 'Seek (drag to scrub, ←/→ to jog)';
  let userScrubbing = false;
  let pendingFromUser = false;
  scrubber.addEventListener('pointerdown', () => { userScrubbing = true; });
  scrubber.addEventListener('pointerup', () => { userScrubbing = false; });
  scrubber.addEventListener('input', () => {
    pendingFromUser = true;
    const t = Number(scrubber.value) / 1000;
    handlers.onSeek(t);
  });

  const counter = document.createElement('div');
  counter.style.cssText = 'font-variant-numeric:tabular-nums;opacity:0.8;min-width:8ch;text-align:right;';
  counter.textContent = '0 / 0';

  const speedBtn = makeButton(fmtSpeed(1), 'Playback speed (click to cycle)');
  speedBtn.style.minWidth = '4ch';
  speedBtn.style.textAlign = 'center';
  speedBtn.style.fontSize = '12px';
  let speedIdx = SPEEDS.indexOf(1);
  speedBtn.addEventListener('click', () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    const m = SPEEDS[speedIdx];
    speedBtn.textContent = fmtSpeed(m);
    handlers.onSpeedChange(m);
  });

  root.appendChild(prev);
  root.appendChild(stepBack);
  root.appendChild(play);
  root.appendChild(stepFwd);
  root.appendChild(next);
  root.appendChild(scrubber);
  root.appendChild(counter);
  root.appendChild(speedBtn);
  document.body.appendChild(root);

  return {
    root,
    setFrame(idx: number, total: number) {
      counter.textContent = `${idx} / ${Math.max(0, total - 1)}`;
      // Don't fight the user's drag — let go of programmatic updates while
      // they have the slider grabbed.
      if (!userScrubbing && !pendingFromUser) {
        const t = total > 1 ? idx / (total - 1) : 0;
        scrubber.value = String(Math.round(t * 1000));
      }
      // After one frame post-input, allow programmatic sync to resume.
      if (pendingFromUser && !userScrubbing) pendingFromUser = false;
    },
    setPaused(paused: boolean) {
      play.textContent = paused ? '▶' : '⏸';
      play.title = paused ? 'Play (Space)' : 'Pause (Space)';
    },
    setSpeed(multiplier: number) {
      const i = SPEEDS.indexOf(multiplier);
      if (i >= 0) speedIdx = i;
      speedBtn.textContent = fmtSpeed(multiplier);
    },
    setEnabled(enabled: boolean) {
      const o = enabled ? '' : '0.5';
      [prev, stepBack, play, stepFwd, next, speedBtn].forEach(b => { b.disabled = !enabled; });
      scrubber.disabled = !enabled;
      if (o) root.style.filter = `opacity(${o})`;
      else root.style.filter = '';
    },
  };
}
